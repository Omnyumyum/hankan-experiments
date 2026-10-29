"""사람 그림 / 아닌 그림 가르기 (HUMAN_ROUTE.md) — Meta 찾기 + 관절 모델을 돌려 신호를 모은다. 코드는 고치지 않고 재기만 한다.

    ../venv/bin/python -I eval_real/human_route_eval.py --hr eval_real/data/hr [--procs 4] [--only meta|qd|tub] [--n 0]
받은폴더(--hr):
  models/  drawn_humanoid_{detector,pose_estimator}/ (AnimatedDrawings v0.0.1 .mar 를 푼 곳)
  meta/    ann.json · x/amateur_drawings/... (tar 앞 1.6GB 를 푼 곳)          — 사람(H). CC BY-NC → 재기에만
  qd/      {category}.ndjson (앞 10,000,000 바이트)                            — yoga · angel = 사람(H), 동물 · 괴물 = 아님(N)
  tub/     TU-Berlin parquet 3개 (Hugging Face kmewhort/tu-berlin-png)        — person walking · sitting · angel = H, 동물 = N
애매(A): 서 있는 동물 · 사람 반쪽 (teddy-bear · monkey · kangaroo · mermaid · snowman · penguin · human-skeleton) — 어느 쪽이 맞다고 하지 않고 따로 센다.
나누기: 학습용(learn, 기준을 고를 때 봄) · 확인용(check, 기준을 정한 뒤 한 번). QuickDraw 는 그림마다 원본 + 사진처럼 바꾼 판(kid_photo, 무작위 한 가지) 두 줄.
결과: eval_real/data/hr/raw_<src>.csv (그림마다 찾기 점수 · 상자 · 관절 17개 · 마스크). 신호 계산과 기준 고르기는 human_route_fit.py.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import kid_photo as kp  # noqa: E402
from more_eval import ALL_EXPECT, TUB_NAME  # noqa: E402

QD_H = ["yoga", "angel"]
AMBIG = ["teddy-bear", "monkey", "kangaroo", "mermaid", "snowman", "penguin", "human-skeleton"]
TUB_H = ["person walking", "person sitting", "angel"]
QD_N = [c for c in ALL_EXPECT if c not in AMBIG]
N_QD = {"H": (60, 90), "N": (10, 15), "A": (0, 15)}  # (학습, 확인) 카테고리마다
N_TUB = {"H": (30, 50), "N": (8, 12), "A": (0, 12)}
N_META = (600, 1400)

MP = None


def init(models):
    global MP
    import torch
    torch.set_num_threads(1)
    from meta_pose import MetaPose
    MP = MetaPose(models)


def one(job):
    meta, img = job
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, tuple):  # ("qd", strokes, variant, seed) — 그리기도 일꾼에서
        _, strokes, variant, seed = img
        img, _ = kp.make(strokes, variant, seed)
    t0 = time.time()
    r = MP.run(img)
    row = dict(meta)
    row.update(h=r["shape"][0], w=r["shape"][1], det1=round(r["det"][0], 4) if r["det"] else 0,
               det2=round(r["det"][1], 4) if len(r["det"]) > 1 else 0, n_found=r.get("n_found", 0), box="", kp="",
               mask_box=0, kp_in_mask=0, t_sec=0)
    if r["box"] is not None:
        x0, y0, x1, y1 = r["box"]
        k = r["kp"]
        m = r["mask"]
        row["box"] = f"{x0} {y0} {x1} {y1}"
        row["kp"] = " ".join(f"{v:.1f}" if j < 2 else f"{v:.3f}" for p in k for j, v in enumerate(p))
        row["mask_box"] = round(float(m.mean()), 3)
        xi = np.clip(np.round(k[:, 0]).astype(int), 0, m.shape[1] - 1)
        yi = np.clip(np.round(k[:, 1]).astype(int), 0, m.shape[0] - 1)
        row["kp_in_mask"] = round(float(m[yi, xi].mean()), 3)
    row["t_sec"] = round(time.time() - t0, 2)
    return row


def load_qd(path, n):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue  # 받다 만 마지막 줄
            if r.get("recognized"):
                rows.append(r["drawing"])
    idx = np.random.default_rng(31).permutation(len(rows))
    return [rows[i] for i in idx[:n]]


def qd_jobs(hr):
    jobs = []
    rng = np.random.default_rng(5)
    for cat in QD_H + QD_N + [c for c in AMBIG if c != "human-skeleton"]:
        lab = "H" if cat in QD_H else "A" if cat in AMBIG else "N"
        nl, nc = N_QD[lab]
        dr = load_qd(hr / "qd" / f"{cat}.ndjson", nl + nc)
        for i, s in enumerate(dr):
            split = "learn" if i < nl else "check"
            for variant in ("orig", kp.VARIANTS[1 + rng.integers(len(kp.VARIANTS) - 1)]):
                seed = int(rng.integers(1 << 30))
                jobs.append((dict(src="qd", cat=cat, label=lab, split=split, id=i, variant=variant, file=""),
                             ("qd", s, variant, seed)))
    return jobs


def tub_jobs(hr):
    import pyarrow.parquet as pq
    names = {}
    for line in open(hr / "tub" / "README.md", encoding="utf-8"):
        line = line.strip()
        if line.startswith("'") and "':" in line:
            k, v = line.split("':", 1)
            names[int(k.strip("'"))] = v.strip()
    by = {}
    for p in sorted(glob.glob(str(hr / "tub" / "*.parquet"))):
        t = pq.read_table(p)
        for im, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            by.setdefault(names[lab], []).append(im["bytes"])
    jobs = []
    for cat in sorted(by):
        short = TUB_NAME.get(cat, cat)
        if cat in TUB_H:
            lab = "H"
        elif cat in AMBIG or short in AMBIG:
            lab = "A"
        elif short in QD_N or cat in TUB_NAME:
            lab = "N"
        else:
            continue  # 동물 · 사람이 아닌 물건은 안 씀
        nl, nc = N_TUB[lab]
        idx = np.random.default_rng(31).permutation(len(by[cat]))[: nl + nc]
        for i, j in enumerate(idx):
            g = cv2.imdecode(np.frombuffer(by[cat][j], np.uint8), cv2.IMREAD_COLOR)
            g = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)  # 1111px 가는 선 → 종이 사진 크기
            jobs.append((dict(src="tub", cat=cat, label=lab, split="learn" if i < nl else "check", id=int(j), variant="orig", file=""), g))
    return jobs


def meta_jobs(hr):
    ann = json.load(open(hr / "meta" / "ann.json"))
    names = {im["file_name"] for im in ann["images"]}
    fs = sorted(glob.glob(str(hr / "meta" / "x" / "amateur_drawings" / "*" / "*.png")))
    fs = [f for f in fs if f.split("/x/")[1] in names]
    idx = np.random.default_rng(31).permutation(len(fs))[: sum(N_META)]
    jobs = []
    for i, j in enumerate(idx):
        jobs.append((dict(src="meta", cat="human", label="H", split="learn" if i < N_META[0] else "check", id=int(j),
                          variant="photo", file=Path(fs[j]).name), fs[j]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", default="meta,qd,tub")
    ap.add_argument("--n", type=int, default=0, help="앞 n 개만 (시험용)")
    a = ap.parse_args()
    hr = Path(a.hr)
    for src in a.only.split(","):
        jobs = {"meta": meta_jobs, "qd": qd_jobs, "tub": tub_jobs}[src](hr)
        if a.n:
            jobs = jobs[:: max(1, len(jobs) // a.n)][: a.n]
        t0 = time.time()
        with Pool(a.procs, initializer=init, initargs=(str(hr / "models"),)) as p:
            rows = p.map(one, jobs, chunksize=2)
        out = hr / f"raw_{src}{'_n%d' % a.n if a.n else ''}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{src}: {len(rows)}줄, {time.time() - t0:.0f}초 → {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
