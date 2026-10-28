"""실제 아이 그림 사진(Meta Amateur Drawings)에서 마스크(그림 잘라 내기)만 잰다.

    준비:  venv/bin/python -I eval_real/photo_mask_eval.py prep --meta 받은폴더
    재기:  venv/bin/python -I eval_real/photo_mask_eval.py run --meta 받은폴더 --set learn|check|old_learn|old_check [--scale 3] [--procs 4] [--out 이름]
받은폴더: ann.json(annotation 전체)과 part.tar(앞부분만 받은 tar), x/(part.tar 를 푼 곳).
prep 은 tar 안 순서대로 그림을 세워 index.json 을 만든다. 나누기:
  old_learn 200장 · old_check 346장 — COLORED_ROBUST.md 가 쓴 앞 150MB 의 그림(그 CSV 의 파일 이름 그대로)
  learn — 그 밖의 그림 중 tar 순서 앞 1000장. 이번에 고칠 때 보는 것
  check — 나머지 전부. 이번 확인용. 고치기 전 숫자를 먼저 재 두고, 고치는 동안 열어 보지 않는다
결과: eval_real/photo_mask_<이름>.csv (Meta 는 파일 이름 · 숫자만), 화면에 요약.
Meta 데이터 annotation 의 licenses 는 CC BY-NC 4.0 → 재기에만 쓴다. 그림 · 모아 보기는 저장소에 올리지 않는다.
"""
import argparse
import csv
import json
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import colored_eval as ce  # noqa: E402
import rig_types as rt  # noqa: E402

SPLITS = ["old_learn", "old_check", "learn", "check"]
N_LEARN = 1000


def prep(meta):
    meta = Path(meta)
    ann = json.load(open(meta / "ann.json"))
    segs = {a["image_id"]: a["segmentation"] for a in ann["annotations"]}
    by_name = {im["file_name"]: im["id"] for im in ann["images"]}
    names = []
    try:
        with tarfile.open(meta / "part.tar") as tf:  # 앞부분만 받았으니 끝에서 끊긴다
            for m in tf:
                if m.isfile() and m.name.endswith(".png"):
                    names.append(m.name)
    except (tarfile.ReadError, EOFError, OSError):
        pass
    old = {}
    for k in ("learn", "check"):
        for r in csv.DictReader(open(HERE / f"colored_meta_{k}_before.csv", encoding="utf-8-sig")):
            old[r["file"]] = "old_" + k
    out = []
    n_new = 0
    for name in names:
        if name not in by_name or cv2.imread(str(meta / "x" / name)) is None:
            continue  # annotation 에 없는 그림 · 잘린 마지막 그림
        sp = old.get(Path(name).name)
        if sp is None:
            sp = "learn" if n_new < N_LEARN else "check"
            n_new += 1
        out.append(dict(i=len(out), file=name, split=sp, segs=segs[by_name[name]]))
    json.dump(out, open(meta / "index.json", "w"))
    print(f"tar 안 그림 {len(names)}장 중 쓸 수 있는 것 {len(out)}장 → {meta / 'index.json'}")
    for k in SPLITS:
        print(f"  {k}: {sum(o['split'] == k for o in out)}장")


def one(job):
    i, path, segs, scale = job
    img = cv2.imread(path)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gt = np.zeros(img.shape[:2], np.uint8)
    for s in segs:
        cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2) * scale).astype(np.int32)], 1)
    row = dict(i=i, file=Path(path).name, h=img.shape[0], w=img.shape[1], drawing_ok=0, iou="", mask_fail=1, t_sec="", notes="", err="")
    t0 = time.time()
    try:
        d = rt.Drawing(img)
    except Exception as e:
        row["t_sec"] = round(time.time() - t0, 3)
        row["err"] = repr(e)[:120]
        return row
    row["t_sec"] = round(time.time() - t0, 3)
    row["drawing_ok"] = 1
    row["notes"] = " / ".join(d.notes)
    iou = ce.mask_iou(img, d, gt > 0)
    row["iou"] = round(iou, 3)
    row["mask_fail"] = int(not iou >= 0.5)
    return row


def run(a):
    meta = Path(a.meta)
    idx = json.load(open(meta / "index.json"))
    items = [it for it in idx if it["split"] == a.set][: a.n or None]
    jobs = [(it["i"], str(meta / "x" / it["file"]), it["segs"], a.scale) for it in items]
    t0 = time.time()
    if a.procs == 1:
        rows = [one(j) for j in jobs]
    else:
        with Pool(a.procs) as p:
            rows = p.map(one, jobs, chunksize=4)
    name = a.out or f"{a.set}{'_x%g' % a.scale if a.scale != 1 else ''}"
    with open(HERE / f"photo_mask_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    ious = [r["iou"] for r in rows if r["iou"] != "" and r["iou"] == r["iou"]]
    ts = [r["t_sec"] for r in rows]
    mf = sum(r["mask_fail"] for r in rows)
    print(f"photo_mask_{name}: {n}장 ({time.time() - t0:.0f}초, 동시 {a.procs})")
    print(f"- Drawing 실패 {sum(1 - r['drawing_ok'] for r in rows)}, 마스크 실패(IoU<0.5) {mf} ({100 * mf / n:.1f}%), IoU 중앙값 {np.median(ious):.3f}")
    print(f"- 한 장 시간 평균 {np.mean(ts):.2f} · 90% {np.percentile(ts, 90):.2f} · 최대 {max(ts):.2f}초")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "run"])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--set", choices=SPLITS, default="learn")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    prep(a.meta) if a.mode == "prep" else run(a)


if __name__ == "__main__":
    main()
