"""색칠한 그림 마스크: 지금 방식(clean_photo + shape_rig.crop_and_mask)과 color_mask 를 같은 그림에서 나란히 잰다. 마스크만 잰다.

    Meta 사진:      venv/bin/python -I eval_real/color_mask_eval.py meta --meta 받은폴더 --set learn|check --method old|new|gate [--out 이름]
    QuickDraw 변환: venv/bin/python -I eval_real/color_mask_eval.py qd --qd 받은폴더 --set learn|check --method … [--n 20]
Meta: photo_mask_eval.py prep 의 index.json 과 color_sets.py 의 color_sets.json 을 쓴다 (학습용 1544장 · 확인용 2401장).
QuickDraw: colored_eval.py 와 같은 그림(30,000,000 바이트, seed 27, 카테고리마다 학습 20 · 확인 30) × 변환 VARIANTS.
방식: old = 지금 Drawing 과 같은 마스크, new = color_mask 만, gate = 색칠이면 color_mask · 아니면 지금 방식.
결과: eval_real/color_mask_<이름>.csv (이름이 tmp 로 시작하면 eval_real/data/ 에) (Meta 는 파일 이름 · 숫자만), 화면에 묶음별 표.
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import color_mask as cm  # noqa: E402
import colored_eval as ce  # noqa: E402
import kid_photo as kp  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT  # noqa: E402

sr = rt.sr
VARIANTS = ["orig", "thick", "linecolor", "fill", "yellow", "shadow", "light", "mix_mild", "mix_heavy"]
QD_CLS = {"fill": "heavy", "mix_mild": "heavy", "mix_heavy": "heavy"}  # 나머지 변환은 선만


def old_mask(img):
    """지금 Drawing 이 쓰는 마스크와 같은 것 + 자른 곳(원본 좌표)."""
    notes = []
    c = rt.clean_photo(img, notes)
    _, mesh, alpha = sr.crop_and_mask(c, notes)
    return (mesh > 0) | (alpha > 0), ce.crop_box(c), notes


def run_mask(img, gt, method):
    t0 = time.time()
    try:
        if method == "old":
            m, box, notes = old_mask(img)
        else:
            notes = []
            agree = float(method[4:]) if method.startswith("safe") else None  # safe0.3 = 겹침 0.3 미만이면 지금 방식
            _, mesh, alpha, box = cm.crop_and_mask_box(img, notes, gate=method == "gate", agree=agree)
            m = (mesh > 0) | (alpha > 0)
    except Exception as e:
        return dict(iou=0.0, fail=1, recall=0.0, prec=0.0, t_sec=round(time.time() - t0, 3), used="", err=repr(e)[:120])
    t = time.time() - t0
    x0, y0, x1, y1 = box
    full = np.zeros(gt.shape, bool)
    full[y0:y1, x0:x1] = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    inter = (full & gt).sum()
    iou = float(inter / max(1, (full | gt).sum()))
    rec, prec = inter / max(1, gt.sum()), inter / max(1, full.sum())  # 정답을 얼마나 덮었나 · 잡은 것 중 정답 비율
    used = "color" if any(n.startswith("색 마스크 (") for n in notes) else "old"
    return dict(iou=round(iou, 3), fail=int(iou < 0.5), recall=round(float(rec), 3), prec=round(float(prec), 3),
                t_sec=round(t, 3), used=used, err="")


def meta_job(job):
    i, path, segs, cls, frac, method = job
    img = cv2.imread(path)
    gt = np.zeros(img.shape[:2], np.uint8)
    for s in segs:
        cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 1)
    return dict(i=i, file=Path(path).name, cls=cls, color_frac=frac, **run_mask(img, gt > 0, method))


def qd_job(job):
    cat, k, v, strokes, other, seed, method = job
    img, gt = kp.make(strokes, v, seed, other)
    return dict(cat=cat, id=k, variant=v, cls=QD_CLS.get(v, "line"), **run_mask(img, gt, method))


def summary(rows, key):
    print(f"| {key} | 장수 | 마스크 실패 (IoU<0.5) | IoU 중앙값 | 덜 잡음 (덮음<0.7) | 더 잡음 (정답 비율<0.5) | 오류 | 색 마스크 씀 | 평균 초 | 90% 초 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    groups = sorted({r[key] for r in rows}, key=lambda g: ["heavy", "some", "line"].index(g) if g in ("heavy", "some", "line") else 9)
    for g in groups + ["전체"]:
        rs = rows if g == "전체" else [r for r in rows if r[key] == g]
        n = len(rs)
        f = sum(r["fail"] for r in rs)
        ts = [r["t_sec"] for r in rs]
        print(f"| {g} | {n} | {f} ({100 * f / n:.1f}%) | {np.median([r['iou'] for r in rs]):.3f} | "
              f"{sum(r['recall'] < 0.7 for r in rs)} | {sum(r['prec'] < 0.5 for r in rs)} | {sum(bool(r['err']) for r in rs)} | "
              f"{sum(r.get('used') == 'color' for r in rs)} | {np.mean(ts):.2f} | {np.percentile(ts, 90):.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["meta", "qd"])
    ap.add_argument("--meta")
    ap.add_argument("--qd")
    ap.add_argument("--set", choices=["learn", "check"], default="learn")
    ap.add_argument("--method", default="old")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.mode == "meta":
        meta = Path(a.meta)
        idx = json.load(open(meta / "index.json"))
        cs = json.load(open(meta / "color_sets.json"))
        jobs = [(it["i"], str(meta / "x" / it["file"]), it["segs"], cs[str(it["i"])]["cls"], cs[str(it["i"])]["color_frac"], a.method)
                for it in idx if cs[str(it["i"])]["split"] == a.set]
        if a.n:
            jobs = jobs[:: max(1, len(jobs) // a.n)][: a.n]
        fn, key = meta_job, "cls"
    else:
        n = a.n or (20 if a.set == "learn" else 30)
        jobs = []
        for ci, cat in enumerate(EXPECT):
            sp = ce.load_qd(Path(a.qd) / f"{cat}.ndjson", n)
            for k, st in enumerate(sp[a.set]):
                seed = 1000 * ci + k + (0 if a.set == "learn" else 500)
                for v in VARIANTS:
                    jobs.append((cat, k, v, st, sp["other"][k % 50], seed, a.method))
        fn, key = qd_job, "variant"
    t0 = time.time()
    with Pool(a.procs) as p:
        rows = p.map(fn, jobs, chunksize=4)
    name = a.out or f"{a.mode}_{a.set}_{a.method}"
    out = (HERE / "data") if name.startswith("tmp") else HERE  # tmp_ 로 시작하면 커밋 안 하는 data/ 에
    with open(out / f"color_mask_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"## color_mask_{name}: {len(rows)}장, {time.time() - t0:.0f}초 (동시 {a.procs})")
    summary(rows, key)
    if a.mode == "qd":
        summary(rows, "cls")


if __name__ == "__main__":
    main()
