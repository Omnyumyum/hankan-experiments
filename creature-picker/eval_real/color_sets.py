"""Meta Amateur Drawings 사진을 「색칠 많음 / 조금 / 선만」으로 나눈다 (정답 마스크 안 화소의 색으로).

    venv/bin/python -I eval_real/color_sets.py --meta 받은폴더
받은폴더: photo_mask_eval.py prep 이 만든 index.json 과 x/.
- 종이 색 = 정답 마스크를 넓게 키운 바깥의 Lab 중앙값.
- 색칠 화소 = 정답 안에서 종이와 색(a·b)이 15 넘게 다른 화소 중 굵은 곳(7px 원으로 열어도 남는 곳 — 색 선은 빠진다).
  색칠 비율 = 색칠 화소 / 정답 넓이. 진하게 칠한 곳(종이보다 60 넘게 어두운 굵은 곳)은 dark_frac 으로 따로 센다.
  많음 ≥ 0.30 · 조금 0.08~0.30 · 선만 < 0.08. 밝은 색칠(밝기가 종이보다 40 이내로 어둡지 않은 색칠 화소)도 따로 센다.
- 학습용(learn) = photo_mask_eval 의 old_learn · old_check · learn (예전 작업들이 본 그림), 확인용(check) = check.
결과: 받은폴더/color_sets.json, eval_real/color_sets.csv (파일 이름 · 숫자만).
Meta 데이터 annotation 의 licenses 는 CC BY-NC 4.0 → 재기에만 쓴다.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CUT_HEAVY, CUT_SOME = 0.30, 0.08


def stats(job):
    i, path, segs = job
    img = cv2.imread(path)
    gt = np.zeros(img.shape[:2], np.uint8)
    for s in segs:
        cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 1)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    k = max(5, int(0.04 * max(gt.shape)) | 1)
    out = cv2.dilate(gt, np.ones((k, k), np.uint8)) == 0
    if out.sum() < 100:
        out = np.ones_like(out)
    paper = np.median(lab[out], axis=0)
    inside = gt > 0
    ch = np.hypot(lab[..., 1] - paper[1], lab[..., 2] - paper[2])
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    col = inside & (cv2.morphologyEx((ch > 15).astype(np.uint8), cv2.MORPH_OPEN, disk) > 0)
    light = col & (lab[..., 0] > paper[0] - 40)
    dark = inside & (cv2.morphologyEx((lab[..., 0] < paper[0] - 60).astype(np.uint8), cv2.MORPH_OPEN, disk) > 0)
    area = max(1, inside.sum())
    frac = col.sum() / area
    cls = "heavy" if frac >= CUT_HEAVY else "some" if frac >= CUT_SOME else "line"
    return dict(i=i, file=Path(path).name, h=img.shape[0], w=img.shape[1], gt_frac=round(area / gt.size, 4),
                color_frac=round(float(frac), 4), light_frac=round(float(light.sum() / area), 4),
                dark_frac=round(float(dark.sum() / area), 4), paper_L=round(float(paper[0]), 1), cls=cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    meta = Path(a.meta)
    idx = json.load(open(meta / "index.json"))
    with Pool() as p:
        rows = p.map(stats, [(it["i"], str(meta / "x" / it["file"]), it["segs"]) for it in idx], chunksize=8)
    for it, r in zip(idx, rows):
        r["split"] = "check" if it["split"] == "check" else "learn"
    json.dump({r["i"]: dict(cls=r["cls"], split=r["split"], color_frac=r["color_frac"]) for r in rows}, open(meta / "color_sets.json", "w"))
    with open(HERE / "color_sets.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| 묶음 | 학습용 | 확인용 |\n|---|---|---|")
    for c in ("heavy", "some", "line"):
        print(f"| {c} | {sum(r['cls'] == c and r['split'] == 'learn' for r in rows)} | {sum(r['cls'] == c and r['split'] == 'check' for r in rows)} |")
    fr = np.array([r["color_frac"] for r in rows])
    print("색칠 비율 분위 10/25/50/75/90:", np.percentile(fr, [10, 25, 50, 75, 90]).round(3))
    print("크기:", np.unique([max(r["h"], r["w"]) for r in rows], return_counts=True))


if __name__ == "__main__":
    main()
