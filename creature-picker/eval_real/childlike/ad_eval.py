"""Amateur Drawings(종이에 그린 그림을 휴대폰으로 찍은 사진)로 지금 파이프라인을 잰다. 코드는 고치지 않는다.

    ../venv/bin/python -I eval_real/childlike/ad_eval.py --data ad_prep.py가꺼낸폴더 --out 결과.csv
재는 것 (그림마다)
- 마스크: Drawing 이 만들어지나, 정답 마스크(Meta 데이터의 사람이 확인한 마스크)와 IoU — 원래 사진 좌표에서 잰다
- 종류 추측: propose 의 guess 가 눈으로 매긴 라벨(Q 네 발 · W 날개 · L 긴 몸 · B 몸통만 · F 얼굴만 → 몸통만)과 맞나
- 기대 종류로 움직였을 때 늘어남 > 5배 · 뒤집힘 > 1% (run_eval.py 와 같은 기준)
몫: id 마지막 글자 0-7 = 학습용(원인 보기), 8-f = 확인용(숫자 판단). (받은 parquet 은 id 순으로 정렬돼 있어 첫 글자는 거의 0 · 1 이다)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import check_wings as cw  # noqa: E402
import rig_types as rt  # noqa: E402

sr = rt.sr
EXPECT = {"Q": "quadruped", "W": "wings", "L": "long", "B": "blob", "F": "blob"}


def crop_box(img):
    """shape_rig.crop_and_mask 의 자르기 부분과 같은 계산 (마스크를 원래 사진 좌표로 되돌리려고). 결과 crop 이 같은지 one() 에서 확인한다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    area = st[:, cv2.CC_STAT_AREA]
    edge = {k for k in range(1, n) if area[k] >= 30 and (st[k, 0] <= 1 or st[k, 1] <= 1 or st[k, 0] + st[k, 2] >= w - 1 or st[k, 1] + st[k, 3] >= h - 1)}
    inner = sum(area[k] for k in range(1, n) if area[k] >= 30 and k not in edge)
    full = inner < sum(area[k] for k in edge)
    for k in range(1, n):
        if area[k] < 30 or (k in edge and (not full or sr.is_straight(lab, k))):
            ink[lab == k] = 0
    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def one(img_path, gt_path, label):
    kind = EXPECT.get(label, "")
    row = dict(drawing_ok=0, iou="", gt_in_crop="", pred_over_gt="", crop_over_gt="", box_ok="", notes="", guess="", match="",
               anim_ok="", stretch="", flipped="", big_stretch="", big_flip="", wing_mode="", t_sec="", err="")
    t0 = time.time()
    img = sr.load_image(img_path)
    gt = cv2.resize(cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    try:
        d = rt.Drawing(img)
        row["drawing_ok"] = 1
        row["notes"] = " / ".join(d.notes)
    except Exception as e:
        row["err"] = "Drawing: " + repr(e)[:120]
        row["t_sec"] = round(time.time() - t0, 2)
        return row, None, None
    x0, y0, x1, y1 = crop_box(img)
    s = sr.MAX_SIDE / max(y1 - y0, x1 - x0)
    same = cv2.resize(img[y0:y1, x0:x1], (int((x1 - x0) * s), int((y1 - y0) * s)), interpolation=cv2.INTER_AREA)
    row["box_ok"] = int(same.shape == d.crop.shape and np.array_equal(same, d.crop))
    pred = np.zeros_like(gt)
    pred[y0:y1, x0:x1] = cv2.resize(d.alpha, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    row["iou"] = round(inter / max(union, 1), 3)
    row["gt_in_crop"] = round(gt[y0:y1, x0:x1].sum() / max(gt.sum(), 1), 3)
    row["pred_over_gt"] = round(pred.sum() / max(gt.sum(), 1), 3)
    gy, gx = np.nonzero(gt)
    row["crop_over_gt"] = round((x1 - x0) * (y1 - y0) / max((np.ptp(gx) + 1) * (np.ptp(gy) + 1), 1), 2) if len(gx) else ""
    anim = None
    try:
        pr = rt.propose(d)
        row["guess"] = pr["guess"] or "None"
        if kind:
            row["match"] = int(pr["guess"] == kind)
            anim = rt.animate(d, kind, pr["points"][kind])
            row["anim_ok"] = 1
            row["stretch"] = round(cw.stretch(anim), 2)
            row["flipped"] = round(cw.flipped(anim), 4)
            row["big_stretch"] = int(row["stretch"] > 5)
            row["big_flip"] = int(row["flipped"] > 0.01)
            row["wing_mode"] = anim.get("wing_mode") or ""
    except Exception as e:
        row["err"] = repr(e)[:160]
        if kind:
            row["anim_ok"] = 0
    row["t_sec"] = round(time.time() - t0, 2)
    return row, d, anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.data)
    meta = list(csv.DictReader(open(D / "meta.csv", encoding="utf-8")))
    rows = []
    for k, m in enumerate(meta):
        r0 = dict(id=m["id"], split="train" if m["id"][-1] in "01234567" else "check", label=m["label"], random=m["random"])
        r, _, _ = one(D / "img" / f"{m['id']}.png", D / "gt" / f"{m['id']}.png", m["label"])
        r0.update(r)
        rows.append(r0)
        print(k, r0["id"][:8], r0["label"] or "-", r0["iou"], r0["guess"], r0["stretch"], r0["flipped"], r0["err"][:50], flush=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
