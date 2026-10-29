"""모아 보기: 한 줄 = 사진(초록 정답 윤곽 · 파랑 찾기 상자 · 빨간 누른 점) | 지금 마스크 | 상자 안 마스크 | 상자 + 누름 최종. 마스크는 초록 = 정답만 · 빨강 = 잡은 것만 · 노랑 = 둘 다.

    ../venv/bin/python -I eval_real/box_sheet.py --set qd_check --pick 상자망침|상자고침|누름|무작위 [--n 12] --out 이름.jpg
Meta(CC BY-NC) 그림은 eval_real/data/ 에만 저장한다. 저장소에는 QuickDraw(CC BY 4.0)로 만든 것만 올린다.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import box_eval as be  # noqa: E402
import box_score as bs  # noqa: E402

S = 220


def fit(img):
    h, w = img.shape[:2]
    k = S / max(h, w)
    im = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    c = np.full((S, S, 3), 255, np.uint8)
    y0, x0 = (S - im.shape[0]) // 2, (S - im.shape[1]) // 2
    c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return c


def over(img, m, gt):
    o = (img * 0.45 + 140).astype(np.uint8)
    o[gt & ~m] = (0, 180, 0)
    o[m & ~gt] = (0, 0, 230)
    o[m & gt] = (0, 210, 230)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--pick", default="무작위")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = pickle.load(open(bs.CACHE / f"{a.set}_rows.pkl", "rb"))
    fin = {r["key"]: bs.final(r) for r in rows}
    m = bs.RULE["margin"]
    sel = {"상자망침": lambda r, f: r["iou_now"] >= 0.5 and f["box"] < 0.5, "상자고침": lambda r, f: r["iou_now"] < 0.5 and f["box"] >= 0.5,
           "누름": lambda r, f: f["ask"], "남은실패": lambda r, f: f["box_tap"] < 0.5, "무작위": lambda r, f: True}[a.pick]
    keys = [r["key"] for r in rows if sel(r, fin[r["key"]])]
    rng = np.random.default_rng(a.seed)
    keys = list(rng.permutation(keys)[: a.n]) if len(keys) > a.n else keys
    base = {b["key"]: b for b in pickle.load(gzip.open(bs.CACHE / f"{a.set}_base.pkl.gz", "rb")) if b["key"] in keys}
    sam = {s["key"]: s for s in pickle.load(gzip.open(bs.CACHE / f"{a.set}_sam.pkl.gz", "rb")) if s["key"] in keys}
    its = {it["key"]: it for it in be.items(a.set) if it["key"] in keys}
    lines = []
    for k in keys:
        b, s, f = base[k], sam[k], fin[k]
        img, gt = be.load_img(its[k])
        full = bs.unpack(b["full"])
        boxm = bs.unpack(b["inbox"][m]["mask"]) if f["used"] else full
        tap = (bs.tap_full(b, s, "in+dbox", boxm) if f["used"] else bs.tap_full(b, s, "in", full)) if f["ask"] else boxm
        ph = img.copy()
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lw = max(2, max(img.shape[:2]) // 150)
        cv2.drawContours(ph, cs, -1, (0, 200, 0), lw)
        if b["box"] is not None:
            x0, y0, x1, y1 = map(int, b["box"])
            cv2.rectangle(ph, (x0, y0), (x1, y1), (255, 80, 0), lw)
        if f["ask"]:
            cv2.circle(ph, tuple(int(v) for v in b["taps"]["i"]), 3 * lw, (0, 0, 255), -1)
        cells = [fit(ph), fit(over(img, full, gt)), fit(over(img, boxm, gt)), fit(over(img, tap, gt))]
        tt = [f"{k} {b['group']}", f"now {f['now']:.2f}", f"box {f['box']:.2f}" + ("" if f["used"] else " (x)"), f"final {f['box_tap']:.2f}" + (" tap" if f["ask"] else "")]
        row = np.full((S + 18, 4 * S, 3), 255, np.uint8)
        for j, (c, t) in enumerate(zip(cells, tt)):
            row[18:, j * S:(j + 1) * S] = c
            cv2.putText(row, t, (j * S + 3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        lines.append(row)
    out = np.vstack(lines) if lines else np.full((20, 4 * S, 3), 255, np.uint8)
    path = Path(a.out)
    if not path.is_absolute():
        path = HERE / path
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(path, len(keys))


if __name__ == "__main__":
    main()
