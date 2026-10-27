"""비휴머노이드 라벨 그림의 움직임 모아 보기: 줄마다 사진 썸네일 + 기대 종류로 움직인 3장면.

    ../venv/bin/python -I eval_real/childlike/ad_motion_sheet.py --data 꺼낸폴더 --res 결과.csv --split check --out 그림.png [--per 4]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import check as ck  # noqa: E402
import rig_types as rt  # noqa: E402

EXPECT = {"Q": "quadruped", "W": "wings", "L": "long", "B": "blob", "F": "blob"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--split", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=4)
    a = ap.parse_args()
    D = Path(a.data)
    rows = [r for r in csv.DictReader(open(a.res, encoding="utf-8-sig")) if r["split"] == a.split and r["drawing_ok"] == "1"]
    rng = np.random.default_rng(5)
    T = 150
    lines = []
    for L in "QWLBF":
        lst = [r for r in rows if r["label"] == L]
        for r in [lst[i] for i in rng.permutation(len(lst))[:a.per]]:
            kind = EXPECT[L]
            img = rt.sr.load_image(D / "img" / f"{r['id']}.png")
            d = rt.Drawing(img)
            pr = rt.propose(d)
            anim = rt.animate(d, kind, pr["points"][kind])
            s = T / max(img.shape[:2])
            src = np.full((T, T, 3), 255, np.uint8)
            im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
            src[:im.shape[0], :im.shape[1]] = im
            cells = [src]
            fr = ck.frames_of(d, kind, anim)
            for f in (0, len(fr) // 3, 2 * len(fr) // 3):
                im = fr[f]
                s = T / max(im.shape[:2])
                im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
                c = np.full((T, T, 3), 255, np.uint8)
                c[:im.shape[0], :im.shape[1]] = im
                cells.append(c)
            line = np.hstack(cells)
            tag = f"{L} {r['id'][:6]} iou={r['iou']} guess={r['guess']} str={r['stretch']} flip={r['flipped']} {r['wing_mode']}"
            cv2.putText(line, tag, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
            lines.append(line)
    cv2.imwrite(a.out, np.vstack(lines))
    print(len(lines), "줄 →", a.out)


if __name__ == "__main__":
    main()
