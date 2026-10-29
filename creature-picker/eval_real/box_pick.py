"""Meta Amateur Drawings 에서 「사람이 아닌 그림」 묶음을 눈으로 고르기 위한 모아 보기 (재기용, 그림은 올리지 않는다).

    ../venv/bin/python -I eval_real/box_pick.py sheets --meta eval_real/data/meta [--per 100]
      → eval_real/data/box_pick/sheet_###.jpg : index.json 순서대로 100장씩, 칸마다 정답 마스크 둘레 상자로 자른 그림 + 번호(index i)
눈으로 본 결과는 box_nonhuman_labels.csv (i, file, kind = quadruped|wings|long|blob, 한 줄 메모) 로 적는다.
나누기(학습 · 확인)는 box_eval.py 가 파일 이름 md5 첫 글자 짝 · 홀로 한다 (FAINT_RULED 와 같은 규칙).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "box_pick"


def thumb(img, segs, s):
    gt = np.zeros(img.shape[:2], np.uint8)
    for g in segs:
        cv2.fillPoly(gt, [np.round(np.array(g).reshape(-1, 2)).astype(np.int32)], 1)
    ys, xs = np.nonzero(gt)
    if len(xs):
        p = 6
        img = img[max(0, ys.min() - p): ys.max() + p, max(0, xs.min() - p): xs.max() + p]
    h, w = img.shape[:2]
    k = s / max(h, w)
    im = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    cell = np.full((s, s, 3), 255, np.uint8)
    y0, x0 = (s - im.shape[0]) // 2, (s - im.shape[1]) // 2
    cell[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return cell


def sheets(a):
    meta = Path(a.meta)
    idx = json.load(open(meta / "index.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    s, cols = a.cell, 10
    for k in range(0, len(idx), a.per):
        part = idx[k: k + a.per]
        rows = -(-len(part) // cols)
        sh = np.full((rows * (s + 14), cols * s, 3), 255, np.uint8)
        for j, it in enumerate(part):
            img = cv2.imread(str(meta / "x" / it["file"]))
            r, c = divmod(j, cols)
            sh[r * (s + 14) + 14: r * (s + 14) + 14 + s, c * s: (c + 1) * s] = thumb(img, it["segs"], s)
            cv2.putText(sh, str(it["i"]), (c * s + 2, r * (s + 14) + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.rectangle(sh, (c * s, r * (s + 14)), ((c + 1) * s - 1, (r + 1) * (s + 14) - 1), (200, 200, 200), 1)
        cv2.imwrite(str(OUT / f"sheet_{k // a.per:03d}.jpg"), sh, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{-(-len(idx) // a.per)}장 → {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sheets"])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--cell", type=int, default=110)
    a = ap.parse_args()
    sheets(a)


if __name__ == "__main__":
    main()
