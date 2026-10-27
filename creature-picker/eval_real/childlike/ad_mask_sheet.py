"""마스크 모아 보기: 사진 위에 정답 마스크 윤곽(초록)과 우리 마스크(빨강 칠)를 겹쳐 그린다.

    ../venv/bin/python -I eval_real/childlike/ad_mask_sheet.py --data 꺼낸폴더 --res 결과.csv --split train --out 그림.png [--n 24] [--max-iou 0.5]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
import rig_types as rt  # noqa: E402
from ad_eval import crop_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-iou", type=float, default=0.5)
    ap.add_argument("--labeled", action="store_true", help="라벨 붙은 그림만")
    a = ap.parse_args()
    D = Path(a.data)
    rows = [r for r in csv.DictReader(open(a.res, encoding="utf-8-sig")) if r["split"] == a.split and r["drawing_ok"] == "1"
            and float(r["iou"]) < a.max_iou and (r["label"] if a.labeled else r["random"] == "1")]
    rows = [rows[i] for i in np.random.default_rng(3).permutation(len(rows))[:a.n]]
    T = 220
    cells = []
    for r in rows:
        img = rt.sr.load_image(D / "img" / f"{r['id']}.png")
        gt = cv2.resize(cv2.imread(str(D / "gt" / f"{r['id']}.png"), cv2.IMREAD_GRAYSCALE), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        d = rt.Drawing(img)
        x0, y0, x1, y1 = crop_box(img)
        pred = np.zeros(gt.shape, np.uint8)
        pred[y0:y1, x0:x1] = cv2.resize(d.alpha, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        out = img.copy()
        out[pred > 0] = (0.55 * out[pred > 0] + 0.45 * np.array([60, 60, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 200, 0), 2)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (255, 120, 0), 1)
        s = T / max(out.shape[:2])
        out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)))
        c = np.full((T + 16, T, 3), 255, np.uint8)
        c[16:16 + out.shape[0], :out.shape[1]] = out
        cv2.putText(c, f"{r['id'][:6]} {r['label'] or '-'} iou={r['iou']} p/g={r['pred_over_gt']}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 180), 1, cv2.LINE_AA)
        cells.append(c)
    while len(cells) % 6:
        cells.append(np.full_like(cells[0], 255))
    cv2.imwrite(a.out, np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]))
    print(len(rows), "장 →", a.out)


if __name__ == "__main__":
    main()
