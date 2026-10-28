"""모아 보기: 원본 · 지금 방식 · color_mask 마스크를 나란히 (초록 = 정답 윤곽, 분홍 = 잡은 마스크).

    venv/bin/python -I eval_real/color_mask_sheet.py meta --meta 받은폴더 --csv A.csv --csv2 B.csv --pick 규칙 --out 그림.png
    venv/bin/python -I eval_real/color_mask_sheet.py qd --qd 받은폴더 --out 그림.png   (QuickDraw 색칠 변환: 저장소에 올려도 됨)
--pick: "newfail" (B 에서 실패), "worse" (B 가 A 보다 0.1 넘게 나쁨), "better", "random". --cls 로 묶음을 고른다.
Meta 그림이 든 모아 보기는 라이선스(CC BY-NC) 때문에 eval_real/data/ 에만 둔다.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import color_mask as cm  # noqa: E402
import colored_eval as ce  # noqa: E402
import kid_photo as kp  # noqa: E402
from color_mask_eval import old_mask  # noqa: E402
from run_eval import EXPECT  # noqa: E402

T = 220


def full_mask(img, method):
    if method == "old":
        m, box, _ = old_mask(img)
    else:
        _, mesh, alpha, box = cm.crop_and_mask_box(img, [], gate=method == "gate", agree=float(method[4:]) if method.startswith("safe") else None)
        m = (mesh > 0) | (alpha > 0)
    x0, y0, x1, y1 = box
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return full


def tile(img, gt, m, title):
    v = img.copy()
    if m is not None:
        v[m] = (0.5 * v[m] + 0.5 * np.array([180, 60, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(v, cs, -1, (0, 200, 0), max(1, max(img.shape[:2]) // 250))
    s = T / max(v.shape[:2])
    v = cv2.resize(v, (int(v.shape[1] * s), int(v.shape[0] * s)))
    t = np.full((T + 14, T, 3), 255, np.uint8)
    t[: v.shape[0], : v.shape[1]] = v
    cv2.putText(t, title, (2, T + 11), 0, 0.36, (0, 0, 0), 1)
    return t


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["meta", "qd"])
    ap.add_argument("--meta")
    ap.add_argument("--qd")
    ap.add_argument("--csv")
    ap.add_argument("--csv2")
    ap.add_argument("--pick", default="newfail")
    ap.add_argument("--cls", default="heavy")
    ap.add_argument("--methods", default="old,new")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ids", default="", help="쉼표로 그림 번호를 바로 고른다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meths = a.methods.split(",")
    rows = []
    if a.mode == "meta":
        meta = Path(a.meta)
        idx = {it["i"]: it for it in json.load(open(meta / "index.json"))}
        A = {r["i"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8-sig"))}
        B = {r["i"]: r for r in csv.DictReader(open(a.csv2, encoding="utf-8-sig"))}
        ids = [i for i in B if B[i]["cls"] in a.cls.split(",")]
        da = {i: float(B[i]["iou"]) - float(A[i]["iou"]) for i in ids}
        if a.pick == "newfail":
            ids = [i for i in ids if B[i]["fail"] == "1"]
        elif a.pick == "worse":
            ids = sorted([i for i in ids if da[i] < -0.1], key=lambda i: da[i])
        elif a.pick == "better":
            ids = sorted([i for i in ids if da[i] > 0.1], key=lambda i: -da[i])
        if a.ids:
            ids = a.ids.split(",")
        elif a.pick in ("newfail", "random"):
            rng = np.random.default_rng(a.seed)
            ids = [ids[j] for j in rng.permutation(len(ids))]
        for i in ids[: a.n]:
            it = idx[int(i)]
            img = cv2.imread(str(meta / "x" / it["file"]))
            gt = np.zeros(img.shape[:2], np.uint8)
            for s in it["segs"]:
                cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 1)
            rows.append((f"#{i}", img, gt > 0))
    else:
        for ci, cat in enumerate(["dog", "horse", "snake", "butterfly", "cloud", "bird"]):
            sp = ce.load_qd(Path(a.qd) / f"{cat}.ndjson", 20)
            k = 0
            seed = 1000 * list(EXPECT).index(cat) + k
            for v in ("fill", "mix_mild", "mix_heavy"):
                img, gt = kp.make(sp["learn"][k], v, seed, sp["other"][k])
                rows.append((f"{cat} {v}", img, gt))
    out = []
    for name, img, gt in rows:
        ts = [tile(img, gt, None, name)]
        for m in meths:
            try:
                fm = full_mask(img, m)
                ts.append(tile(img, gt, fm, f"{m} IoU {iou(fm, gt):.2f}"))
            except Exception as e:
                ts.append(tile(img, gt, None, f"{m} err {type(e).__name__}"))
        out.append(np.hstack(ts))
    cols = 2 if len(out) > 6 else 1
    while len(out) % cols:
        out.append(np.full_like(out[0], 255))
    grid = np.vstack([np.hstack(out[i:i + cols]) for i in range(0, len(out), cols)])
    cv2.imwrite(a.out, grid)
    print(a.out, grid.shape)


if __name__ == "__main__":
    main()
