"""모아 보기: 원본 | 사진 정리 결과 | 마스크(초록 = 정답만, 빨강 = 잡은 것만, 노랑 = 둘 다).

    Meta (재기용, 저장소에 올리지 않음):
      ../venv/bin/python -I eval_real/faint_ruled_sheet.py meta --meta eval_real/data/meta --impl old|new --group faint --set learn [--fail] [--n 24] --png eval_real/data/view/이름.jpg
    QuickDraw (CC BY 4.0, 올려도 됨):
      ../venv/bin/python -I eval_real/faint_ruled_sheet.py qd --qd eval_real/data/qd --group faint --set check --png eval_real/sheet_faint_ruled_qd.png
--impl both 면 한 줄에 old · new 를 나란히 (원본 | old 정리 | old 마스크 | new 정리 | new 마스크).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import colored_eval as ce  # noqa: E402
import faint_ruled_eval as fe  # noqa: E402
import faint_ruled_pick as fp  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT  # noqa: E402

T = 200


def fit(img):
    t = np.full((T, T, 3), 255, np.uint8)
    s = min(T / img.shape[1], T / img.shape[0])
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    t[:im.shape[0], :im.shape[1]] = im
    return t


def full_mask(img, d):
    cl = rt.clean_photo(img)
    x0, y0, x1, y1 = ce.crop_box(cl)
    m = ((d.mask > 0) | (d.alpha > 0)).astype(np.uint8)
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return cl, full


def tiles(img, gt, impl, label):
    fe.use(impl)
    try:
        d = rt.Drawing(img)
        cl, pm = full_mask(img, d)
        iou = (pm & gt).sum() / max(1, (pm | gt).sum())
    except Exception:
        cl, pm, iou = rt.clean_photo(img), np.zeros_like(gt), 0.0
    vis = np.full(img.shape, 255, np.uint8)
    vis[gt & ~pm] = (0, 170, 0)
    vis[pm & ~gt] = (0, 0, 230)
    vis[pm & gt] = (0, 200, 230)
    a, b = fit(cl), fit(vis)
    cv2.putText(b, f"{impl} IoU {iou:.2f}", (3, T - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return [a, b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["meta", "qd"])
    ap.add_argument("--meta")
    ap.add_argument("--qd")
    ap.add_argument("--impl", default="old")
    ap.add_argument("--group", default="faint")
    ap.add_argument("--set", default="learn")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--fail", action="store_true", help="old 로 마스크 실패한 그림만")
    ap.add_argument("--ids", default="", help="Meta index 번호 (쉼표)만")
    ap.add_argument("--png", required=True)
    a = ap.parse_args()
    impls = ["old", "new"] if a.impl == "both" else [a.impl]
    items = []
    if a.mode == "meta":
        root = Path(a.meta)
        G = fe.meta_groups(root)
        for it in G[a.group]:
            if fp.split_of(it["file"]) != a.set or (a.ids and str(it["i"]) not in a.ids.split(",")):
                continue
            img = cv2.imread(str(root / "x" / it["file"]))
            items.append((img, fp.gt_mask(it["segs"], img.shape[:2]) > 0, str(it["i"])))
    else:
        for ci, cat in enumerate(EXPECT):
            sp = ce.load_qd(Path(a.qd) / f"{cat}.ndjson", 30)
            for k in range(3):
                seed = 1000 * ci + k + (0 if a.set == "learn" else 500)
                img, gt = fe.make(sp[a.set][k], a.group, seed, sp["other"][k % 50])
                items.append((img, gt, f"{cat}#{k}"))
    rows = []
    for img, gt, name in items[a.skip:]:
        if a.fail:
            fe.use("old")
            r = fe.measure(img, gt)
            if not r["mask_fail"]:
                continue
        o = fit(img)
        cv2.putText(o, name, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        row = [o]
        for im in impls:
            row += tiles(img, gt, im, name)
        rows.append(np.hstack(row))
        if len(rows) >= a.n:
            break
    per = 2 if len(impls) == 2 else 3
    while len(rows) % per:
        rows.append(np.full_like(rows[0], 255))
    sheet = np.vstack([np.hstack(rows[i:i + per]) for i in range(0, len(rows), per)])
    cv2.imwrite(a.png, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(a.png, len(rows))


if __name__ == "__main__":
    main()
