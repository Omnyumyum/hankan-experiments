"""모아 보기: 변환마다 그림 · 잡힌 마스크(분홍) · 정답 윤곽(초록) · IoU · 추측.

    venv/bin/python -I eval_real/colored_sheet.py --qd 받은폴더 --tag before|after [--meta 받은폴더 --meta-out 경로]
QuickDraw 결과는 eval_real/sheet_colored_<tag>.png (학습용 그림만 씀). Meta 그림은 라이선스가 분명하지 않아 --meta-out(저장소 밖)에만 쓴다.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import colored_eval as ce  # noqa: E402
import kid_photo as kp  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT  # noqa: E402

T = 200
SHOW = ["orig", "fill", "lined", "tilt", "eraser", "text", "mix_mild", "mix_heavy"]


def cell(img, gt, label):
    pre = rt.paper_only(img) if hasattr(rt, "paper_only") else img
    vis = pre.copy()
    try:
        d = rt.Drawing(img)
        x0, y0, x1, y1 = ce.crop_box(pre)
        m = ((d.mask > 0) | (d.alpha > 0)).astype(np.uint8)
        full = np.zeros(gt.shape, np.uint8)
        full[y0:y1, x0:x1] = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        vis[full > 0] = (vis[full > 0] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
        txt = f"{label} IoU {ce.mask_iou(img, d, gt):.2f} {rt.propose(d)['guess'] or 'None'}"
    except Exception as e:
        txt = f"{label} ERR {repr(e)[:20]}"
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (0, 170, 0), 2)
    s = T / max(vis.shape[:2])
    vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)), interpolation=cv2.INTER_AREA)
    c = np.full((T, T, 3), 255, np.uint8)
    c[:vis.shape[0], :vis.shape[1]] = vis
    cv2.rectangle(c, (0, 0), (T, 15), (255, 255, 255), -1)
    cv2.putText(c, txt, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--meta-out")
    a = ap.parse_args()
    rows = []
    for ci, cat in enumerate(EXPECT):
        if cat not in ("dog", "horse", "snake", "butterfly", "cloud"):
            continue
        sp = ce.load_qd(Path(a.qd) / f"{cat}.ndjson", 20)
        k = 1
        rows.append(np.hstack([cell(*kp.make(sp["learn"][k], v, 1000 * ci + k, sp["other"][k]), f"{cat} {v}") for v in SHOW]))
    cv2.imwrite(str(HERE / f"sheet_colored_{a.tag}.png"), np.vstack(rows))
    if a.meta and a.meta_out:
        ann = json.load(open(Path(a.meta) / "ann_part.json"))
        cells = []
        for k, (im, an) in enumerate(list(zip(ann["images"], ann["annotations"]))[:48]):
            img = cv2.imread(str(Path(a.meta) / im["file_name"]))
            gt = np.zeros(img.shape[:2], np.uint8)
            for s in an["segmentation"]:
                cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 1)
            cells.append(cell(img, gt > 0, f"#{k}"))
        cv2.imwrite(a.meta_out, np.vstack([np.hstack(cells[i:i + 8]) for i in range(0, len(cells), 8)]))


if __name__ == "__main__":
    main()
