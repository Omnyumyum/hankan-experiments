"""누른 점 분할 모아 보기: 그림마다 한 줄 = 원본(초록 정답 윤곽 · 빨간 누른 점) · 지금 clean_photo 마스크 · 모델 후보 3개 (작은 것 → 큰 것) · 고른 최종.

    ../venv/bin/python -I eval_real/tap_sheet.py --data meta --set learn --model mobile_sam --prompt in --keys meta1,meta2 --out data/sheet.png [--strat guardbig+fill]
Meta 그림이 든 모아 보기는 저장소에 올리지 않는다(CC BY-NC) — --out 은 eval_real/data/ 아래로.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tap_segment_eval as te  # noqa: E402

C = 200


def tile(img, m=None, col=(255, 0, 255), gt=None, pts=(), title=""):
    t = img.copy()
    if m is not None:
        t[m] = (0.5 * t[m] + 0.5 * np.array(col)).astype(np.uint8)
    if gt is not None:
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (0, 180, 0), max(1, max(t.shape) // 200))
    for p in pts:
        cv2.circle(t, (int(p[0]), int(p[1])), max(3, max(t.shape) // 60), (0, 0, 255), -1)
    s = C / max(t.shape[:2])
    t = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((C + 18, C, 3), 255, np.uint8)
    out[18:18 + t.shape[0], :t.shape[1]] = t
    cv2.putText(out, title, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="meta")
    ap.add_argument("--set", default="learn")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--variants", default=",".join(te.kp.VARIANTS))
    ap.add_argument("--model", default="mobile_sam")
    ap.add_argument("--prompt", default="in")
    ap.add_argument("--strat", default="guardbig+fill")
    ap.add_argument("--fill", type=float, default=0.03)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tag = te.tag_of(a)
    base = {b["key"]: b for b in pickle.load(gzip.open(te.cache_path(a.data, a.set, "base", tag), "rb"))}
    mod = {r["key"]: r for r in pickle.load(gzip.open(te.cache_path(a.data, a.set, a.model, tag), "rb"))["rows"]}
    items = {it["key"]: it for it in te.items_for(a)}
    rows = []
    keys = list(base)[: int(a.keys[6:])] if a.keys.startswith("first:") else a.keys.split(",")
    for k in keys:
        b = base[k]
        img, gt = te.load_img(items[k])
        bm = te.unpack(b["base"])
        cands, sc = mod[k]["out"][a.prompt]
        cands = [te.unpack(x) for x in cands]
        pts, box = te.prompt(a.prompt, b["taps"], b["pbox"], b["cbox"])
        fk = max(5, int(a.fill * max(gt.shape)) | 1)
        fin = te.final_mask(a.strat, cands, sc, b["pbox"], bm, pts[0], fk)
        im0 = img.copy()
        if box is not None:
            cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        r = [tile(im0, None, gt=gt, pts=pts, title=k), tile(img, bm, (0, 160, 255), gt=gt, title=f"now {te.iou(bm, gt):.2f}")]
        r += [tile(img, c, gt=gt, pts=pts, title=f"c{j} s{sc[j]:.2f} iou{te.iou(c, gt):.2f}") for j, c in enumerate(cands)]
        r += [tile(img, fin, (255, 120, 0), gt=gt, title=f"{a.strat} {te.iou(fin, gt):.2f}")]
        rows.append(np.hstack(r))
    cv2.imwrite(str(HERE / a.out), np.vstack(rows))


if __name__ == "__main__":
    main()
