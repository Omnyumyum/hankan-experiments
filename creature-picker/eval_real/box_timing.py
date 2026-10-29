"""CPU 한 장 시간을 따로 잰다 (다른 일 없이, 한 프로세스): 찾기 모델 · MobileSAM 불러오기, 그림 한 장마다 찾기 · 지금 방식 · 상자 안 모양 따기 · MobileSAM 인코더.

    ../venv/bin/python -I eval_real/box_timing.py --set nh_check --n 30 --threads 4
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="nh_check")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.2)
    a = ap.parse_args()
    import cv2
    import torch
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    import box_eval as be
    import find_box as fb
    import rig_types as rt
    import tap_models as tm
    r0 = rss()
    t0 = time.time()
    det = fb.load()
    t_det_load = time.time() - t0
    r1 = rss()
    t0 = time.time()
    sam = tm.load("mobile_sam")
    t_sam_load = time.time() - t0
    r2 = rss()
    its = be.items(a.set)[: a.n]
    T = {k: [] for k in ["det", "full", "inbox", "clean", "sam_enc", "sam_dec", "sam_enc_box"]}
    for it in its:
        img, _ = be.load_img(it)
        t0 = time.time()
        box, sc, _ = fb.detect_top(det, img)
        T["det"].append(time.time() - t0)
        t0 = time.time()
        be.drawing_mask(img)
        T["full"].append(time.time() - t0)
        if box is not None:
            t0 = time.time()
            be.in_box(img, box, a.margin, be.drawing_mask)
            T["inbox"].append(time.time() - t0)
        t0 = time.time()
        clean = rt.clean_photo(img)
        T["clean"].append(time.time() - t0)
        t0 = time.time()
        sam.set_image(clean)
        T["sam_enc"].append(time.time() - t0)
        h, w = img.shape[:2]
        t0 = time.time()
        sam.predict([[w / 2, h / 2]])
        T["sam_dec"].append(time.time() - t0)
        if box is not None:
            x0, y0, x1, y1 = be.expand(box, img.shape[:2], be.SAM_MARGIN)
            t0 = time.time()
            sam.set_image(np.ascontiguousarray(clean[y0:y1, x0:x1]))
            T["sam_enc_box"].append(time.time() - t0)
    print(f"스레드 {a.threads}, {a.set} 앞 {len(its)}장")
    print(f"- 불러오기: 찾기 모델 {t_det_load:.2f}초 (+{r1 - r0:.0f}MB), MobileSAM {t_sam_load:.2f}초 (+{r2 - r1:.0f}MB), 최대 메모리 {rss():.0f}MB")
    for k, v in T.items():
        if v:
            print(f"- {k}: 평균 {np.mean(v):.2f} · 90% {np.percentile(v, 90):.2f} · 최대 {max(v):.2f}초 ({len(v)}장)")


if __name__ == "__main__":
    main()
