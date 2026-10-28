"""모델마다 CPU 시간 · 메모리만 따로 잰다 (다른 일을 돌리지 않을 때). 모델마다 새 프로세스.

    ../venv/bin/python -I eval_real/tap_timing.py --model mobile_sam --threads 4 [--n 20] [--clean]
한 줄 JSON: 불러오기 초, 그림 한 장 인코더(+clean_photo) 초, 점 하나 디코더 초, 메모리(torch 만 불러온 뒤 → 모델 → 최대, MB).
그림은 Meta learn 앞 n장 (긴 변 ≤ 512px. SAM 계열은 어떤 크기든 긴 변 1024 로 바꿔 넣으므로 큰 사진에서도 인코더 시간은 같다).
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import rig_types as rt
    import tap_models as tm
    idx = json.load(open(HERE / "data" / "meta" / "index.json"))
    imgs = [cv2.imread(str(HERE / "data" / "meta" / "x" / it["file"])) for it in idx if it["split"] == "learn"][: a.n]
    m0 = rss()
    t0 = time.time()
    m = tm.load(a.model)
    load_t = time.time() - t0
    m1 = rss()
    enc, dec = [], []
    for k, img in enumerate(imgs):
        t0 = time.time()
        if a.clean:
            img = rt.clean_photo(img)
        m.set_image(img)
        t = time.time() - t0
        if k:  # 첫 장은 데우기라 뺀다
            enc.append(t)
        h, w = img.shape[:2]
        t0 = time.time()
        m.predict([(w / 2, h / 2)])
        dec.append(time.time() - t0)
    print(json.dumps(dict(model=a.model + ("_clean" if a.clean else ""), threads=a.threads, load_s=round(load_t, 2),
                          enc_mean=round(float(np.mean(enc)), 3), enc_p90=round(float(np.percentile(enc, 90)), 3),
                          dec_mean=round(float(np.mean(dec[1:])), 3), rss_torch=round(m0), rss_model=round(m1), rss_peak=round(rss()))))


if __name__ == "__main__":
    main()
