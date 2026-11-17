"""「몸 전체가 옆모습으로 보이는 그림」 자동 고르기에 쓰는 모양 숫자. 규칙(quadruped.py)과 따로 계산한다.

몸 덩어리 = 굵은 선 그림 + 선으로 닫힌 안쪽 (작은 틈은 메움).
  aspect   : 덩어리 상자 폭 / 높이
  extent   : 덩어리 넓이 / 상자 넓이 (얼굴만 그린 동그라미는 크다)
  solidity : 덩어리 넓이 / 볼록 껍질 넓이 (다리 사이가 비면 작다)
  legs     : 상자 아래쪽(높이 75~95%) 여러 줄에서 가로로 끊긴 덩어리 수의 가장 큰 값 (다리 수 짐작)
  legs_med : 같은 줄들의 가운데 값
  low_gap  : 아래 20% 띠에서 덩어리가 있는 열들 사이의 빈 열 비율 (다리 사이 틈)
"""

import cv2
import numpy as np

import qd


def silhouette(d: dict) -> np.ndarray:
    img, _ = qd.render(d, "fill")
    return (img.min(axis=2) < 250)


def feats(d: dict) -> dict:
    m = silhouette(d).astype(np.uint8)
    ys, xs = np.where(m > 0)
    l, t, r, b = xs.min(), ys.min(), xs.max(), ys.max()
    W, H = r - l + 1, b - t + 1
    area = float(m.sum())
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(cnts))
    hull_area = max(1.0, cv2.contourArea(hull))
    min_w = max(2, round(0.015 * W))
    counts = []
    for f in (0.75, 0.8, 0.85, 0.9, 0.95):
        row = m[t + int(f * (H - 1)), l:r + 1]
        dd = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        runs = [(a, z) for a, z in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]) if z - a >= min_w]
        counts.append(len(runs))
    band = m[t + int(0.8 * H):b + 1, l:r + 1].any(axis=0)
    filled = np.where(band)[0]
    low_gap = 0.0
    if filled.size:
        span = band[filled[0]:filled[-1] + 1]
        low_gap = float(1 - span.mean())
    return {"aspect": W / H, "extent": area / (W * H), "solidity": area / hull_area,
            "legs": max(counts), "legs_med": float(np.median(counts)), "low_gap": low_gap}
