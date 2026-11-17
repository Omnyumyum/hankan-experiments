"""Quick, Draw! 그림 읽기와 그리기.

그림 꼴 세 가지:
  line  : 선만 (원본과 비슷한 가는 선)
  thick : 굵은 선
  fill  : 선 안을 한 가지 색으로 채운 것 (아이 색칠 그림 흉내)

흔들기(돌리기 · 크기 · 뒤집기 · 선 굵기)는 그림을 다시 그려서 만든다. 기준 그림(흔들지 않은 것)의
좌표와 흔든 그림 좌표 사이를 오가는 함수(to_img / to_base)를 같이 돌려준다.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "qd"

CATS = ["horse", "cow", "sheep", "pig", "dog", "cat", "elephant", "giraffe", "zebra", "lion",
        "tiger", "camel", "rhinoceros", "crocodile", "mouse", "raccoon", "squirrel"]
STYLES = ["line", "thick", "fill"]

SIZE = 512          # 그림 칸 크기 (px)
FIT = 0.70          # 그림이 칸에서 차지하는 비율 (1.2배로 키워도 칸 안에 남게)
WIDTH = {"line": 3, "thick": 10, "fill": 4}            # 기본 선 굵기
WIDTH_ALT = {"line": 6, "thick": 16, "fill": 8}        # 「선 굵기 바꾸기」 흔들기

# 아이 크레파스 · 색연필 느낌의 채움 색 (BGR)
PALETTE = [(60, 170, 250), (70, 200, 120), (230, 140, 70), (90, 90, 230), (180, 120, 220),
           (40, 210, 230), (150, 110, 70), (200, 200, 90), (120, 160, 240), (100, 180, 180)]
INK = (35, 35, 35)


def load(cat: str, n: int = 3000) -> list[dict]:
    """종류 하나의 앞에서부터 recognized 가 true 인 그림 n 장."""
    out = []
    for line in open(DATA / f"{cat}.part", "rb").read().split(b"\n")[:-1]:
        d = json.loads(line)
        if d["recognized"]:
            out.append({"key": d["key_id"], "cat": cat, "strokes": d["drawing"]})
            if len(out) >= n:
                break
    return out


def _base_params(strokes) -> tuple[float, np.ndarray]:
    pts = np.concatenate([np.array(s, float).T for s in strokes])
    lo, hi = pts.min(0), pts.max(0)
    s = FIT * SIZE / max(1.0, (hi - lo).max())
    off = SIZE / 2 - s * (lo + hi) / 2
    return s, off


def make_transform(rot: float = 0.0, scale: float = 1.0, flip: bool = False):
    """기준 그림 좌표 → 흔든 그림 좌표 (칸 가운데를 중심으로). (to_img, to_base) 함수 둘."""
    c = np.array([SIZE / 2, SIZE / 2])
    t = np.deg2rad(rot)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    F = np.diag([-1.0, 1.0]) if flip else np.eye(2)
    M = scale * R @ F
    Mi = np.linalg.inv(M)

    def to_img(p):
        p = np.asarray(p, float)
        return (p - c) @ M.T + c

    def to_base(p):
        p = np.asarray(p, float)
        return (p - c) @ Mi.T + c
    return to_img, to_base


def render(d: dict, style: str, rot: float = 0.0, scale: float = 1.0, flip: bool = False,
           width: int | None = None):
    """그림 한 장 (BGR uint8) 과 (to_img, to_base)."""
    s, off = _base_params(d["strokes"])
    to_img, to_base = make_transform(rot, scale, flip)
    w = WIDTH[style] if width is None else width
    w = max(1, int(round(w * scale)))                       # 크기를 바꾸면 선도 같이 (사진 크기 바꾸기와 같게)
    polys = [np.round(to_img(np.array(st, float).T * s + off) * 4).astype(np.int32) for st in d["strokes"]]
    ink = np.zeros((SIZE, SIZE), np.uint8)
    cv2.polylines(ink, polys, False, 255, w, cv2.LINE_AA, shift=2)
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    if style == "fill":
        # 선으로 닫힌 안쪽을 채운다. 작은 틈은 메운 뒤에 바깥을 찾는다 (아이도 틈을 넘어 칠하지 않는다)
        k = max(3, int(round(0.025 * SIZE * scale)))
        closed = cv2.morphologyEx((ink > 60).astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        pad = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(pad, None, (0, 0), 2)
        inside = pad[1:-1, 1:-1] != 2
        color = PALETTE[int(d["key"]) % len(PALETTE)]
        img[inside] = color
    a = (ink.astype(np.float32) / 255.0)[..., None]
    img = (img * (1 - a) + np.array(INK, np.float32) * a).astype(np.uint8)
    return img, (to_img, to_base)
