"""긴 몸 똬리 · 꼬인 뱀 시험 그림 → drawings/long/
몸 가운데 선(꼬리 → 머리)을 따라 「윤곽 원 → 칠 원」을 차례로 찍어 그린다. 나중에 지나간 몸의 윤곽이 먼저 지나간 몸 위에 그어져,
몸이 닿은 곳에는 아이 그림처럼 선 하나가 생긴다.
TRUTH: 그림마다 머리 · 꼬리 정답 점 (원래 그림 좌표). check_points_jitter.py 가 읽는다.

    venv/bin/python gen_long_cases.py
"""
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "drawings" / "long"
INK = (40, 40, 40)
GREEN = (90, 190, 110)
ORANGE = (60, 150, 240)


def spiral(c, r0, gap, turns, a0=0.0, step=1.5):
    """가운데(r0)에서 바깥으로 도는 나선. 바퀴 사이 거리 gap."""
    pts, a = [], a0
    while a < a0 + turns * 2 * math.pi:
        r = r0 + gap * (a - a0) / (2 * math.pi)
        pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
        a += step / max(r, 5)
    return pts


def line(p, q, step=1.5):
    n = max(2, int(math.dist(p, q) / step))
    return [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(1, n + 1)]


def bezier(p0, p1, p2, p3, n=200):
    out = []
    for i in range(1, n + 1):
        t = i / n
        out.append(tuple((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)))
    return out


def draw_body(img, path, w, fill, outline=4, taper=0.25, chunk=110):
    """path: 꼬리 → 머리. 꼬리 쪽은 가늘게(taper 만큼 길이 동안). fill=None 이면 윤곽만.
    chunk px 씩 「윤곽 → 칠」을 그린다. 칠은 앞 조각 끝을 몸 굵기만큼 덮어 조각 사이에 금이 안 생기게."""
    n = len(path)
    rad = [w / 2 * min(1.0, 0.25 + 0.75 * i / max(1, taper * n)) for i in range(n)]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))])
    starts = list(range(0, int(s[-1]) + 1, chunk))
    for c0 in starts:
        idx = np.nonzero((s >= c0) & (s < c0 + chunk))[0]
        back = np.nonzero((s >= c0 - 1.2 * w) & (s < c0 + chunk))[0]
        for i in idx:
            cv2.circle(img, (int(round(path[i][0])), int(round(path[i][1]))), int(round(rad[i] + outline)), INK, -1, cv2.LINE_AA)
        for i in back:
            cv2.circle(img, (int(round(path[i][0])), int(round(path[i][1]))), int(round(rad[i])), fill if fill is not None else (255, 255, 255), -1, cv2.LINE_AA)
    return path[0], path[-1]


def head(img, p, d, w, fill):
    """머리: 몸 끝에서 d 방향으로 조금 큰 타원 + 눈."""
    d = np.array(d, float) / np.linalg.norm(d)
    c = np.array(p) + d * w * 0.3
    ang = math.degrees(math.atan2(d[1], d[0]))
    cv2.ellipse(img, tuple(int(v) for v in c), (int(w * 0.85), int(w * 0.62)), ang, 0, 360, INK, -1, cv2.LINE_AA)
    cv2.ellipse(img, tuple(int(v) for v in c), (int(w * 0.85) - 4, int(w * 0.62) - 4), ang, 0, 360, fill if fill is not None else (255, 255, 255), -1, cv2.LINE_AA)
    n = np.array([-d[1], d[0]])
    for s in (-1, 1):
        e = c + d * w * 0.3 + n * s * w * 0.28
        cv2.circle(img, tuple(int(v) for v in e), 4, INK, -1, cv2.LINE_AA)
    return tuple(float(v) for v in c + d * w * 0.8)  # 머리 끝


def coil(fill, name, gap_ratio=1.0, w=34):
    """똬리: 꼬리는 가운데, 두 바퀴 반 감고 바깥에서 목이 위로 들린다. gap_ratio=1 이면 바퀴끼리 닿는다."""
    img = np.full((460, 460, 3), 255, np.uint8)
    c = (215, 270)
    sp = spiral(c, 14, (w + 8) * gap_ratio, 2.3, a0=math.pi / 2)
    neck = bezier(sp[-1], (sp[-1][0] + 60, sp[-1][1] - 70), (330, 60), (250, 60))
    path = sp + neck
    tail, end = draw_body(img, path, w, fill, taper=0.15)
    tip = head(img, end, np.subtract(path[-1], path[-6]), w, fill)
    return img, {"head": tip, "tail": tail}


def coil_line():
    """한 줄로 그은 똬리 (바퀴 사이 14px): 틈 메우기로 원판이 되던 경우."""
    img = np.full((460, 460, 3), 255, np.uint8)
    c = (200, 250)
    sp = spiral(c, 10, 26, 3.2, a0=0.0)
    tail = bezier(sp[-1], (sp[-1][0] + 40, sp[-1][1] + 60), (380, 420), (430, 400))
    path = [(int(round(x)), int(round(y))) for x, y in sp + tail]
    cv2.polylines(img, [np.array(path, np.int32)], False, INK, 6, cv2.LINE_AA)
    cv2.circle(img, path[0], 9, INK, -1, cv2.LINE_AA)  # 머리는 가운데 점
    return img, {"head": tuple(map(float, path[0])), "tail": tuple(map(float, path[-1]))}


def hairpin(fill):
    """몸이 한 번 접혀 위아래 몸이 길게 맞닿은 뱀."""
    img = np.full((320, 560, 3), 255, np.uint8)
    w = 36
    r = (w + 6) / 2  # 접힌 위아래 몸이 윤곽선 하나를 사이에 두고 맞닿게
    path = line((200, 210), (420, 210)) + [(420 + r * math.sin(a), 210 - r + r * math.cos(a)) for a in np.linspace(0, math.pi, 120)][1:] \
        + line((420, 210 - 2 * r), (90, 210 - 2 * r))
    tail, end = draw_body(img, path, w, fill, taper=0.12)
    tip = head(img, end, (-1, 0), w, fill)
    return img, {"head": tip, "tail": tail}


def loop_knot(fill):
    """S 자로 오다가 몸이 한 번 고리를 만들며 제 몸 위를 지나간다 (꼬인 뱀)."""
    img = np.full((420, 560, 3), 255, np.uint8)
    w = 30
    a = bezier((40, 330), (140, 360), (220, 330), (280, 260))
    loop = [(280 + 80 * math.cos(t) - 80, 180 + 80 * math.sin(t)) for t in np.linspace(math.pi * 0.0, 0, 2)]  # 자리만
    c = (300, 180)
    lp = [(c[0] + 75 * math.cos(t), c[1] + 75 * math.sin(t)) for t in np.linspace(math.radians(110), math.radians(110 + 330), 300)]
    b = bezier(a[-1], (300, 230), lp[0], lp[0], 40)[:-1] + lp
    rest = bezier(lp[-1], (420, 300), (470, 280), (520, 200))
    path = a + b + rest
    del loop
    tail, end = draw_body(img, path, w, fill, taper=0.2)
    tip = head(img, end, np.subtract(path[-1], path[-6]), w, fill)
    return img, {"head": tip, "tail": tail}


CASES = {
    "coil_colored.png": lambda: coil(GREEN, "coil_colored"),
    "coil_outline.png": lambda: coil(None, "coil_outline"),
    "coil_loose_colored.png": lambda: coil(ORANGE, "coil_loose", gap_ratio=1.35),
    "coil_line.png": coil_line,
    "hairpin_colored.png": lambda: hairpin(GREEN),
    "loop_knot_outline.png": lambda: loop_knot(None),
}
TRUTH = {}


def make(write=True):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in CASES.items():
        img, pts = fn()
        TRUTH[name] = {k: [round(v[0], 1), round(v[1], 1)] for k, v in pts.items()}
        if write:
            cv2.imwrite(str(OUT / name), img)
    return TRUTH


if __name__ == "__main__":
    for k, v in make().items():
        print(k, v)
