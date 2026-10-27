"""QuickDraw 선 낙서를 「아이가 크레파스 · 색연필로 칠한 종이 그림을 휴대폰으로 찍은 사진」처럼 바꾸는 변환.

    img, gt = make(strokes, variant, seed)
img: BGR 그림, gt: 정답 몸 마스크(선 + 선으로 닫힌 안쪽, 같은 좌표). 변환 이름은 VARIANTS.
변환마다 따로 한 가지씩 넣은 것과(thick … text), 섞은 것(mix_mild · mix_heavy)이 있다.
"""
import cv2
import numpy as np

CANVAS = 512  # 종이 한 장 (찍기 전)
BOX = 320  # 그림이 들어가는 크기: 옆에 글자를 쓸 자리가 남는다
# 크레파스 · 색연필 색 (BGR)
PALETTE = [(40, 40, 220), (30, 140, 250), (40, 210, 250), (60, 180, 60), (200, 120, 30), (160, 60, 120),
           (150, 110, 240), (40, 80, 140), (120, 120, 120), (200, 200, 60), (90, 200, 160), (60, 60, 60)]
LINE_DARK = [(20, 20, 20), (40, 30, 30), (70, 60, 50), (30, 30, 90), (90, 40, 20)]
WORDS = ["Mina 7", "MY DOG", "Jiho", "2026.10.27", "LOVE", "Seo-yun", "cat!!", "Hana 6 sae", "ABC", "mom"]

VARIANTS = ["orig", "thick", "linecolor", "fill", "crayon", "paper", "yellow", "lined",
            "shadow", "light", "tilt", "blur", "eraser", "text", "mix_mild", "mix_heavy"]
NAME = {"orig": "원본 (검은 선 6px, 흰 종이)", "thick": "선 굵기 3~14px", "linecolor": "색 선(크레파스 색 윤곽)",
        "fill": "안쪽 색칠(여러 색 · 삐짐 · 덜 칠함)", "crayon": "크레파스 결", "paper": "종이 결", "yellow": "노란 종이",
        "lined": "줄 · 모눈 공책", "shadow": "그림자", "light": "고르지 않은 조명", "tilt": "기울어짐(책상 보임)",
        "blur": "흐림 · 압축", "eraser": "지우개 자국", "text": "옆에 쓴 글자", "mix_mild": "섞음: 보통 사진",
        "mix_heavy": "섞음: 나쁜 사진"}


def _geom(strokes, rng, box=BOX):
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    sc = box / max(max(xs) - min(xs), max(ys) - min(ys), 1)
    w, h = sc * (max(xs) - min(xs)), sc * (max(ys) - min(ys))
    m = 24
    ox = rng.uniform(m, CANVAS - w - m) - sc * min(xs)
    oy = rng.uniform(m, CANVAS - h - m) - sc * min(ys)
    return [np.round(np.array([[x * sc + ox, y * sc + oy] for x, y in zip(*s)])).astype(np.int32) for s in strokes]


def _lines(polys, thick):
    cov = np.zeros((CANVAS, CANVAS), np.uint8)
    for p in polys:
        cv2.polylines(cov, [p], False, 255, int(thick), cv2.LINE_AA)
        if len(p) == 1:
            cv2.circle(cov, tuple(int(v) for v in p[0]), max(1, int(thick) // 2), 255, -1)
    return cov.astype(np.float32) / 255


def _interior(polys):
    """선(6px)으로 닫힌 안쪽. 15px 까지 끊긴 곳은 닫힌 것으로 본다 (아이는 선이 조금 열려 있어도 안을 칠한다)."""
    ln = (_lines(polys, 6) > 0.5).astype(np.uint8) * 255
    cl = cv2.morphologyEx(ln, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ff = np.pad(cl, 1)
    cv2.floodFill(ff, np.zeros((CANVAS + 4, CANVAS + 4), np.uint8), (0, 0), 128)
    inside = (ff[1:-1, 1:-1] != 128) & (cl == 0)
    return inside, ln > 0


def _smooth_noise(rng, scale, shape=(CANVAS, CANVAS)):
    n = rng.random((max(2, shape[0] // scale), max(2, shape[1] // scale))).astype(np.float32)
    return cv2.resize(n, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _tooth(rng, shape=(CANVAS, CANVAS), strength=0.5):
    """크레파스가 종이 결에 걸려 생기는 하얀 틈: 잔 점 + 한 방향 긁힌 결."""
    n = rng.random(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.8)
    k = np.zeros((9, 9), np.float32)
    ang = rng.uniform(0, np.pi)
    cv2.line(k, (int(4 - 4 * np.cos(ang)), int(4 - 4 * np.sin(ang))), (int(4 + 4 * np.cos(ang)), int(4 + 4 * np.sin(ang))), 1, 1)
    n = cv2.filter2D(n, -1, k / k.sum())
    n = (n - n.mean()) / (n.std() + 1e-6)
    return np.clip(1 - strength * np.clip(n * 0.9 + 0.3, 0, 3), 0, 1)


def _fill(polys, rng, inside, overflow=(0, 8), under=(0.05, 0.3)):
    """칸마다 다른 색으로 칠한다. 일부 칸은 안 칠하고, 칠은 선 밖으로 삐지거나 덜 칠한다."""
    n, lab = cv2.connectedComponents(inside.astype(np.uint8), connectivity=4)
    layers = []
    for k in range(1, n):
        reg = (lab == k).astype(np.uint8)
        if reg.sum() < 60 or rng.random() < 0.2:
            continue
        dil = int(rng.integers(overflow[0], overflow[1] + 1))
        if dil:
            reg = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
        dx, dy = rng.integers(-5, 6, 2)
        reg = np.roll(reg, (int(dy), int(dx)), (0, 1))
        hole = _smooth_noise(rng, int(rng.integers(12, 40))) < rng.uniform(*under)
        reg = reg.astype(np.float32) * (~hole)
        reg = cv2.GaussianBlur(reg, (0, 0), 1.2) * rng.uniform(0.6, 0.95)
        layers.append((reg, PALETTE[int(rng.integers(len(PALETTE)))]))
    return layers


def _multiply(img, cov, color):
    c = np.array(color, np.float32) / 255
    return img * (1 - cov[..., None] * (1 - c))


def _paper(rng, color=(250, 250, 250), grain=0.0, lined=False):
    img = np.empty((CANVAS, CANVAS, 3), np.float32)
    img[:] = color
    if grain:
        g = rng.normal(0, 1, (CANVAS, CANVAS)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.7) * grain + (_smooth_noise(rng, 64) - 0.5) * grain * 2
        img += g[..., None]
    if lined:
        if rng.random() < 0.5:  # 줄 공책
            step = int(rng.integers(22, 34))
            for y in range(int(rng.integers(10, step)), CANVAS, step):
                cv2.line(img, (0, y), (CANVAS, y), (230, 190, 160), 1, cv2.LINE_AA)
        else:  # 모눈
            step = int(rng.integers(14, 22))
            for t in range(int(rng.integers(0, step)), CANVAS, step):
                cv2.line(img, (0, t), (CANVAS, t), (220, 200, 185), 1, cv2.LINE_AA)
                cv2.line(img, (t, 0), (t, CANVAS), (220, 200, 185), 1, cv2.LINE_AA)
    return img


def _eraser(img, rng, other_polys):
    """지운 자국: 다른 그림의 연한 회색 흔적 + 문질러 번진 얼룩."""
    ghost = np.zeros((CANVAS, CANVAS), np.float32)
    for p in other_polys:
        cv2.polylines(ghost, [p], False, 1.0, int(rng.integers(4, 9)), cv2.LINE_AA)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 2.0) * rng.uniform(0.15, 0.3)
    for _ in range(int(rng.integers(2, 5))):
        c = tuple(int(v) for v in rng.integers(40, CANVAS - 40, 2))
        cv2.ellipse(ghost, c, tuple(int(v) for v in rng.integers(15, 50, 2)), float(rng.uniform(0, 180)), 0, 360, 0.12, -1)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 3)
    return _multiply(img, np.clip(ghost, 0, 0.35), (90, 90, 90))


def _text(img, rng, gt):
    """옆 빈 곳에 이름 · 날짜 같은 손글씨."""
    ys, xs = np.nonzero(gt)
    top, bot, left, right = ys.min(), CANVAS - ys.max(), xs.min(), CANVAS - xs.max()
    word = WORDS[int(rng.integers(len(WORDS)))]
    scale = rng.uniform(1.2, 2.2)
    th = int(rng.integers(3, 7))
    (tw, tht), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, th)
    if max(top, bot) >= tht + 12:
        y = int(rng.integers(tht + 4, max(tht + 5, top - 4))) if top >= bot else int(rng.integers(ys.max() + tht + 6, max(ys.max() + tht + 7, CANVAS - 6)))
        x = int(rng.integers(4, max(5, CANVAS - tw - 4)))
    else:
        y = int(rng.integers(tht + 6, CANVAS - 6))
        x = 4 if left >= right else max(4, CANVAS - tw - 4)
    cov = np.zeros((CANVAS, CANVAS), np.uint8)
    cv2.putText(cov, word, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, 255, th, cv2.LINE_AA)
    color = LINE_DARK[int(rng.integers(len(LINE_DARK)))] if rng.random() < 0.6 else PALETTE[int(rng.integers(len(PALETTE)))]
    return _multiply(img, cov.astype(np.float32) / 255 * 0.9, color)


def _light(img, rng, strength):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32) / max(img.shape[:2])
    a = rng.uniform(0, 2 * np.pi)
    lin = (np.cos(a) * (xx - 0.5) + np.sin(a) * (yy - 0.5))
    cx, cy = rng.uniform(0.2, 0.8, 2)
    rad = np.hypot(xx - cx, yy - cy)
    f = 1 - strength * (0.6 * (lin + 0.7) / 1.4 + 0.4 * np.clip(rad / 0.8, 0, 1))
    warm = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.95, 1.0), 1.0], np.float32)  # 형광등 · 전구 색
    return img * f[..., None] * warm


def _shadow(img, rng, strength):
    """폰 · 손 그림자: 가장자리에서 들어오는 큰 덩어리, 가장자리가 흐리다."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.float32)
    side = int(rng.integers(4))
    pts = []
    for t in np.linspace(0, 1, 6):
        if side == 0:
            pts.append((t * W, rng.uniform(0.2, 0.6) * H))
        elif side == 1:
            pts.append((t * W, H - rng.uniform(0.2, 0.6) * H))
        elif side == 2:
            pts.append((rng.uniform(0.2, 0.6) * W, t * H))
        else:
            pts.append((W - rng.uniform(0.2, 0.6) * W, t * H))
    corner = {0: [(W, 0), (0, 0)], 1: [(W, H), (0, H)], 2: [(0, H), (0, 0)], 3: [(W, H), (W, 0)]}[side]
    cv2.fillPoly(m, [np.array(pts + corner, np.int32)], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(8, 25))
    return img * (1 - strength * m)[..., None]


def _tilt(img, gt, rng, deg, keystone):
    """책상 위 종이를 비스듬히 찍음: 돌림 + 원근, 바깥은 책상."""
    S = int(CANVAS * 1.25)
    desk = np.empty((S, S, 3), np.float32)
    desk[:] = [rng.uniform(60, 140), rng.uniform(80, 160), rng.uniform(110, 190)]  # 나무 · 회색 책상
    desk += ((_smooth_noise(rng, 16, (S, S)) - 0.5) * 30)[..., None]
    c = np.float32([[0, 0], [CANVAS, 0], [CANVAS, CANVAS], [0, CANVAS]])
    a = np.radians(rng.uniform(-deg, deg))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    d = (c - CANVAS / 2) @ R.T * rng.uniform(0.85, 0.98) + S / 2
    d += rng.uniform(-keystone, keystone, (4, 2)).astype(np.float32) * CANVAS
    M = cv2.getPerspectiveTransform(c, d.astype(np.float32))
    paper = cv2.warpPerspective(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
    inside = cv2.warpPerspective(np.ones((CANVAS, CANVAS), np.float32), M, (S, S)) > 0.5
    out = np.where(inside[..., None], paper, desk)
    g2 = cv2.warpPerspective(gt.astype(np.uint8) * 255, M, (S, S), flags=cv2.INTER_NEAREST) > 127
    return out, g2


def _blur(img, rng, strength):
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.2) * strength)
    else:
        k = int(rng.integers(5, 10) * strength) | 1
        ker = np.zeros((k, k), np.float32)
        ang = rng.uniform(0, np.pi)
        cv2.line(ker, (int(k / 2 - k / 2 * np.cos(ang)), int(k / 2 - k / 2 * np.sin(ang))),
                 (int(k / 2 + k / 2 * np.cos(ang)), int(k / 2 + k / 2 * np.sin(ang))), 1, 1)
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1))
    return img


def _jpeg(img, rng, q=(45, 80), noise=3.0):
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*q))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make(strokes, variant, seed, other=None):
    """other: 지우개 자국용 다른 그림의 획 (없으면 같은 그림을 옮겨 씀).
    같은 seed 면 변환이 달라도 그림 자리 · 크기는 같다 (변환끼리 짝지어 비교)."""
    polys = _geom(strokes, np.random.default_rng(seed))
    rng = np.random.default_rng([seed, VARIANTS.index(variant)])
    inside, ln6 = _interior(polys)
    v = variant
    mild, heavy = v == "mix_mild", v == "mix_heavy"
    mix = mild or heavy
    thick = 6
    if v == "thick" or heavy:
        thick = int(rng.integers(3, 15))
    elif mild:
        thick = int(rng.integers(4, 11))
    line_col = (0, 0, 0)
    if v == "linecolor" or (mix and rng.random() < 0.5):
        line_col = PALETTE[int(rng.integers(len(PALETTE)))]
    elif mix:
        line_col = LINE_DARK[int(rng.integers(len(LINE_DARK)))]
    paper_col = (250, 250, 250)
    if v == "yellow" or (mild and rng.random() < 0.5) or heavy:
        paper_col = (rng.uniform(175, 215), rng.uniform(225, 242), rng.uniform(240, 252))
    grain = 6.0 if v == "paper" or mild else 9.0 if heavy else 0.0
    lined = v == "lined" or (heavy and rng.random() < 0.4)
    img = _paper(rng, paper_col, grain, lined)

    lines = _lines(polys, thick)
    gt = inside | (lines > 0.5) | ln6
    crayon = v == "crayon" or mix
    if v == "fill" or (mild and rng.random() < 0.85) or heavy:
        layers = _fill(polys, rng, inside, overflow=(0, 12) if heavy else (0, 8), under=(0.1, 0.45) if heavy else (0.05, 0.3))
        for cov, col in layers:
            if crayon:
                cov = cov * _tooth(rng, strength=0.6 if heavy else 0.45)
            img = _multiply(img, cov, col)
    lcov = lines * rng.uniform(0.75, 1.0) if mix else lines
    if crayon:
        lcov = lcov * _tooth(rng, strength=0.5 if heavy else 0.35)
    img = _multiply(img, lcov, line_col)
    if v == "eraser" or heavy:
        img = _eraser(img, rng, _geom(other, rng) if other else [p + rng.integers(-60, 61, 2) for p in polys])
    if v == "text" or heavy:
        img = _text(img, rng, gt)
    # --- 찍기 ---
    if v == "tilt" or mix:
        img, gt = _tilt(img, gt, rng, 15 if heavy or v == "tilt" else 5, 0.06 if heavy or v == "tilt" else 0.02)
    if v == "light" or mix:
        img = _light(img, rng, 0.45 if heavy or v == "light" else 0.2)
    if v == "shadow" or heavy:
        img = _shadow(img, rng, rng.uniform(0.3, 0.55))
    if v == "blur" or mix:
        img = _blur(img, rng, 1.0 if heavy or v == "blur" else 0.6)
    if v in ("blur", "tilt", "light", "shadow") or mix:
        img = _jpeg(img, rng, (40, 70) if heavy else (55, 85), 4.0 if heavy else 2.5)
    return np.clip(img, 0, 255).astype(np.uint8), gt
