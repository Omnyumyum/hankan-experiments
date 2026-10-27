"""날개 「점 3개」 흔들기용 시험 그림 세트 (경우의 수 넓히기).

모양 7가지(위에서 본 나비 · 앞모습 새 · 잠자리 · 박쥐 · 요정 · 옆모습 새(날개 하나) · 옆모습 나비(날개 하나))
× 결 3가지(색칠 · 윤곽선만 · 크레파스) × 무작위 씨앗. 씨앗 0 · 1 은 학습용(고칠 때 보는 것), 2 · 3 은 확인용(마지막에 한 번 재는 것).
그림마다 정답 점(날개마다 뿌리 위 · 뿌리 아래 · 끝)과 「움직여야 할 점 · 가만히 있어야 할 점」을 같이 만든다. 좌표는 원래 그림 좌표.
check_wings.py 의 18 조합(gen_wing_cases.py)은 건드리지 않는다.

    ../venv/bin/python gen_wing_point_cases.py      # drawings/cases_pts/ 에 그림 저장 (눈으로 보기용, 올리지 않음)
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "drawings" / "cases_pts"
INK = (40, 40, 40)
SHAPES = ("butterfly_top", "bird_front", "dragonfly", "bat", "fairy", "bird_side", "butterfly_side")
STYLES = ("color", "line", "crayon")
SPLIT = {"train": (0, 1), "test": (2, 3)}
PALETTE = [(90, 170, 250), (250, 160, 90), (120, 200, 120), (200, 120, 220), (80, 210, 240), (230, 200, 120), (160, 160, 250)]


def ell(c, ax, ang=0):
    return cv2.ellipse2Poly((int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), int(ang), 0, 360, 4).astype(np.int32)


def ell_x(c, ax, y):
    """세운 타원 (c, ax) 의 높이 y 에서 오른쪽 가장자리까지 가로 거리."""
    t = (y - c[1]) / ax[1]
    return ax[0] * np.sqrt(max(0.0, 1 - t * t))


def far_point(poly, c):
    j = np.argmax(((poly - c) ** 2).sum(1))
    return tuple(int(v) for v in poly[j])


def part(poly, color, role, deco=None):
    return {"poly": np.asarray(poly, np.int32), "color": color, "role": role, "deco": deco or []}


def crayon_line(img, poly, color, rng, th):
    """크레파스 선: 조금씩 흔들리고 굵기가 바뀌는 선."""
    p = poly.astype(np.float32)
    for a, b in zip(p, np.roll(p, -1, 0)):
        j = rng.normal(0, 1.2, 2)
        cv2.line(img, tuple(int(v) for v in a + j), tuple(int(v) for v in b + j), color, int(th + rng.integers(-1, 2)))


def render(parts, style, rng, size):
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    for pt in parts:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pt["poly"]], 255)
        col = pt["color"]
        if style == "line":
            img[m > 0] = 255
        elif style == "color":
            img[m > 0] = col
        else:  # 크레파스: 색 + 결 (짧은 빗금 · 얼룩)
            layer = np.full((h, w, 3), 255, np.uint8)
            layer[:] = col
            noise = rng.integers(-28, 28, (h, w))
            layer = np.clip(layer.astype(int) + noise[..., None], 0, 255).astype(np.uint8)
            for _ in range(int(m.sum() / 255 / 120)):
                ys, xs = np.nonzero(m)
                k = rng.integers(len(xs))
                a = rng.uniform(0, np.pi)
                L = rng.uniform(6, 16)
                p0 = (int(xs[k]), int(ys[k]))
                p1 = (int(xs[k] + L * np.cos(a)), int(ys[k] + L * np.sin(a)))
                shade = tuple(int(max(0, c - rng.integers(25, 60))) for c in col)
                cv2.line(layer, p0, p1, shade, 2)
            # 칠이 가장자리까지 꽉 차지 않는다 (흰 틈이 군데군데)
            gaps = (rng.random((h, w)) < 0.04).astype(np.uint8)
            gaps = cv2.dilate(gaps, np.ones((2, 2), np.uint8)) > 0
            mm = (m > 0) & ~gaps
            img[mm] = layer[mm]
            img[(m > 0) & gaps] = 255
        # 무늬 · 날개맥: 이 부분 안에만
        if pt["deco"]:
            dl = img.copy()
            for kind, *args in pt["deco"]:
                c2 = INK if style != "crayon" else tuple(int(c * 0.45) for c in col)
                if kind == "line":
                    cv2.line(dl, args[0], args[1], c2, 3)
                elif kind == "circle":
                    if style == "line":
                        cv2.circle(dl, args[0], args[1], INK, 3)
                    else:
                        cv2.circle(dl, args[0], args[1], args[2], -1)
                        cv2.circle(dl, args[0], args[1], c2, 3)
            inner = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
            img[inner] = dl[inner]
        if style == "crayon":
            if pt["role"] != "wing" or rng.random() < 0.5:  # 크레파스 날개는 윤곽선이 없을 때가 많다
                crayon_line(img, pt["poly"], tuple(int(c * 0.4) for c in col), rng, 4)
        else:
            cv2.polylines(img, [pt["poly"]], True, INK, 4 if style == "color" else 3)
    return img


def veins(rng, root, tip_poly, n):
    """뿌리 쪽에서 날개 끝 쪽으로 뻗은 날개맥 n 줄."""
    out = []
    root = np.asarray(root, float)
    idx = np.linspace(0, len(tip_poly) - 1, n + 2)[1:-1].astype(int)
    for j in idx:
        p = np.asarray(tip_poly[j], float)
        out.append(("line", tuple(int(v) for v in root + 0.15 * (p - root)), tuple(int(v) for v in p)))
    return out


def wing_deco(rng, root, poly, color):
    deco = []
    if rng.random() < 0.6:
        deco += veins(rng, root, poly, int(rng.integers(2, 5)))
    if rng.random() < 0.5:
        c = poly.mean(0)
        r = int(rng.uniform(10, 22))
        deco.append(("circle", (int(c[0]), int(c[1])), r, tuple(int(255 - (255 - v) * 0.5) for v in color)))
    return deco


def butterfly_top(rng):
    W, H, cx, cy = 520, 440, 260, 220
    bw, bh = rng.uniform(14, 22), rng.uniform(85, 105)
    cf, ch = PALETTE[rng.integers(len(PALETTE))], PALETTE[rng.integers(len(PALETTE))]
    parts, wings, move = [], [], []
    fax, fay, fang = rng.uniform(80, 100), rng.uniform(60, 78), rng.uniform(10, 30)
    hax, hay = rng.uniform(52, 66), rng.uniform(42, 54)
    for s in (-1, 1):
        fc = (cx + s * (fax * 0.95 + bw * 0.4), cy - rng.uniform(55, 70))
        hc = (cx + s * (hax * 0.9 + bw * 0.3), cy + rng.uniform(55, 70))
        fp, hp = ell(fc, (fax, fay), -s * fang), ell(hc, (hax, hay), s * rng.uniform(15, 30))
        root = (cx + s * bw, cy - 20)
        parts += [part(hp, ch, "wing", wing_deco(rng, root, hp, ch)), part(fp, cf, "wing", wing_deco(rng, root, fp, cf))]
        y_top = cy - min(bh * 0.7, fay + 20)
        a = (cx + s * (ell_x((cx, cy), (bw, bh), y_top + 15) - 2), y_top + 15)
        b = (cx + s * (ell_x((cx, cy), (bw, bh), cy + 55) - 2), cy + 55)
        wings.append([a, b, far_point(fp, (cx, cy))])
        move += [(fc[0] + s * 0.4 * fax, fc[1]), (hc[0] + s * 0.35 * hax, hc[1] + 0.2 * hay)]
    parts.append(part(ell((cx, cy), (bw, bh)), (110, 110, 110), "body"))
    hr = rng.uniform(18, 24)
    parts.append(part(ell((cx, cy - bh - hr + 6), (hr, hr)), (110, 110, 110), "head"))
    return parts, wings, move, [(cx, cy), (cx, cy - bh - hr + 6)], "fold", (W, H)


def bird_front(rng):
    W, H, cx, cy = 560, 400, 280, 235
    bw, bh = rng.uniform(48, 60), rng.uniform(62, 75)
    col = PALETTE[rng.integers(len(PALETTE))]
    parts, wings, move = [], [], []
    up = rng.uniform(-30, 25)
    span = rng.uniform(200, 245)
    for s in (-1, 1):
        poly = [(cx + s * 30, cy - 40), (cx + s * 140, cy - 110 + up), (cx + s * span, cy - 95 + up * 1.2),
                (cx + s * (span - 30), cy - 70 + up), (cx + s * (span - 70), cy - 50 + up * 0.8), (cx + s * 150, cy - 35 + up * 0.5),
                (cx + s * 35, cy + 15)]
        poly = np.array(poly, np.int32)
        root = (cx + s * bw, cy - 15)
        parts.append(part(poly, col, "wing", wing_deco(rng, root, poly, col) if rng.random() < 0.7 else []))
        a = (cx + s * (ell_x((cx, cy), (bw, bh), cy - 35) - 4), cy - 35)
        b = (cx + s * (ell_x((cx, cy), (bw, bh), cy + 8) - 4), cy + 8)
        wings.append([a, b, far_point(poly, (cx, cy))])
        move.append((cx + s * (span - 40), cy - 85 + up))
    parts.append(part(ell((cx, cy), (bw, bh)), PALETTE[rng.integers(len(PALETTE))], "body"))
    hr = rng.uniform(32, 40)
    parts.append(part(ell((cx, cy - bh - hr + 15), (hr, hr)), col, "head"))
    return parts, wings, move, [(cx, cy + 10), (cx, cy - bh - hr + 15)], "updown", (W, H)


def dragonfly(rng):
    W, H, cx = 560, 420, 280
    bw, bh, cy = rng.uniform(11, 16), rng.uniform(130, 150), 230
    col = PALETTE[rng.integers(len(PALETTE))]
    parts, wings, move = [], [], []
    L = rng.uniform(100, 120)
    y1, y2 = 105 + rng.uniform(-8, 8), 155 + rng.uniform(-8, 8)
    wing_col = (250, 245, 235) if rng.random() < 0.5 else PALETTE[rng.integers(len(PALETTE))]
    for s in (-1, 1):
        tips = []
        for yy, ang in ((y1, -rng.uniform(6, 16)), (y2, rng.uniform(6, 16))):
            c = (cx + s * (L + 8), yy)
            p = ell(c, (L, rng.uniform(18, 25)), s * ang)
            parts.append(part(p, wing_col, "wing", veins(rng, (cx, yy), p, 2) if rng.random() < 0.5 else []))
            tips.append(far_point(p, (cx, yy)))
            move.append((c[0] + s * 0.4 * L, c[1] + s * 0.4 * L * np.tan(np.radians(s * ang))))
        a = (cx + s * (bw - 2), y1 - 10)
        b = (cx + s * (bw - 2), y2 + 10)
        wings.append([a, b, tips[0]])
    parts.append(part(ell((cx, cy), (bw, bh)), col, "body"))
    hr = rng.uniform(18, 24)
    parts.append(part(ell((cx, cy - bh - hr + 8), (hr, hr)), col, "head"))
    return parts, wings, move, [(cx, cy + 60), (cx, cy - bh - hr + 8)], "updown", (W, H)


def bat(rng):
    W, H, cx, cy = 560, 380, 280, 200
    bw, bh = rng.uniform(32, 42), rng.uniform(55, 70)
    col = PALETTE[rng.integers(len(PALETTE))] if rng.random() < 0.6 else (90, 70, 80)
    parts, wings, move = [], [], []
    span, lift = rng.uniform(210, 250), rng.uniform(40, 90)
    for s in (-1, 1):
        top = [(cx + s * 20, cy - 35), (cx + s * 110, cy - lift - 10), (cx + s * span, cy - lift)]
        # 아래 가장자리: 손가락 사이가 파인 물결 3번
        n = 3
        xs = np.linspace(span, 25, n + 1)
        bottom = []
        for i in range(n):
            x0, x1 = xs[i], xs[i + 1]
            for t in np.linspace(0, 1, 7)[1:]:
                x = x0 + (x1 - x0) * t
                y = cy + 25 - 45 * (1 - t) * (i == 0) - 25 * np.sin(np.pi * t)
                bottom.append((cx + s * x, y))
        poly = np.array(top + bottom, np.int32)
        root = (cx + s * bw, cy - 10)
        parts.append(part(poly, col, "wing", veins(rng, root, poly[2:9], 2) if rng.random() < 0.6 else []))
        a = (cx + s * (ell_x((cx, cy), (bw, bh), cy - 30) - 3), cy - 30)
        b = (cx + s * (ell_x((cx, cy), (bw, bh), cy + 20) - 3), cy + 20)
        wings.append([a, b, (cx + s * span, cy - lift)])
        move.append((cx + s * (span - 50), cy - lift + 25))
    parts.append(part(ell((cx, cy), (bw, bh)), col, "body"))
    hr = rng.uniform(24, 30)
    hc = (cx, cy - bh - hr + 18)
    ears = []
    for s in (-1, 1):
        ears.append(part([(hc[0] + s * 6, hc[1] - hr + 6), (hc[0] + s * hr, hc[1] - hr - 22), (hc[0] + s * (hr - 2), hc[1] - 4)], col, "head"))
    parts += ears + [part(ell(hc, (hr, hr)), col, "head")]
    return parts, wings, move, [(cx, cy + 15), hc], "updown", (W, H)


def fairy(rng):
    W, H, cx, cy = 520, 460, 260, 280
    bw, bh = rng.uniform(26, 34), rng.uniform(85, 100)
    gap = rng.choice([0, 0, 10])  # 몸에 붙은 날개 · 떨어진 날개
    col = PALETTE[rng.integers(len(PALETTE))]
    parts, wings, move = [], [], []
    ax, ay, ang = rng.uniform(65, 85), rng.uniform(42, 55), rng.uniform(20, 40)
    for s in (-1, 1):
        x_edge = ell_x((cx, cy), (bw, bh), cy - 55)
        c = (cx + s * (x_edge + gap + ax * 0.85), cy - 55 - ax * 0.35)
        p = ell(c, (ax, ay), s * ang)
        parts.append(part(p, col, "wing", wing_deco(rng, (cx, cy - 55), p, col)))
        if gap:
            inner = p[np.argsort(s * p[:, 0])[:12]]  # 몸 쪽 가장자리
            a, b = inner[np.argmin(inner[:, 1])], inner[np.argmax(inner[:, 1])]
            a, b = (a[0] + s * 3, a[1]), (b[0] + s * 3, b[1])
        else:
            a = (cx + s * (ell_x((cx, cy), (bw, bh), cy - 80) - 3), cy - 80)
            b = (cx + s * (ell_x((cx, cy), (bw, bh), cy - 30) - 3), cy - 30)
        wings.append([tuple(int(v) for v in a), tuple(int(v) for v in b), far_point(p, (cx, cy - 55))])
        move.append((c[0] + s * 0.45 * ax, c[1] - 0.25 * ax))
    parts.append(part(ell((cx, cy), (bw, bh)), PALETTE[rng.integers(len(PALETTE))], "body"))
    hr = rng.uniform(30, 36)
    parts.append(part(ell((cx, cy - bh - hr + 10), (hr, hr)), (200, 220, 250), "head"))
    return parts, wings, move, [(cx, cy + 20), (cx, cy - bh - hr + 10)], "updown", (W, H)


def bird_side(rng):
    """옆모습 새: 날개 하나가 등 위로 들림 (날개 뒤에 몸)."""
    W, H, cx, cy = 520, 420, 250, 280
    bw, bh = rng.uniform(100, 120), rng.uniform(52, 64)
    col = PALETTE[rng.integers(len(PALETTE))]
    parts = []
    x0, x1 = cx - rng.uniform(30, 50), cx + rng.uniform(40, 60)
    tip = (cx + rng.uniform(40, 110), cy - bh - rng.uniform(120, 170))
    poly = np.array([(x0, cy - 20), (tip[0] - 70, tip[1] + 30), tip, (tip[0] + 10, tip[1] + 50), (x1, cy - 20)], np.int32)
    wcol = PALETTE[rng.integers(len(PALETTE))]
    parts.append(part(poly, wcol, "wing", veins(rng, (cx, cy - bh), poly, 3) if rng.random() < 0.6 else []))
    parts.append(part(np.array([(cx + bw - 10, cy - 10), (cx + bw + 60, cy - 40), (cx + bw + 55, cy + 20)], np.int32), col, "body"))  # 꼬리
    parts.append(part(ell((cx, cy), (bw, bh)), col, "body"))
    hc = (cx - bw - 5, cy - bh + 5)
    parts.append(part(np.array([(hc[0] - 30, hc[1] - 6), (hc[0] - 65, hc[1] + 4), (hc[0] - 30, hc[1] + 14)], np.int32), (60, 170, 240), "head"))
    parts.append(part(ell(hc, (34, 34)), col, "head"))
    a = (x0 + 5, cy - np.sqrt(max(0, 1 - ((x0 + 5 - cx) / bw) ** 2)) * bh + 4)  # 등 윤곽선 바로 안쪽
    b = (x1 - 5, cy - np.sqrt(max(0, 1 - ((x1 - 5 - cx) / bw) ** 2)) * bh + 4)
    move = [(0.3 * ((x0 + x1) / 2) + 0.7 * tip[0], 0.3 * (cy - bh) + 0.7 * tip[1] + 15)]
    return parts, [[a, b, tip]], move, [(cx, cy + 10), hc, (cx + bw + 40, cy - 10)], "updown", (W, H)


def butterfly_side(rng):
    """옆모습 나비: 날개를 위로 모은 한 장 (앞날개 + 뒷날개), 가는 몸."""
    W, H, cx, cy = 520, 420, 260, 330
    bw, bh = rng.uniform(85, 100), rng.uniform(12, 16)
    cf, ch = PALETTE[rng.integers(len(PALETTE))], PALETTE[rng.integers(len(PALETTE))]
    fc = (cx - rng.uniform(5, 25), cy - rng.uniform(120, 140))
    hc = (cx + rng.uniform(35, 55), cy - rng.uniform(55, 70))
    fp = ell(fc, (rng.uniform(70, 85), rng.uniform(105, 125)), rng.uniform(-25, -10))
    hp = ell(hc, (rng.uniform(50, 62), rng.uniform(52, 64)), rng.uniform(10, 30))
    root = (cx, cy - bh)
    parts = [part(hp, ch, "wing", wing_deco(rng, root, hp, ch)), part(fp, cf, "wing", wing_deco(rng, root, fp, cf))]
    parts.append(part(ell((cx, cy), (bw, bh)), (110, 110, 110), "body"))
    hc2 = (cx - bw - 12, cy - 6)
    parts.append(part(ell(hc2, (18, 18)), (110, 110, 110), "head"))
    a = (cx - 45, cy - bh + 3)
    b = (cx + 45, cy - bh + 3)
    tip = far_point(fp, root)
    # 뒷날개는 뿌리 바로 옆이라 위아래로 돌려도 조금만 움직인다 → 앞날개 쪽 점만 본다
    move = [(root[0] + 0.65 * (tip[0] - root[0]), root[1] + 0.65 * (tip[1] - root[1]))]
    return parts, [[a, b, tip]], move, [(cx, cy), hc2, (cx + bw - 10, cy)], "updown", (W, H)


MAKERS = {k: globals()[k] for k in SHAPES}


def make(shape, style, seed):
    """→ (그림 BGR, {"wings", "move", "stay", "style"}). 같은 인자면 늘 같은 그림."""
    rng = np.random.default_rng(1000 * SHAPES.index(shape) + 10 * STYLES.index(style) + seed)
    parts, wings, move, stay, flap, size = MAKERS[shape](rng)
    img = render(parts, style, rng, size)
    wings = [[tuple(float(v) for v in p) for p in w] for w in wings]
    return img, {"wings": wings, "move": [tuple(map(float, p)) for p in move], "stay": [tuple(map(float, p)) for p in stay], "style": flap}


def names(split="all"):
    seeds = SPLIT[split] if split in SPLIT else SPLIT["train"] + SPLIT["test"]
    return [f"{sh}-{st}-{sd}" for sh in SHAPES for st in STYLES for sd in seeds]


def by_name(name):
    sh, st, sd = name.split("-")
    return make(sh, st, int(sd))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for nm in names():
        img, info = by_name(nm)
        cv2.imwrite(str(OUT / f"{nm}.png"), img)
        v = img.copy()
        for w in info["wings"]:
            for j, p in enumerate(w):
                cv2.circle(v, tuple(int(c) for c in p), 6, (0, 160, 0) if j < 2 else (0, 110, 240), -1)
        for p in info["move"]:
            cv2.circle(v, tuple(int(c) for c in p), 5, (0, 200, 0), 2)
        for p in info["stay"]:
            cv2.drawMarker(v, tuple(int(c) for c in p), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        v = cv2.resize(v, (260, 210))
        cv2.putText(v, nm, (3, 12), 0, 0.4, (0, 0, 200), 1)
        tiles.append(v)
    rows = [np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]
    (Path(__file__).parent / "out").mkdir(exist_ok=True)
    cv2.imwrite(str(Path(__file__).parent / "out" / "wing_point_cases.png"), np.vstack(rows))
    print("→", OUT, len(tiles), "장, 정답 점 모아 보기 out/wing_point_cases.png")
