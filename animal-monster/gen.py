"""아이가 그린 듯한 동물·괴물 낙서를 만든다 (흔들리는 선, 종이 바탕)."""
import math
import random
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "drawings"
OUT.mkdir(exist_ok=True)
W = H = 768


def paper():
    img = np.full((H, W, 3), (232, 238, 240), np.uint8)
    noise = np.random.normal(0, 4, (H, W, 1)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def wobble(pts, amp=2.5):
    out = []
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 6))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append((x0 + (x1 - x0) * t + random.uniform(-amp, amp),
                        y0 + (y1 - y0) * t + random.uniform(-amp, amp)))
    out.append(pts[-1])
    return np.array(out, np.int32)


def line(img, pts, color, th=5):
    cv2.polylines(img, [wobble(pts)], False, color, th, cv2.LINE_AA)


def ellipse(img, c, ax, color, th=5, ang=0, start=0, end=360):
    cx, cy = c
    a, b = ax
    r = math.radians(ang)
    pts = []
    for t in np.linspace(math.radians(start), math.radians(end), 60):
        x, y = a * math.cos(t), b * math.sin(t)
        pts.append((cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)))
    line(img, pts, color, th)


def dot(img, c, r, color):
    cv2.circle(img, (int(c[0]), int(c[1])), r, color, -1, cv2.LINE_AA)


def eyes(img, c, gap=18, color=(30, 30, 30)):
    dot(img, (c[0] - gap, c[1]), 6, color)
    dot(img, (c[0] + gap, c[1]), 6, color)


def smile(img, c, w=20, color=(40, 40, 200)):
    ellipse(img, c, (w, 10), color, 4, 0, 20, 160)


BROWN, BLACK, BLUE, GREEN, PURPLE, ORANGE, RED = (40, 80, 140), (40, 40, 40), (180, 90, 30), (60, 150, 60), (150, 50, 130), (30, 130, 230), (40, 40, 200)


def dog():
    img = paper()
    ellipse(img, (380, 400), (150, 75), BROWN)          # 몸
    ellipse(img, (565, 300), (65, 58), BROWN)           # 머리
    line(img, [(520, 260), (500, 200), (545, 245)], BROWN)  # 귀
    dot(img, (585, 290), 6, BLACK); dot(img, (628, 310), 9, BLACK)
    for x in (270, 320, 440, 490):
        line(img, [(x, 460), (x - 5, 580), (x + 25, 585)], BROWN)
    line(img, [(232, 380), (170, 320), (150, 290)], BROWN)  # 꼬리
    return img


def cat():
    img = paper()
    ellipse(img, (384, 470), (110, 140), ORANGE)       # 몸 (앉음, 정면)
    ellipse(img, (384, 270), (90, 80), ORANGE)         # 머리
    line(img, [(310, 230), (300, 160), (350, 200)], ORANGE)
    line(img, [(418, 200), (468, 160), (458, 230)], ORANGE)
    eyes(img, (384, 260), 30); smile(img, (384, 300), 15)
    for s in (-1, 1):
        line(img, [(384 + s * 20, 290), (384 + s * 90, 280)], BLACK, 2)
    line(img, [(330, 600), (330, 640), (360, 640)], ORANGE)
    line(img, [(438, 600), (438, 640), (408, 640)], ORANGE)
    line(img, [(494, 520), (580, 480), (600, 400)], ORANGE)  # 꼬리
    return img


def bird():
    img = paper()
    ellipse(img, (370, 380), (120, 85), BLUE)
    ellipse(img, (500, 290), (50, 45), BLUE)
    line(img, [(545, 285), (590, 295), (545, 305)], ORANGE)
    dot(img, (510, 280), 6, BLACK)
    line(img, [(330, 360), (250, 250), (420, 330)], BLUE)    # 날개
    line(img, [(255, 400), (170, 360), (200, 430), (255, 410)], BLUE)  # 꼬리
    line(img, [(350, 462), (345, 560), (320, 575)], ORANGE)
    line(img, [(400, 462), (405, 560), (430, 575)], ORANGE)
    return img


def giraffe():
    img = paper()
    ellipse(img, (330, 450), (130, 60), ORANGE)
    line(img, [(430, 420), (480, 260), (500, 170)], ORANGE)
    line(img, [(450, 450), (510, 280), (530, 190)], ORANGE)
    ellipse(img, (530, 160), (50, 30), ORANGE, ang=-15)
    dot(img, (540, 150), 5, BLACK)
    for x in (230, 270, 390, 430):
        line(img, [(x, 500), (x, 650)], ORANGE)
    for c in [(300, 440), (360, 460), (250, 450), (480, 300)]:
        dot(img, c, 12, BROWN)
    return img


def fish():
    img = paper()
    ellipse(img, (380, 384), (160, 90), GREEN)
    line(img, [(222, 384), (130, 300), (140, 470), (222, 390)], GREEN)
    dot(img, (480, 360), 8, BLACK); smile(img, (510, 410), 15)
    line(img, [(360, 300), (330, 230), (420, 295)], GREEN)
    return img


def monster_five_legs():
    img = paper()
    ellipse(img, (384, 330), (170, 110), PURPLE)
    eyes(img, (384, 300), 40); smile(img, (384, 360), 40)
    for i, x in enumerate((250, 310, 384, 460, 520)):
        line(img, [(x, 430), (x + random.randint(-30, 30), 560), (x + 20, 600)], PURPLE)
    line(img, [(330, 225), (300, 150)], PURPLE); dot(img, (300, 145), 10, PURPLE)
    line(img, [(440, 225), (470, 150)], PURPLE); dot(img, (470, 145), 10, PURPLE)
    return img


def monster_two_heads():
    img = paper()
    ellipse(img, (384, 450), (130, 120), GREEN)
    ellipse(img, (290, 250), (70, 65), GREEN); eyes(img, (290, 245), 22); smile(img, (290, 275), 15)
    ellipse(img, (480, 250), (70, 65), GREEN); eyes(img, (480, 245), 22); smile(img, (480, 275), 15)
    line(img, [(320, 312), (340, 340)], GREEN); line(img, [(450, 312), (430, 340)], GREEN)
    line(img, [(256, 430), (150, 380), (120, 330)], GREEN)
    line(img, [(512, 430), (620, 470), (660, 520)], GREEN)
    line(img, [(340, 568), (330, 680)], GREEN); line(img, [(430, 568), (440, 680)], GREEN)
    return img


def monster_octopus():
    img = paper()
    ellipse(img, (384, 280), (150, 130), RED)
    dot(img, (384, 260), 22, BLACK); dot(img, (390, 255), 7, (255, 255, 255))   # 외눈
    smile(img, (384, 330), 40)
    for i in range(7):
        x0 = 260 + i * 41
        pts = [(x0, 390)]
        for k in range(1, 6):
            pts.append((x0 + 25 * math.sin(k + i), 390 + k * 50))
        line(img, pts, RED)
    return img


def monster_blob():
    img = paper()
    pts = []
    for t in np.linspace(0, 2 * math.pi, 40):
        r = 170 + 25 * math.sin(5 * t) + random.uniform(-10, 10)
        pts.append((384 + r * math.cos(t), 400 + 0.8 * r * math.sin(t)))
    line(img, pts, BLUE)
    for c in [(320, 360), (400, 340), (450, 400)]:
        dot(img, c, 9, BLACK)
    smile(img, (384, 460), 50)
    return img


def monster_spiky():
    img = paper()
    ellipse(img, (384, 400), (120, 120), ORANGE)
    for k in range(9):
        a = -math.pi / 2 + (k - 4) * 0.35
        x0, y0 = 384 + 120 * math.cos(a), 400 + 120 * math.sin(a)
        line(img, [(x0, y0), (384 + 200 * math.cos(a), 400 + 200 * math.sin(a))], ORANGE)
    eyes(img, (384, 380), 35); smile(img, (384, 440), 40)
    line(img, [(300, 490), (260, 580), (300, 620)], ORANGE)
    line(img, [(468, 490), (508, 580), (468, 620)], ORANGE)
    line(img, [(264, 400), (180, 420)], ORANGE)
    return img


def monster_eye_stalks():
    img = paper()
    ellipse(img, (384, 450), (140, 90), PURPLE)
    for x, top in [(300, 200), (384, 160), (470, 220)]:
        line(img, [(x, 370), (x + random.randint(-20, 20), top)], PURPLE)
        ellipse(img, (x, top - 18), (20, 20), PURPLE); dot(img, (x, top - 18), 7, BLACK)
    smile(img, (384, 470), 50)
    for x in (300, 470):
        line(img, [(x, 535), (x, 620), (x + 30, 625)], PURPLE)
    return img


def tube(img, center, width, color, head=None, th=5):
    """가운데 선을 따라 양옆으로 테두리를 그린 몸 (아이들이 뱀 · 지렁이를 그리는 흔한 방식).
    width[i] 는 그 점의 반폭, head 는 (앞 끝 반지름) — 없으면 둥근 끝."""
    c = np.array(center, float)
    d = np.gradient(c, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    w = np.asarray(width, float)[:, None]
    left, right = c + n * w, c - n * w
    # 앞 끝(마지막 점)은 반원으로 닫고, 뒤 끝은 뾰족하거나 둥글게
    a0 = math.atan2(n[-1, 1], n[-1, 0])
    cap = [(c[-1, 0] + w[-1, 0] * math.cos(a0 - t), c[-1, 1] + w[-1, 0] * math.sin(a0 - t)) for t in np.linspace(0, math.pi, 12)]
    b0 = math.atan2(-n[0, 1], -n[0, 0])
    tail = [(c[0, 0] + w[0, 0] * math.cos(b0 - t), c[0, 1] + w[0, 0] * math.sin(b0 - t)) for t in np.linspace(0, math.pi, 8)]
    pts = list(map(tuple, left)) + cap + list(map(tuple, right[::-1])) + tail
    line(img, pts, color, th)
    return c, n, w[:, 0]


def snake_s():
    """S자로 기어가는 뱀. 꼬리는 가늘고 머리는 둥글다."""
    img = paper()
    t = np.linspace(0, 1, 60)
    cx = 110 + 540 * t
    cy = 400 + 110 * np.sin(2.2 * math.pi * t)
    w = 6 + 30 * np.clip(t * 1.6, 0, 1)
    c, n, w = tube(img, np.stack([cx, cy], 1), w, GREEN)
    hx, hy = c[-1]
    dot(img, (hx + 4, hy - 12), 6, BLACK)
    line(img, [(hx + 30, hy), (hx + 60, hy - 4), (hx + 72, hy - 14)], RED, 3)  # 혀
    line(img, [(hx + 60, hy - 4), (hx + 74, hy + 4)], RED, 3)
    for k in range(8, 52, 7):                                                # 무늬
        dot(img, c[k], 6, ORANGE)
    return img


def snake_line():
    """한 줄로 그은 뱀 — 테두리 없이 굵은 선 하나, 끝에 눈."""
    img = paper()
    t = np.linspace(0, 1, 40)
    pts = [(120 + 520 * s, 420 + 90 * math.sin(3 * math.pi * s)) for s in t]
    line(img, pts, GREEN, 9)
    dot(img, (pts[-1][0] + 2, pts[-1][1] - 14), 6, BLACK)
    line(img, [pts[-1], (pts[-1][0] + 30, pts[-1][1] + 5)], RED, 3)
    return img


def snake_coil():
    """똬리 튼 뱀 — 몸이 빙글 돌아 위로 머리를 든다."""
    img = paper()
    t = np.linspace(0, 1, 70)
    ang = 2.6 * math.pi * t
    r = 170 - 120 * t
    cx = 384 + r * np.cos(ang)
    cy = 470 + 0.55 * r * np.sin(ang)
    up = np.stack([np.full(14, cx[-1]), cy[-1] - np.linspace(0, 200, 14) ** 1.0], 1)
    up[:, 0] += 20 * np.sin(np.linspace(0, math.pi, 14))
    c = np.vstack([np.stack([cx, cy], 1), up[1:]])
    w = np.concatenate([6 + 22 * np.clip(t * 1.5, 0, 1), np.full(13, 28.0)])
    c, n, w = tube(img, c, w, BROWN)
    hx, hy = c[-1]
    eyes(img, (hx, hy - 6), 12)
    line(img, [(hx, hy + 28), (hx + 5, hy + 55)], RED, 3)
    return img


def worm():
    """지렁이 — 굵기가 고르고 양 끝이 둥글며 마디 줄이 있다."""
    img = paper()
    t = np.linspace(0, 1, 50)
    cx = 150 + 470 * t
    cy = 420 - 70 * np.sin(1.6 * math.pi * t)
    c, n, w = tube(img, np.stack([cx, cy], 1), np.full(50, 34.0), (150, 120, 230))
    for k in range(5, 45, 5):                                                # 마디
        a, b = c[k] + n[k] * (w[k] - 4), c[k] - n[k] * (w[k] - 4)
        line(img, [tuple(a), tuple(b)], (150, 120, 230), 3)
    hx, hy = c[-1]
    dot(img, (hx + 2, hy - 10), 6, BLACK); dot(img, (hx + 16, hy - 4), 6, BLACK)
    smile(img, (hx + 8, hy + 10), 10)
    return img


MAKERS = {
    "animal_dog": dog, "animal_cat": cat, "animal_bird": bird, "animal_giraffe": giraffe, "animal_fish": fish,
    "monster_5legs": monster_five_legs, "monster_2heads": monster_two_heads, "monster_octopus": monster_octopus,
    "monster_blob": monster_blob, "monster_spiky": monster_spiky, "monster_eyestalks": monster_eye_stalks,
    # 팔다리 없는 긴 몸 (2026-10-19). 앞 11장과 난수 순서가 같도록 맨 뒤에 둔다
    "snake_s": snake_s, "snake_line": snake_line, "snake_coil": snake_coil, "worm": worm,
}

if __name__ == "__main__":
    random.seed(3); np.random.seed(3)
    for name, fn in MAKERS.items():
        cv2.imwrite(str(OUT / f"{name}.png"), fn())
    # 실제 아이 그림 두 장 (Meta 저장소 예제)
    root = Path(r"C:\Users\Omnyumyum\Desktop\hankan-story\AnimatedDrawings\examples")
    cv2.imwrite(str(OUT / "real_pig.png"), cv2.imread(str(root / "characters/char6/image.png")))
    g = cv2.imread(str(root / "drawings/garlic.png"))
    cv2.imwrite(str(OUT / "real_garlic.png"), cv2.resize(g, (756, 1008)))
    print(sorted(p.name for p in OUT.iterdir()))
