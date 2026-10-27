"""quadruped.py 경우의 수 시험 그림 → drawings_quad/. 아이가 그릴 만한 네발 캐릭터를 모양을 바꿔 가며 만든다.

몸 · 머리 · 다리 · 꼬리 · 귀 · 코를 한 마스크로 합쳐 칠하고 윤곽을 그린다 (아이 그림처럼 윤곽은 살짝 떨린다).
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "drawings_quad"
OUT.mkdir(exist_ok=True)
S = 768
rng = np.random.default_rng(7)


def leg(m, x, y0, y1, w, slant=0):
    pts = np.array([[x - w // 2, y0], [x + w // 2, y0], [x + w // 2 + slant, y1], [x - w // 2 + slant, y1]], np.int32)
    cv2.fillConvexPoly(m, pts, 255)


def tail(m, pts, w):
    cv2.polylines(m, [np.array(pts, np.int32)], False, 255, w)


def render(name, m, fill=(80, 170, 230), outline=True, eyes=(), mouth=None, flip=False, line=6):
    if flip:
        m = m[:, ::-1].copy()
        eyes = [(S - x, y) for x, y in eyes]
        mouth = None if mouth is None else (S - mouth[0], mouth[1])
    img = np.full((S, S, 3), 250, np.uint8)
    if fill is not None:
        img[m > 0] = fill
    if outline:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = c + rng.integers(-1, 2, c.shape)  # 손 떨림
            cv2.polylines(img, [c], True, (30, 30, 30), line)
    for x, y in eyes:
        cv2.circle(img, (int(x), int(y)), 9, (20, 20, 20), -1)
    if mouth is not None:
        cv2.ellipse(img, (int(mouth[0]), int(mouth[1])), (14, 8), 0, 0, 180, (20, 20, 20), 3)
    cv2.imwrite(str(OUT / f"{name}.png"), img)


def base_dog(m, legs=True, tail_on=True, leg_len=170, body=(380, 360, 190, 80)):
    cx, cy, rx, ry = body
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    cv2.circle(m, (cx + rx + 40, cy - 70), 75, 255, -1)  # 머리 (오른쪽)
    if legs:
        for x in (cx - rx + 45, cx - rx + 105, cx + rx - 105, cx + rx - 45):
            leg(m, x, cy + 30, cy + ry + leg_len - 80 + 30, 34)
    if tail_on:
        tail(m, [(cx - rx + 10, cy - 20), (cx - rx - 50, cy - 70), (cx - rx - 70, cy - 130)], 18)


cases = {}

m = np.zeros((S, S), np.uint8); base_dog(m, tail_on=False)
cases["q_notail_right"] = dict(m=m, eyes=[(640, 270)], mouth=(645, 315))
cases["q_notail_left"] = dict(m=m, eyes=[(640, 270)], mouth=(645, 315), flip=True, fill=(120, 200, 120))

m = np.zeros((S, S), np.uint8); base_dog(m)
cases["q_tail_outline_only"] = dict(m=m, eyes=[(640, 270)], fill=None, line=4)
cases["q_tail_fill_no_outline"] = dict(m=m, eyes=[(640, 270)], outline=False, fill=(60, 120, 200))

# 닥스훈트: 몸 길고 다리 짧음
m = np.zeros((S, S), np.uint8)
cv2.ellipse(m, (370, 420), (270, 70), 0, 0, 360, 255, -1)
cv2.circle(m, (660, 360), 65, 255, -1)
for x in (150, 200, 540, 590):
    leg(m, x, 460, 560, 30)
tail(m, [(105, 400), (60, 360)], 14)
cases["q_dachshund_short_legs"] = dict(m=m, eyes=[(680, 345)])

# 말: 다리 길고 목 길게 위로
m = np.zeros((S, S), np.uint8)
cv2.ellipse(m, (360, 330), (170, 70), 0, 0, 360, 255, -1)
cv2.fillConvexPoly(m, np.array([[480, 300], [530, 300], [610, 150], [570, 130]], np.int32), 255)
cv2.ellipse(m, (610, 145), (60, 30), -20, 0, 360, 255, -1)
for x, s in ((230, -10), (280, 10), (450, -10), (500, 10)):
    leg(m, x, 360, 680, 26, s)
tail(m, [(195, 310), (160, 360), (150, 450)], 16)
cases["q_horse_long_legs"] = dict(m=m, eyes=[(615, 135)], fill=(60, 100, 160))

# 다리가 두 개만 보이게 겹친 그림 (굵은 다리)
m = np.zeros((S, S), np.uint8)
cv2.ellipse(m, (380, 360), (200, 85), 0, 0, 360, 255, -1)
cv2.circle(m, (610, 290), 80, 255, -1)
for x in (250, 510):
    leg(m, x, 400, 560, 70)
cases["q_two_legs_visible_notail"] = dict(m=m, eyes=[(630, 270)], fill=(200, 160, 220))

# 풀 뜯기: 머리가 배보다 아래, 땅 가까이
m = np.zeros((S, S), np.uint8)
cv2.ellipse(m, (350, 330), (180, 75), 0, 0, 360, 255, -1)
cv2.fillConvexPoly(m, np.array([[490, 300], [530, 330], [600, 520], [555, 535]], np.int32), 255)
cv2.circle(m, (590, 560), 55, 255, -1)
for x in (220, 270, 430, 480):
    leg(m, x, 370, 620, 30)
tail(m, [(175, 320), (130, 340)], 14)
cases["q_grazing_head_down"] = dict(m=m, eyes=[(610, 550)], fill=(240, 240, 240))

# 코끼리: 코가 땅까지, 꼬리 짧음
m = np.zeros((S, S), np.uint8)
cv2.ellipse(m, (360, 340), (200, 110), 0, 0, 360, 255, -1)
cv2.circle(m, (580, 300), 90, 255, -1)
cv2.polylines(m, [np.array([(640, 340), (670, 450), (660, 560), (680, 600)], np.int32)], False, 255, 30)
for x in (220, 290, 430, 500):
    leg(m, x, 400, 580, 55)
cases["q_elephant_trunk_to_ground"] = dict(m=m, eyes=[(600, 280)], fill=(170, 170, 180))

# 귀가 길게 늘어진 강아지 (귀 끝이 배보다 아래)
m = np.zeros((S, S), np.uint8); base_dog(m, tail_on=False)
cv2.ellipse(m, (580, 330), (28, 95), 10, 0, 360, 255, -1)
cases["q_long_ears_notail"] = dict(m=m, eyes=[(640, 270)], fill=(150, 110, 70))

# 머리가 몸보다 큰 캐릭터, 짧고 굵은 다리, 꼬리 없음
m = np.zeros((S, S), np.uint8)
cv2.ellipse(m, (330, 430), (150, 70), 0, 0, 360, 255, -1)
cv2.circle(m, (520, 330), 140, 255, -1)
for x in (220, 270, 380, 430):
    leg(m, x, 470, 560, 40)
cases["q_bighead_chibi_notail"] = dict(m=m, eyes=[(500, 300), (580, 300)], mouth=(545, 370), fill=(250, 200, 120))

for name, c in cases.items():
    render(name, c.pop("m"), **c)
print(len(cases), "장 →", OUT)
