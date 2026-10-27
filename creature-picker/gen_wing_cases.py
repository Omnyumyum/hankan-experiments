"""날개 칸 고르기 경우의 수 시험 그림 → drawings/cases/
check_wings.py 가 그림마다 고를 칸(누르기 · 문지르기)과 「움직여야 할 점 · 가만히 있어야 할 점」을 이 파일의 CASES 에서 읽는다.
좌표는 모두 원래 그림 좌표 (잘라 낸 그림 좌표로는 check_wings.py 가 바꾼다)."""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "drawings" / "cases"
INK = (40, 40, 40)


def blank(w=520, h=440):
    return np.full((h, w, 3), 255, np.uint8)


def lineart_butterfly():
    """선만 있는 위에서 본 나비. 날개 안에 동그라미 무늬(갇힌 칸)."""
    img = blank()
    for s in (-1, 1):
        cv2.ellipse(img, (260 + s * 100, 160), (95, 75), s * -20, 0, 360, INK, 4)
        cv2.circle(img, (260 + s * 110, 160), 25, INK, 3)
        cv2.ellipse(img, (260 + s * 75, 290), (60, 50), s * 25, 0, 360, INK, 4)
        cv2.circle(img, (260 + s * 80, 295), 15, INK, 3)
        cv2.line(img, (260 + s * 8, 95), (260 + s * 45, 40), INK, 3)
    cv2.ellipse(img, (260, 220), (18, 100), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (260, 220), (18, 100), 0, 0, 360, INK, 4)
    cv2.circle(img, (260, 105), 22, (255, 255, 255), -1)
    cv2.circle(img, (260, 105), 22, INK, 4)
    return img


def broken_bird():
    """앞모습 새, 왼쪽 날개 윤곽선이 크게(10px) 끊김 → 그 칸은 바깥과 이어져 칸이 안 된다 (점 3개 방식으로 가야 함)."""
    img = blank(560, 400)
    body = (280, 230)
    cv2.ellipse(img, body, (55, 70), 0, 0, 360, (230, 200, 120), -1)
    cv2.ellipse(img, body, (55, 70), 0, 0, 360, (150, 100, 40), 4)
    for s in (-1, 1):
        wing = np.array([[280 + s * 40, 200], [280 + s * 150, 120], [280 + s * 245, 140], [280 + s * 160, 190], [280 + s * 45, 245]], np.int32)
        cv2.fillPoly(img, [wing], (255, 255, 255))
        cv2.polylines(img, [wing], True, (150, 100, 40), 4)
    cv2.line(img, (90, 130), (100, 133), (255, 255, 255), 8)  # 왼쪽 날개 윗선 끊김
    cv2.circle(img, (280, 140), 38, (230, 200, 120), -1)
    cv2.circle(img, (280, 140), 38, (150, 100, 40), 4)
    cv2.circle(img, (266, 132), 5, (0, 0, 0), -1)
    cv2.circle(img, (294, 132), 5, (0, 0, 0), -1)
    return img


def fairy():
    """몸에서 떨어져 그린 날개 (요정 · 천사). 날개와 몸 사이 12px 틈."""
    img = blank()
    cv2.ellipse(img, (260, 260), (30, 90), 0, 0, 360, (200, 170, 250), -1)
    cv2.ellipse(img, (260, 260), (30, 90), 0, 0, 360, INK, 4)
    cv2.circle(img, (260, 140), 35, (255, 255, 255), -1)
    cv2.circle(img, (260, 140), 35, INK, 4)
    for s in (-1, 1):
        cv2.ellipse(img, (260 + s * 120, 210), (75, 50), s * 30, 0, 360, (250, 230, 160), -1)
        cv2.ellipse(img, (260 + s * 120, 210), (75, 50), s * 30, 0, 360, INK, 4)
    return img


def crayon_wings():
    """날개에 검은 윤곽선 없이 색만 칠함 (크레파스 느낌). 몸만 윤곽선."""
    img = blank()
    rng = np.random.default_rng(3)
    for s in (-1, 1):
        cv2.ellipse(img, (260 + s * 105, 190), (90, 70), s * -15, 0, 360, (90, 170, 250), -1)
        noise = rng.integers(-25, 25, img.shape[:2])
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(m, (260 + s * 105, 190), (90, 70), s * -15, 0, 360, 255, -1)
        img[m > 0] = np.clip(img[m > 0].astype(int) + noise[m > 0, None], 0, 255).astype(np.uint8)
    cv2.ellipse(img, (260, 220), (20, 100), 0, 0, 360, (120, 120, 120), -1)
    cv2.ellipse(img, (260, 220), (20, 100), 0, 0, 360, INK, 4)
    cv2.circle(img, (260, 110), 24, (120, 120, 120), -1)
    cv2.circle(img, (260, 110), 24, INK, 4)
    return img


def dragonfly():
    """잠자리: 가는 날개 네 장이 따로, 긴 몸."""
    img = blank(560, 400)
    cv2.ellipse(img, (280, 220), (14, 140), 0, 0, 360, (120, 200, 120), -1)
    cv2.ellipse(img, (280, 220), (14, 140), 0, 0, 360, INK, 4)
    cv2.circle(img, (280, 80), 22, (120, 200, 120), -1)
    cv2.circle(img, (280, 80), 22, INK, 4)
    for s in (-1, 1):
        for dy, ang in ((120, -12), (165, 12)):
            c = (280 + s * 120, dy)
            cv2.ellipse(img, c, (110, 22), s * ang, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(img, c, (110, 22), s * ang, 0, 360, INK, 3)
    return img


# 이름: (그림 함수 또는 기존 파일, 누르기, 문지르기, 움직여야 할 점, 가만히 있어야 할 점, 날개짓, 기대)
# 기대: "ok" = 위 점 검사 통과, "empty" = 칸이 안 잡혀야 정상(선이 크게 끊김 → 점 3개 방식 안내)
CASES = {
    "lineart_butterfly": (lineart_butterfly, [(160, 130), (360, 130), (225, 300), (295, 300)], [],  # 뒷날개는 무늬 동그라미 밖을 누른다
                          [(110, 120), (410, 120), (150, 160), (370, 160), (200, 320), (340, 300)], [(260, 220), (260, 105)], "fold", "ok"),
    "broken_bird": (broken_bird, [(130, 160), (430, 160)], [],
                    [(480, 150)], [(280, 230), (280, 140)], "updown", "partial"),
    "fairy": (fairy, [(150, 190), (370, 190)], [],
              [(110, 175), (410, 175)], [(260, 260), (260, 140)], "updown", "ok"),
    "crayon_wings": (crayon_wings, [(170, 190), (350, 190)], [],
                     [(110, 175), (410, 175)], [(260, 220), (260, 110)], "fold", "ok"),
    "dragonfly": (dragonfly, [(160, 115), (160, 170), (400, 115), (400, 170)], [],
                  [(190, 105), (190, 175), (370, 105), (370, 175)], [(280, 220), (280, 80)], "updown", "ok"),
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (fn, *_rest) in CASES.items():
        cv2.imwrite(str(OUT / f"{name}.png"), fn())
    print("→", OUT, len(CASES), "장")
