"""날개 시험 그림 2장 (animal-monster 그림에는 날개가 몸 밖으로 튀어나온 그림이 없다) → drawings/"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "drawings"
OUT.mkdir(exist_ok=True)


def butterfly():
    img = np.full((420, 520, 3), 255, np.uint8)
    for s in (-1, 1):
        cv2.ellipse(img, (260 + s * 95, 170), (95, 75), s * -20, 0, 360, (90, 160, 250), -1)
        cv2.ellipse(img, (260 + s * 95, 170), (95, 75), s * -20, 0, 360, (40, 80, 200), 3)
        cv2.ellipse(img, (260 + s * 70, 280), (60, 50), s * 25, 0, 360, (120, 200, 255), -1)
        cv2.ellipse(img, (260 + s * 70, 280), (60, 50), s * 25, 0, 360, (40, 80, 200), 3)
        cv2.line(img, (260, 110), (260 + s * 40, 50), (40, 40, 40), 3)
    cv2.ellipse(img, (260, 220), (16, 95), 0, 0, 360, (60, 60, 60), -1)
    cv2.circle(img, (260, 112), 22, (60, 60, 60), -1)
    cv2.circle(img, (252, 108), 4, (255, 255, 255), -1)
    cv2.circle(img, (268, 108), 4, (255, 255, 255), -1)
    return img


def flying_bird():
    img = np.full((400, 560, 3), 255, np.uint8)
    body = (280, 230)
    cv2.ellipse(img, body, (55, 70), 0, 0, 360, (230, 200, 120), -1)
    cv2.ellipse(img, body, (55, 70), 0, 0, 360, (150, 100, 40), 3)
    for s in (-1, 1):
        wing = np.array([[280 + s * 40, 200], [280 + s * 150, 120], [280 + s * 245, 140], [280 + s * 160, 190], [280 + s * 45, 245]], np.int32)
        cv2.fillPoly(img, [wing], (230, 200, 120))
        cv2.polylines(img, [wing], True, (150, 100, 40), 3)
    cv2.circle(img, (280, 140), 38, (230, 200, 120), -1)
    cv2.circle(img, (280, 140), 38, (150, 100, 40), 3)
    cv2.circle(img, (266, 132), 5, (0, 0, 0), -1)
    cv2.circle(img, (294, 132), 5, (0, 0, 0), -1)
    cv2.fillPoly(img, [np.array([[270, 150], [290, 150], [280, 168]], np.int32)], (0, 140, 255))
    return img


if __name__ == "__main__":
    cv2.imwrite(str(OUT / "wings_butterfly.png"), butterfly())
    cv2.imwrite(str(OUT / "wings_bird_front.png"), flying_bird())
    print("→", OUT)
