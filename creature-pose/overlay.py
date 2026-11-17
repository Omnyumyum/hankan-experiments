"""그림 위에 규칙 관절(뼈대)을 겹쳐 그린다. 모아 보기 그림 만들 때 쓴다.

색 (BGR): 몸통 축 엉덩이-어깨-목 = 검정 굵은 선 + 보라 점
  뒷다리 바깥(left_b) 빨강 · 뒷다리 안(right_b) 주황 · 앞다리 안(right_f) 파랑 · 앞다리 바깥(left_f) 하늘
  꼬리 = 초록. 다리마다 위(네모) · 무릎(동그라미) · 발(채운 동그라미)
"""

import cv2
import numpy as np

import rule

IDX = {n: i for i, n in enumerate(rule.ORDER)}
LEG_COLORS = {"left_b": (40, 40, 230), "right_b": (0, 150, 255), "right_f": (230, 90, 30), "left_f": (230, 210, 40)}


def draw(img: np.ndarray, J: np.ndarray, found: bool = True, scale: float = 1.0) -> np.ndarray:
    out = img.copy()
    P = lambda n: tuple(int(round(v * scale)) for v in J[IDX[n]])  # noqa: E731
    lw = max(2, int(round(3 * scale)))
    cv2.line(out, P("hip"), P("torso"), (20, 20, 20), lw + 1, cv2.LINE_AA)
    cv2.line(out, P("torso"), P("neck"), (20, 20, 20), lw + 1, cv2.LINE_AA)
    for leg, col in LEG_COLORS.items():
        side, fb = leg.split("_")
        a, k, f = P(f"{side}_hip_{fb}"), P(f"{side}_knee_{fb}"), P(f"{side}_foot_{fb}")
        cv2.line(out, a, k, col, lw, cv2.LINE_AA)
        cv2.line(out, k, f, col, lw, cv2.LINE_AA)
        cv2.rectangle(out, (a[0] - 4, a[1] - 4), (a[0] + 4, a[1] + 4), col, 2)
        cv2.circle(out, k, 4, col, 2, cv2.LINE_AA)
        cv2.circle(out, f, 5, col, -1, cv2.LINE_AA)
    t0, t1, t2 = P("tail"), P("tail_mid"), P("tail_tip")
    if t0 != t2:
        cv2.line(out, P("hip"), t0, (40, 170, 40), lw, cv2.LINE_AA)
        cv2.line(out, t0, t1, (40, 170, 40), lw, cv2.LINE_AA)
        cv2.line(out, t1, t2, (40, 170, 40), lw, cv2.LINE_AA)
        cv2.circle(out, t2, 5, (40, 170, 40), -1, cv2.LINE_AA)
    for n in ("hip", "torso", "neck"):
        cv2.circle(out, P(n), 6, (200, 40, 160), -1, cv2.LINE_AA)
    # 머리 쪽 표시: 목에서 바깥으로 작은 화살표
    nx, ny = P("neck")
    hx = 1 if J[IDX["neck"]][0] > J[IDX["hip"]][0] else -1
    cv2.arrowedLine(out, (nx, ny), (nx + hx * int(28 * scale), ny), (200, 40, 160), 2, cv2.LINE_AA, tipLength=0.4)
    if not found:
        cv2.putText(out, "NOT FOUND", (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
