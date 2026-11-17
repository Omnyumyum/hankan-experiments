"""서비스의 네 발 규칙(drawtale-ai runtime/ai_server/quadruped.py, 커밋 4c2147f)을 서버 없이 부르는 감싸기.

부르는 법:
  1. quadruped._detections 를 「찾은 상자 없음」([])으로 바꿔 끼운다. 서비스에서 Meta 탐지 서버가 상자를 하나도
     못 찾았을 때와 같은 길이다: 상자 = 선 범위(2~98 백분위), 바탕이 고르니 _grow 로 넓힌다.
  2. quadruped.cut_out(그림 파일) 로 마스크를 만든다 (create_mask · refine_mask · restore_lines ·
     drop_background · attach_near 를 서비스 그대로).
  3. quadruped.find_joints(mask > 127, facing=None) 로 관절 18개와 「다리를 찾았나」를 받는다.
     facing=None 은 머리 쪽을 규칙이 굵기로 짐작하는 것 (서비스에서 어른이 방향을 고쳐 주기 전 값).
저 저장소 코드는 고치지 않는다.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

DRAWTALE = Path(os.environ.get("DRAWTALE_AI", Path(__file__).resolve().parent / "work" / "drawtale-ai"))
sys.path.insert(0, str(DRAWTALE / "runtime"))

from ai_server import quadruped as Q  # noqa: E402

Q._detections = lambda img: []           # Meta 탐지 서버 대신: 찾은 상자 없음
cv2.setNumThreads(1)                     # 일꾼(프로세스) 여럿을 돌리므로 OpenCV 는 한 실만

ORDER = Q.ORDER


def run(img: np.ndarray) -> dict:
    """{"ok": 마스크를 만들었나, "found": 다리를 찾았나, "joints": (18,2) 배열, "box": 마스크 상자 (l,t,r,b)}"""
    tmp = Path(tempfile.gettempdir()) / f"qrule_{os.getpid()}.png"   # 일꾼마다 따로 (fork 뒤 pid 로)
    cv2.imwrite(str(tmp), img)
    try:
        cut = Q.cut_out(str(tmp))
    except Exception as e:                # 서비스에서도 실패로 끝나는 그림
        return {"ok": False, "found": False, "joints": None, "box": None, "error": repr(e)[:80]}
    if "error" in cut:
        return {"ok": False, "found": False, "joints": None, "box": None, "error": cut["error"]}
    m = cut["mask"] > 127
    try:
        j, found = Q.find_joints(m, None)
    except Exception as e:
        return {"ok": False, "found": False, "joints": None, "box": None, "error": repr(e)[:80]}
    ys, xs = np.where(m)
    return {"ok": True, "found": bool(found), "joints": np.array([j[n] for n in ORDER], float),
            "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}


# 빠르게: 종이색 재기(analyzer.paper_distance)는 그림 자리를 inpaint 로 메워 종이색을 짐작한다 (한 장 0.3초).
# 이 실험의 그림은 바탕이 완전한 흰색이라 짐작한 종이색이 늘 흰색(255,255,255)이다. 같은 값을 바로 낸다.
# 같은 결과인지는 check_fast.py 로 확인 (보고서에 적음).
_slow_paper = Q.A.paper_distance


def _fast_paper(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cv2.medianBlur(img, 3).astype(np.float32) - 255.0, axis=2)


def use_fast(on: bool = True) -> None:
    Q.A.paper_distance = _fast_paper if on else _slow_paper


use_fast(os.environ.get("QRULE_SLOW") != "1")
