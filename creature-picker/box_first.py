"""사진 → 찾기 상자(Meta 사람 그림 찾기 모델) 안에서만 모양 따기 + 「누름을 물을지」 신호 (INTEGRATE4.md).

    import find_box as fb, box_first as bf
    det = fb.load()
    d, info = bf.drawing(img, det)       # d = rig_types.Drawing (상자로 자른 그림 또는 사진 전체), info = 상자 · 자른 곳 · 누름 물음
    if info["ask_tap"]:                  # 화면: 「그림 한가운데를 눌러 주세요」 → MobileSAM 에 누른 점 + info["box"] (eval_real/box_eval.py 참고)
        ...
규칙(학습용에서 정함):
  - 찾기 상자(점수 0.5 이상 중 가장 높은 것)가 있으면 상자를 너비 · 높이의 MARGIN 만큼 넓혀 자르고 Drawing.
  - 그 마스크와 사진 전체 Drawing 마스크가 거의 안 겹치면(IoU < MIN_AGREE) 상자를 버리고 사진 전체 결과를 쓴다.
  - 상자를 썼는데 두 마스크 IoU < ASK_T 이면 누름을 묻는다. 상자를 안 썼으면 마스크가 비었을 때만.
두 번 Drawing 을 만들므로 사진 정리 · 모양 따기 시간이 두 배다(한 장 0.3 ~ 0.6초 더).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import find_box as fb
import rig_types as rt

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval_real"))
import colored_eval as ce  # noqa: E402  (자른 곳 계산을 재기 스크립트와 똑같이 쓰려고)

MARGIN = 0.1
MIN_AGREE = 0.1
ASK_T = 0.4


def _full_mask(img, d):
    """Drawing 마스크를 img 좌표로 (eval_real/colored_eval.mask_iou 와 같은 계산: 정리한 사진에서 자른 곳을 다시 구해 되돌림)."""
    x0, y0, x1, y1 = ce.crop_box(rt.clean_photo(img))
    m = ((d.mask > 0) | (d.alpha > 0)).astype(np.uint8)
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return full


def _try(img):
    try:
        d = rt.Drawing(img)
        return d, _full_mask(img, d)
    except Exception:
        return None, np.zeros(img.shape[:2], bool)


def expand(box, shape, f):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h, w = shape
    return (max(0, int(np.floor(x0 - f * bw))), max(0, int(np.floor(y0 - f * bh))),
            min(w, int(np.ceil(x1 + f * bw))), min(h, int(np.ceil(y1 + f * bh))))


def drawing(img, det):
    box, score, _ = fb.detect_top(det, img)
    d_full, m_full = _try(img)
    info = dict(box=box, score=score, used_box=False, crop=None, agree=0.0, ask_tap=False)
    if box is not None:
        x0, y0, x1, y1 = expand(box, img.shape[:2], MARGIN)
        d_box, m = _try(np.ascontiguousarray(img[y0:y1, x0:x1]))
        if d_box is not None and m.any():
            mb = np.zeros(img.shape[:2], bool)
            mb[y0:y1, x0:x1] = m
            u = (mb | m_full).sum()
            info["agree"] = float((mb & m_full).sum() / u) if u else 0.0
            if info["agree"] >= MIN_AGREE:
                info.update(used_box=True, crop=(x0, y0, x1, y1), ask_tap=info["agree"] < ASK_T)
                return d_box, info
    info["ask_tap"] = d_full is None or not m_full.any()
    if d_full is None:
        raise ValueError("그림을 찾지 못함")
    return d_full, info
