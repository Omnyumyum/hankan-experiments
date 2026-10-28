"""색을 아는 마스크: 색칠한 아이 그림을 종이에서 잘라 낸다.

지금 마스크(shape_rig.crop_and_mask)는 흑백 적응 문턱으로 선을 찾고 「색칠 더하기」로 보충한다.
밝은 색(노랑 · 분홍 · 하늘색)은 흑백으로는 종이와 비슷해 선으로 안 잡히고, 색 보충은 종이 · 책상 색까지 끌어온다.
여기서는 종이 색(최빈값)과 조명(좁은 중앙값 바탕)을 먼저 정하고, Lab 에서 「종이가 아닌 것」(바탕보다 어두움 | 종이와 색이 다름)을 찾는다.

    crop, mesh, alpha = crop_and_mask(img, notes)            # 원본 사진을 받는다 (clean_photo 를 안에서 부른다)
    crop, mesh, alpha, box = crop_and_mask_box(img, notes)   # box: 원본 좌표의 자른 곳 (x0, y0, x1, y1)
돌려주는 crop · mesh · alpha 는 shape_rig.crop_and_mask 와 같은 모양(긴 변 400px)이다.
Drawing 에 넣으려면 `img = clean_photo(img, notes); crop, mask, alpha = sr.crop_and_mask(img, notes)` 두 줄을
`crop, mask, alpha = color_mask.crop_and_mask(img, notes)` 로 바꾼다 (이번에는 넣지 않았다: eval_real/COLOR_MASK.md).
gate=True 면 색칠한 그림일 때만 이 방식을 쓰고, 아니면 지금 방식을 그대로 쓴다 (시험한 갈래. 더 낫지 않아 기본은 끔).
"""
import math

import cv2
import numpy as np

import rig_types as rt

sr = rt.sr
WORK = 512  # 그림 고르기 · 채우기는 긴 변 512px 에서
GATE = 0.15  # gate=True 일 때: 그림 영역 중 굵게 칠한 곳이 이만큼 넘으면 색칠한 그림으로 본다
AGREE = None  # 지금 방식과 겹침(IoU)이 이보다 작으면 지금 방식을 쓴다 (시험 중)


def _disk(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_holes(m):
    ff = np.pad(m, 1)
    cv2.floodFill(ff, None, (0, 0), 128)
    m = m.copy()
    m[ff[1:-1, 1:-1] == 0] = 255
    return m


def paper_model(sm, paper):
    """종이 색과 조명. 돌려주는 것: Lab, 「바탕보다 어두운 정도」 dark, 종이 a · b."""
    size = max(sm.shape[:2])
    lab = cv2.cvtColor(cv2.GaussianBlur(sm, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    # 조명: 좁은 중앙값 바탕 (그림자 경계를 따라간다). 닫기 바탕보다 25 넘게 어둡지 않게 (clean_photo 와 같은 생각)
    # 넓게 흐린 바탕은 그림자 경계를 흐려서 경계가 초승달 모양 덩어리로 잡혔다 (QuickDraw 나쁜 사진 변환)
    k = max(5, int(0.025 * size) | 1)
    bgC = cv2.blur(cv2.morphologyEx(L, cv2.MORPH_CLOSE, _disk(k)), (k, k))
    bg = np.maximum(cv2.medianBlur(L.astype(np.uint8), k).astype(np.float32), bgC - 25)
    pL = float(np.median(bgC[paper]))
    # 넓게 진하게 칠한 곳은 바탕도 어두워져 안 잡힌다 → 종이 밝기보다 60 넘게 어두운 곳은 바로 그림
    dark = np.maximum(np.maximum(bg - L, 0), np.where(pL - L > 60, pL - L, 0))
    # 종이 색 = 종이 안 · 밝은 곳 a·b 의 가장 흔한 값(최빈값). 노란 · 분홍 종이도 종이로 본다
    # (알려진 약점: 그림이 사진을 거의 채우고 넓게 밝은 색으로 칠하면 그 색이 최빈값이 된다. 가장자리 띠에서 보는 안은 학습용에서 더 낫지 않았다)
    sel = paper & (L >= np.percentile(L[paper], 40))
    ab = lab[..., 1:][sel]
    q = np.round(ab / 2).astype(np.int32) + 200
    keys, cnt = np.unique(q[:, 0] * 1000 + q[:, 1], return_counts=True)
    top = keys[np.argmax(cnt)]
    pa, pb = 2.0 * (top // 1000 - 200), 2.0 * (top % 1000 - 200)
    near = np.hypot(ab[:, 0] - pa, ab[:, 1] - pb) < 6
    if near.sum() > 50:
        pa, pb = np.median(ab[near], axis=0)
    return lab, dark, float(pa), float(pb)


def ink_map(sm, paper):
    """「종이가 아닌 것」(0/1): 바탕보다 어두운 곳(선) | 종이와 색(a·b)이 다른 곳(색칠). 문턱은 종이 잡음의 5배."""
    lab, dark, pa, pb = paper_model(sm, paper)
    chroma = np.hypot(lab[..., 1] - pa, lab[..., 2] - pb)
    on = paper & (dark < 10) & (chroma < 10)
    if on.sum() < 100:
        on = paper
    nd = 1.48 * np.median(np.abs(dark[on] - np.median(dark[on])))
    nc = 1.48 * np.median(np.abs(chroma[on] - np.median(chroma[on])))
    color = chroma > max(9.0, 5 * nc)
    fg = ((dark > max(10.0, 5 * nd)) | color).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    fg[~paper] = 0
    return fg, color


def find_drawing(sm):
    """긴 변 512px 이하 그림에서 그림 하나를 고른다. 돌려주는 것: 고른 선 · 색칠(0/255), 색 다른 곳(bool)."""
    hh, ww = sm.shape[:2]
    size = max(hh, ww)
    paper, _ = rt._find_paper(sm)
    fg, color = ink_map(sm, paper)
    # 사진 테두리에 닿는 큰 덩어리는 책상 · 그림자 · 종이 끝이다. 안쪽보다 작으면 버리고, 크면(그림이 사진을 채움) 덜 쳐준다
    n, lab_i, st, _ = cv2.connectedComponentsWithStats(fg)
    touch = (st[:, 0] <= 1) | (st[:, 1] <= 1) | (st[:, 0] + st[:, 2] >= ww - 1) | (st[:, 1] + st[:, 3] >= hh - 1)
    touch[0] = False
    inner = st[~touch, cv2.CC_STAT_AREA][1:].sum() if n > 1 else 0
    tarea = st[touch, cv2.CC_STAT_AREA].sum()
    small = st[:, cv2.CC_STAT_AREA] < max(6, 0.0001 * size * size)
    drop = small | (touch & (tarea < 2 * inner))
    drop[0] = True
    fg[drop[lab_i]] = 0
    edge = (touch & ~drop)[lab_i] & (fg > 0)
    if not fg.any():
        return None, None
    # 그림 하나 고르기: 가까운 것끼리 묶어 가장 큰 묶음(가운데일수록, 사진 테두리에 안 닿을수록 더 쳐줌) + 그 근처
    r = max(5, int(0.03 * size) | 1)
    n, gl, st, cent = cv2.connectedComponentsWithStats(cv2.dilate(fg, _disk(r)))
    cnt = np.bincount(gl[fg > 0], minlength=n).astype(np.float32)
    ecnt = np.bincount(gl[edge], minlength=n)
    dc = np.hypot(cent[:, 0] - ww / 2, cent[:, 1] - hh / 2) / size
    score = cnt * (1 - 0.6 * dc) * np.where(ecnt > 0.5 * cnt, 0.3, 1.0)
    score[0] = -1
    best = int(np.argmax(score))
    x, y, bw, bh = st[best, :4]
    mx, my = 0.1 * bw, 0.1 * bh
    keep = [g for g in range(1, n) if g == best or (x - mx <= cent[g, 0] <= x + bw + mx and y - my <= cent[g, 1] <= y + bh + my and cnt[g] > 0.01 * cnt[best])]
    sel = (np.isin(gl, keep) & (fg > 0)).astype(np.uint8) * 255
    return sel, color


def fill_inside(ink, scale=1.0):
    """선으로 닫힌 안쪽 채우기: 채워질 때까지 틈 메우기(원 모양 닫기)를 키운다 (shape_rig.crop_and_mask 와 같은 순서).
    scale = 그림 크기 / 400 → 틈 크기를 그림 크기에 맞춘다. 지금 방식과 달리 선을 굵게 불리지는 않는다."""
    for k in (5, 9, 13, 17):
        k = max(3, int(round(k * scale)) | 1)
        m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _disk(k))
        filled = _fill_holes(m)
        if (filled > 0).sum() > 1.5 * (m > 0).sum():
            break
    return filled


def _mesh_alpha(filled):
    """shape_rig.crop_and_mask 끝부분과 같다: 가장 큰 덩어리 = 메시, 가까운 조각은 다리를 놓아 붙인다."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + np.argmax(areas)
    alpha = np.zeros_like(filled)
    mesh = np.where(lab == big, 255, 0).astype(np.uint8)
    main_pts = np.argwhere(mesh > 0)
    H, W = filled.shape
    size = max(H, W)
    for k in range(1, n):
        if k == big or areas[k - 1] < 0.01 * areas.max():
            continue
        comp = np.argwhere(lab == k)
        sub_c = comp[:: max(1, len(comp) // 300)]
        sub_m = main_pts[:: max(1, len(main_pts) // 1500)]
        d = ((sub_c[:, None, :] - sub_m[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if math.sqrt(d[i, j]) > 0.2 * size:
            continue
        mesh[lab == k] = 255
        a, b = sub_c[i], sub_m[j]
        cv2.line(mesh, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 255, 7)
        alpha[lab == k] = 255
    alpha[lab == big] = 255
    return mesh, alpha


def _old_box(img):
    """shape_rig.crop_and_mask 가 자른 곳(원본 좌표). 같은 계산을 따라 한다 (eval_real/colored_eval.crop_box 와 같음)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    area = st[:, cv2.CC_STAT_AREA]
    edge = {k for k in range(1, n) if area[k] >= 30 and (st[k, 0] <= 1 or st[k, 1] <= 1 or st[k, 0] + st[k, 2] >= w - 1 or st[k, 1] + st[k, 3] >= h - 1)}
    inner = sum(area[k] for k in range(1, n) if area[k] >= 30 and k not in edge)
    full = inner < sum(area[k] for k in edge)
    for k in range(1, n):
        if area[k] < 30 or (k in edge and (not full or sr.is_straight(lab, k))):
            ink[lab == k] = 0
    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def old_way(img, notes, cleaned=None):
    """지금 방식 (Drawing 과 같음): crop, mesh, alpha, 자른 곳(원본 좌표)."""
    c = rt.clean_photo(img, notes) if cleaned is None else cleaned
    crop, mesh, alpha = sr.crop_and_mask(c, notes)
    return crop, mesh, alpha, _old_box(c)


def _to_grid(m, box, shape):
    """자른 그림의 마스크를 원본 좌표 · shape 크기 격자로 되돌린다."""
    x0, y0, x1, y1 = box
    full = np.zeros(shape, np.uint8)
    full[y0:y1, x0:x1] = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return full > 0


def crop_and_mask_box(img, notes, gate=False, agree=AGREE):
    """돌려주는 것: crop, mesh, alpha, 자른 곳(원본 좌표).
    gate: 색칠한 그림일 때만 새 방식. agree: 지금 방식과 겹침(IoU)이 이보다 작으면 지금 방식 (None 이면 안 봄)."""
    img = rt.clean_photo(img, notes)  # 종이 밖 · 공책 줄 · 떨어진 글자는 사진 정리가 먼저 지운다 (안 부르면 학습용 실패 39.8%)
    h, w = img.shape[:2]
    s = min(1.0, WORK / max(h, w))
    sm = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    sel, color = find_drawing(sm)
    if sel is None:
        return old_way(img, notes, img)
    ys, xs = np.nonzero(sel)
    filled = fill_inside(sel, max(xs.max() - xs.min(), ys.max() - ys.min()) / 400)
    colfrac = float((cv2.morphologyEx((color & (filled > 0)).astype(np.uint8), cv2.MORPH_OPEN, _disk(5)) > 0).sum() / max(1, (filled > 0).sum()))
    if gate and colfrac < GATE:
        return old_way(img, notes, img)
    if agree is not None:
        # 그림 대신 그림자 · 조명 얼룩 덩어리를 고르는 일이 있다 → 지금 방식과 너무 다르면 지금 방식을 믿는다
        old = old_way(img, [], img)
        om = _to_grid((old[1] > 0) | (old[2] > 0), old[3], (h, w))
        om = cv2.resize(om.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        nm = filled > 0
        if (om & nm).sum() < agree * (om | nm).sum():
            notes.append("색 마스크가 지금 방식과 너무 달라 지금 방식을 씀")
            return old
    # 자르기: shape_rig.crop_and_mask 와 같이 그림 둘레 + 15% 를 긴 변 400px 로
    x0, x1, y0, y1 = xs.min() / s, (xs.max() + 1) / s, ys.min() / s, (ys.max() + 1) / s
    pad = 0.15 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1, y1 = int(min(w, math.ceil(x1 + pad))), int(min(h, math.ceil(y1 + pad)))
    crop = img[y0:y1, x0:x1]
    k = sr.MAX_SIDE / max(crop.shape[:2])
    cw, ch = int(crop.shape[1] * k), int(crop.shape[0] * k)
    crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
    full = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
    filled = (cv2.resize(full, (cw, ch), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    mesh, alpha = _mesh_alpha(filled)
    notes.append(f"색 마스크 (칠한 곳 {colfrac:.0%})")
    return crop, mesh, alpha, (x0, y0, x1, y1)


def crop_and_mask(img, notes, gate=False, agree=AGREE):
    crop, mesh, alpha, _ = crop_and_mask_box(img, notes, gate=gate, agree=agree)
    return crop, mesh, alpha
