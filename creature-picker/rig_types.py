"""비휴머노이드 4종 (네 발 · 긴 몸 · 날개 · 몸통만): 점 미리 놓기 + 사용자가 고친 점으로 움직이기.

점은 모두 잘린 그림(crop, 긴 변 400px) 좌표 (x, y).
- 네 발: head, tail(없으면 None), feet [발 끝 2~6개]
- 긴 몸: head, tail
- 날개: head, wings [날개 끝 1~2개]
- 몸통만: 점 없음

마스크 · ARAP · 걸음 곡선은 ../animal-monster 의 shape_rig · quadruped · arap2d 를 그대로 쓴다.
"""
import base64
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

AM = Path(__file__).resolve().parent.parent / "animal-monster"
sys.path.insert(0, str(AM))
import quadruped as q  # noqa: E402
import shape_rig as sr  # noqa: E402
from arap2d import ARAP2D  # noqa: E402

from mesh_cdt import build_mesh  # noqa: E402
from skel import Skel  # noqa: E402

FRAMES = 32
FRAME_MS = 60
TYPES = ("quadruped", "long", "wings", "blob")
WING_FLAP = math.radians(28)  # 위아래 날개짓 각
WING_FOLD = math.radians(65)  # 접었다 펴기: 가장 접힐 때 날개 폭이 cos(65°)≈0.42 배


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def paper_only(img, notes=None):
    """책상 위 종이를 찍은 사진이면 종이 밖(책상 · 손 · 종이 가장자리)을 종이 색으로 지운다.
    지우지 않으면 종이 테두리가 닫힌 선이 되어 종이 전체가 몸 마스크가 된다 (사진 시험: 기울여 찍은 그림 98% 실패).
    종이가 사진을 거의 채우거나, 밝은 곳이 네모나지 않거나, 경계가 흐리면(그림자) 그대로 둔다."""
    h, w = img.shape[:2]
    s = 256 / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB)[..., 0]
    _, bright = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    if n < 2:
        return img
    paper = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
    ff = np.pad(paper, 1)  # 그림 선이 둘러싼 안쪽(얼굴 등)은 종이에 난 구멍이다 → 채운다
    cv2.floodFill(ff, None, (0, 0), 128)
    paper[ff[1:-1, 1:-1] == 0] = 255
    hull = cv2.convexHull(cv2.findNonZero(paper))
    hmask = np.zeros_like(paper)
    cv2.fillConvexPoly(hmask, hull, 255)
    area, harea, tot = (paper > 0).sum(), (hmask > 0).sum(), paper.size
    outside = hmask == 0
    if harea > 0.95 * tot or harea < 0.15 * tot or area < 0.8 * harea or outside.sum() < 0.03 * tot:
        return img
    # 종이 밖은 대부분 종이보다 어두워야 한다. 그림이 사진을 꽉 채워 윤곽선이 흰 종이를 안 · 밖으로 나눈 경우
    # (QuickDraw 고양이 · 돼지 얼굴) 「밖」도 흰 종이라 지우면 윤곽선이 사라진다
    if np.median(L[outside]) > np.median(L[paper > 0]) - 25 or (bright[outside] > 0).mean() > 0.3:
        return img
    # 종이 가장자리는 날카롭고 그림자 경계는 흐리다: 종이 경계(사진 테두리 위는 빼고)의 60% 이상에서 밝기가 확 바뀌어야 종이로 본다
    Lf = L.astype(np.float32)
    grad = np.hypot(cv2.Sobel(Lf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lf, cv2.CV_32F, 0, 1, ksize=3)) / 8
    grad = cv2.dilate(grad, np.ones((5, 5), np.uint8))
    hh, ww = L.shape
    b = hull[:, 0, :].astype(np.float32)
    edge = []
    for i in range(len(b)):
        p0, p1 = b[i], b[(i + 1) % len(b)]
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(p1 - p0))), endpoint=False):
            x, y = p0 + t * (p1 - p0)
            if 3 <= x < ww - 3 and 3 <= y < hh - 3:
                edge.append((int(y), int(x)))
    if len(edge) < 20:
        return img
    ys, xs = np.array(edge).T
    if (grad[ys, xs] > 12).mean() < 0.6:
        return img
    ms = cv2.erode(hmask, np.ones((5, 5), np.uint8)) > 127  # 종이 테두리 · 가장자리 그림자 몇 px 도 뺀다
    # 종이 밖은 가까운 종이 색으로 (경계에 새 선이 생기지 않게). 넓게 흐리는 계산은 작은 그림에서 한다 (큰 사진에서 몇 초 걸렸다)
    fs = small.astype(np.float32)
    sig = 0.05 * max(hh, ww)
    wgt = cv2.GaussianBlur(ms.astype(np.float32), (0, 0), sig)
    est = cv2.GaussianBlur(fs * ms[..., None], (0, 0), sig) / np.maximum(wgt, 1e-4)[..., None]
    est = np.where((wgt > 1e-3)[..., None], est, np.median(fs[ms], axis=0))
    est = cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(ms.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR) > 127
    if notes is not None:
        notes.append("종이 밖(책상 · 손)을 지움")
    return np.clip(np.where(m[..., None], img.astype(np.float32), est), 0, 255).astype(np.uint8)


def _disk(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_holes(m):
    ff = np.pad(m, 1)
    cv2.floodFill(ff, None, (0, 0), 128)
    m[ff[1:-1, 1:-1] == 0] = 255
    return m


def _find_paper(sm):
    """작은 그림(긴 변 128px)에서 종이 영역을 찾는다. 못 믿겠으면 사진 전체를 종이로 본다."""
    hh, ww = sm.shape[:2]
    size = max(hh, ww)
    s2 = 128 / size
    big = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, _disk(max(5, int(0.08 * size) | 1)))  # 그림 선을 지운 바탕
    tiny = cv2.GaussianBlur(cv2.resize(big, (max(1, round(ww * s2)), max(1, round(hh * s2))), interpolation=cv2.INTER_AREA), (3, 3), 0)
    th, tw = tiny.shape[:2]
    # 그림자는 밝기만 바뀌고 색은 그대로 → 밝기 차이를 1/3 로 줄여서, 가운데 밝은 곳 여러 점부터 색이 천천히 바뀌는 데까지 흘려 채운다
    tlab = cv2.cvtColor(tiny, cv2.COLOR_BGR2LAB).astype(np.float32)
    tl = tlab[..., 0]
    tq = np.dstack([tl / 3, tlab[..., 1], tlab[..., 2]]).astype(np.uint8)
    cy0, cy1, cx0, cx1 = th // 4, th - th // 4, tw // 4, tw - tw // 4
    cen = cv2.blur(tl, (5, 5))[cy0:cy1, cx0:cx1]
    fm = np.zeros((th + 2, tw + 2), np.uint8)
    cand = np.argwhere(cen >= cen.max() - 8)
    for yy, xx in cand[:: max(1, len(cand) // 8)]:
        if fm[yy + cy0 + 1, xx + cx0 + 1]:
            continue
        cv2.floodFill(tq, fm, (int(xx + cx0), int(yy + cy0)), 0, (3, 3, 3), (3, 3, 3), 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    paper = _fill_holes(cv2.morphologyEx(fm[1:-1, 1:-1], cv2.MORPH_CLOSE, _disk(5)))
    frac = (paper > 0).mean()
    ok = 0.05 <= frac
    if ok and frac < 0.97:  # 종이로 믿으려면: 바깥보다 밝거나(책상) 바깥이 더 얼룩덜룩하고(무늬 천), 모양이 네모에 가까워야
        pin, pout = paper > 0, paper == 0
        raw = cv2.cvtColor(cv2.resize(sm, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        tex = np.abs(raw - cv2.blur(raw, (5, 5)))
        ok = np.median(tl[pin]) - np.median(tl[pout]) > 15 or np.mean(tex[pout]) > 1.5 * np.mean(tex[pin])
        ok = ok and pin.sum() >= 0.75 * cv2.contourArea(cv2.convexHull(cv2.findNonZero(paper)))
    if not ok:
        paper[:] = 255
        frac = 1.0
    else:
        paper = cv2.erode(paper, _disk(3))  # 종이 가장자리 선 · 그림자 띠도 뺀다
    return cv2.resize(paper, (ww, hh), interpolation=cv2.INTER_LINEAR) > 127, frac


def _rule_lines(ink, size, dark):
    """공책 줄 · 모눈 · 종이 끝까지 가는 곧은 선 (1 = 줄). 둘째 값: 같은 방향 줄이 여럿(공책)인가."""
    rules = np.zeros_like(ink)
    ink_dark = np.median(dark[ink > 0]) if ink.any() else 0
    ruled = False
    lines = cv2.HoughLinesP(ink * 255, 1, np.pi / 180, int(0.15 * size), minLineLength=int(0.3 * size), maxLineGap=int(0.015 * size))
    if lines is not None:
        segs = lines.reshape(-1, 4).astype(np.float32)
        ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
        ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        th_ = np.radians(ang)
        mx_, my_ = (segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2
        for i in range(len(segs)):  # 같은 방향 선이 서로 다른 자리에 4개 넘게 있거나, 사진 반쯤 되는 긴 선
            same = np.abs((ang - ang[i] + 90) % 180 - 90) < 3
            off = np.sort(mx_[same] * np.sin(th_[i]) - my_[same] * np.cos(th_[i]))  # 선에 수직인 방향의 자리
            par = 1 + (np.diff(off) > 6).sum() >= 4  # 굵은 선 하나가 허프 선 여러 개로 나오는 것은 하나로 센다
            ruled |= bool(par)
            x0, y0, x1, y1 = segs[i].astype(int)
            if not par and ln[i] > 0.45 * size:  # 긴 선 하나: 그림 선보다 연할 때만 (종이 끝 · 여백 줄). 그림의 긴 선(말 목 · 별)은 남긴다
                t = np.linspace(0, 1, int(ln[i]))
                xs = np.clip(np.round(x0 + t * (x1 - x0)).astype(int), 0, ink.shape[1] - 1)
                ys = np.clip(np.round(y0 + t * (y1 - y0)).astype(int), 0, ink.shape[0] - 1)
                on = ink[ys, xs] > 0
                par = on.any() and np.median(dark[ys, xs][on]) < 0.6 * ink_dark
            if par:
                cv2.line(rules, (x0, y0), (x1, y1), 1, 3)
    # 끊긴 줄(허프가 놓침): 가로(세로)로 긴 가는 선이 고른 간격으로 5개 넘게
    lk = max(9, int(0.06 * size))
    for ax, ker in ((1, np.ones((1, lk), np.uint8)), (0, np.ones((lk, 1), np.uint8))):
        n, lab, st, cen = cv2.connectedComponentsWithStats(cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker))
        ext = st[:, 2] if ax == 1 else st[:, 3]
        longs = [i for i in range(1, n) if ext[i] >= 0.2 * size and st[i, 4] / max(ext[i], 1) < 3.5]
        if len(longs) < 5:
            continue
        gaps = np.diff(np.sort(cen[longs, 1 - ax]))
        gaps = gaps[gaps > 3]
        if len(gaps) >= 4 and np.std(gaps) < 0.35 * np.median(gaps):
            rules[np.isin(lab, longs)] = 1
            ruled = True
    return rules, ruled


def clean_photo(img, notes=None):
    """종이 그림 사진을 스캔한 그림처럼 정리한다 (실제 아이 그림 사진 Meta 에서 마스크 실패 45%).
    종이 밖 · 공책 줄 · 떨어진 다른 그림 · 글자를 종이 색으로 지우고, 연한 선은 진하게 한다.
    깨끗한 그림(흰 종이를 꽉 채운 진한 선)은 그대로 돌려준다. 크기 · 좌표는 바꾸지 않는다.
    판단은 긴 변 512px 로 줄인 그림에서 하고, 결과는 원래 크기로 만든다."""
    img = paper_only(img, notes)
    h, w = img.shape[:2]
    s = min(1.0, 512 / max(h, w))
    sm = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    hh, ww = sm.shape[:2]
    size = max(hh, ww)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    # 1) 선: 바탕(선 굵기보다 큰 닫기로 선을 지운 밝기)보다 어두운 곳 + 둘레와 색이 다른 곳. 그림자 · 조명은 바탕이 따라간다
    k = max(5, int(0.025 * size) | 1)
    bgC = cv2.blur(cv2.morphologyEx(L, cv2.MORPH_CLOSE, _disk(k)), (k, k))
    # 선을 찾는 바탕은 중앙값(지우개 자국 · 그림자 경계처럼 넓고 연한 얼룩은 바탕이 된다). 단 닫기 바탕보다 25 넘게 어둡지는 않게
    bgL = np.maximum(cv2.medianBlur(L.astype(np.uint8), k).astype(np.float32), bgC - 25)
    dark = np.maximum(bgL - L, 0)
    bga = cv2.medianBlur(lab[..., 1].astype(np.uint8), k)
    bgb = cv2.medianBlur(lab[..., 2].astype(np.uint8), k)
    chroma = np.hypot(lab[..., 1] - bga, lab[..., 2] - bgb)
    noise = 1.48 * np.median(np.abs(dark - np.median(dark)))
    ink_d = dark > max(7.0, 4 * noise)
    ink0 = (ink_d | (chroma > 18)).astype(np.uint8)
    # 2) 종이
    paper, pfrac = _find_paper(sm)
    # 3) 넓게 칠한 곳: 바탕 밝기에 묻혀 선으로 안 잡힌다(가운데가 빠져 그림이 조각남) → 종이 색과 색이 다르거나 아주 어두운 곳을 더한다
    ink = ink0.copy()
    fill_in = np.zeros_like(ink)
    plab = None
    pm0 = paper & (cv2.dilate(ink0, _disk(5)) == 0)
    if pm0.sum() > 50:
        pm0 &= np.linalg.norm(lab - np.median(lab[pm0], axis=0), axis=-1) < 20
    if pm0.sum() > 50:
        sg = 0.06 * size
        wg = cv2.GaussianBlur(pm0.astype(np.float32), (0, 0), sg)
        # 종이가 먼 곳은 종이 색 중앙값으로 부드럽게 넘어간다 (뚝 끊기면 그 경계가 새 선이 된다)
        plab = (cv2.GaussianBlur(lab * pm0[..., None], (0, 0), sg) + 1e-2 * np.median(lab[pm0], axis=0)) / (wg + 1e-2)[..., None]
        fill = (np.hypot(lab[..., 1] - plab[..., 1], lab[..., 2] - plab[..., 2]) > 15) | (plab[..., 0] - L > 60)
        fill = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # 사진 테두리에 닿는 덩어리는 칠한 곳이 아니라 책상 · 그림자 · 배경이다
        nf, fl, fst, _ = cv2.connectedComponentsWithStats(fill)
        edge = (fst[:, 0] <= 1) | (fst[:, 1] <= 1) | (fst[:, 0] + fst[:, 2] >= ww - 1) | (fst[:, 1] + fst[:, 3] >= hh - 1)
        edge[0] = True
        fill_in = ((~edge[fl]) & (ink0 == 0)).astype(np.uint8)
        ink |= fill_in
    ink &= paper
    # 4) 공책 줄 (칠한 곳은 안 자른다. 줄보다 훨씬 진한 곳은 줄 위를 지나는 그림 선이라 남긴다)
    rules, ruled = _rule_lines(ink0 & paper, size, dark)
    rules0 = rules.copy()
    rules[fill_in > 0] = 0
    if (rules & ink0).any():
        rd = np.median(dark[(rules & ink0) > 0])
        rules[dark > max(1.6 * rd, rd + 15)] = 0
    has_rules = rules.any()
    ink[rules > 0] = 0
    n, lab_i, st, _ = cv2.connectedComponentsWithStats(ink)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < max(4, 0.00005 * size * size)
    ink[small[lab_i]] = 0
    if not ink.any():
        return img
    # 깨끗한 그림이면 손대지 않는다: 종이가 사진을 채움 · 맨 종이 밝기가 고름 · 선이 진함 · 공책 줄 없음
    # (줄로 잡힌 것이 그림만큼 진하면 빗금 · 털 같은 그림 선이라 공책으로 치지 않는다)
    bare = paper & (cv2.dilate(ink, _disk(5)) == 0)
    if ruled and (rules0 & ink0).any() and (ink0 & (rules0 == 0)).any():
        ruled = np.median(dark[(rules0 & ink0) > 0]) < 0.6 * np.median(dark[(ink0 & (rules0 == 0)) > 0])
    if pfrac > 0.97 and bare.sum() > 0.2 * bare.size and not ruled:
        flat = np.percentile(L[bare], 95) - np.percentile(L[bare], 5)
        # 맨 종이가 완전히 고르면 사진이 아니라 그린 파일이다. 아니면 종이가 고르고 선이 진해야 깨끗한 그림
        if flat <= 3 or (flat < 15 and np.median((bgC - L)[(bgC - L > 25) & paper]) > 50):
            return img
    # 5) 그림 하나 고르기: 가까운 선끼리 묶어 가장 큰 묶음(가운데일수록, 사진 테두리에 안 닿을수록 더 쳐줌) + 그 둘레 안의 묶음
    r = max(3, int(0.03 * size) | 1)
    n, gl, st, cent = cv2.connectedComponentsWithStats(cv2.dilate(ink, _disk(r)))
    cnt = np.bincount(gl[ink > 0], minlength=n).astype(np.float32)
    dc = np.hypot(cent[:, 0] - ww / 2, cent[:, 1] - hh / 2) / size
    touch = (st[:, 0] <= r) | (st[:, 1] <= r) | (st[:, 0] + st[:, 2] >= ww - r) | (st[:, 1] + st[:, 3] >= hh - r)
    score = cnt * (1 - 0.6 * dc) * np.where(touch, 0.3, 1.0)
    score[0] = -1
    best = int(np.argmax(score))
    x, y, bw, bh = st[best, :4]
    mx, my = 0.15 * bw, 0.15 * bh
    keep_g = [g for g in range(1, n) if x - mx <= cent[g, 0] <= x + bw + mx and y - my <= cent[g, 1] <= y + bh + my]
    sel = np.isin(gl, keep_g) & (ink > 0)
    region = _fill_holes(cv2.morphologyEx(sel.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _disk(r)))
    region = cv2.dilate(region, _disk(5))
    # 6) 종이 색 (선 · 칠한 곳을 뺀 종이를 넓게 흐려서)
    if plab is not None:
        pm = (paper & (cv2.dilate(ink, _disk(5)) == 0) & (np.linalg.norm(lab - plab, axis=-1) < 15)).astype(np.float32)
    else:
        pm = (paper & (cv2.dilate(ink, _disk(5)) == 0) & (chroma < 12)).astype(np.float32)
    if pm.sum() < 50:
        pm = np.ones_like(pm)
    fs = sm.astype(np.float32)
    sig = 0.04 * size
    wgt = cv2.GaussianBlur(pm, (0, 0), sig)
    pcol = (cv2.GaussianBlur(fs * pm[..., None], (0, 0), sig) + 1e-2 * np.median(fs[pm > 0], axis=0)) / (wgt + 1e-2)[..., None]
    # 7) 고른 그림만 남기고 나머지는 종이 색. 어두운 선은 종이보다 적어도 80 어둡게, 나머지(칠한 곳)는 선 어두움 중앙값이 60 쯤 되게
    gain = float(np.clip(60 / max(1.0, np.median(dark[sel])), 1, 4))
    line = sel & ink_d
    if line.any():  # 그림 선보다 훨씬 연한 것(지우개 자국 · 비친 뒷면)은 진하게 하지 않는다
        line &= dark >= 0.35 * np.median(dark[line])
    # 가는 선(1~2px)은 shape_rig 의 3×3 열기에서 사라진다 → 한 겹 굵게
    thin = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).sum() < 0.5 * line.sum()
    if thin:
        line = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pcol_f = cv2.resize(pcol, (w, h), interpolation=cv2.INTER_LINEAR)
    keep = cv2.resize(region, (w, h), interpolation=cv2.INTER_LINEAR) > 127
    line_f = cv2.resize(line.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR) > 127
    if has_rules:
        keep &= cv2.resize(rules * 255, (w, h), interpolation=cv2.INTER_LINEAR) <= 127
    f = img.astype(np.float32)
    if thin:
        kk = max(3, int(round(1 / s)) | 1)
        f = np.where(line_f[..., None], cv2.erode(img, np.ones((kk, kk), np.uint8)).astype(np.float32), f)
    d = pcol_f - f
    fac = np.where(line_f, np.clip(80 / np.maximum(d.mean(-1), 1), gain, 6), gain)
    out = np.clip(np.where(keep[..., None], pcol_f - fac[..., None] * d, pcol_f), 0, 255).astype(np.uint8)
    if notes is not None:
        notes.append(f"사진 정리 (줄 {'지움' if has_rules else '없음'}, 선 진하게 ×{gain:.1f})")
    return out


class Drawing:
    def __init__(self, img):
        self.notes = []
        img = clean_photo(img, self.notes)
        self.crop, self.mask, self.alpha = sr.crop_and_mask(img, self.notes)
        g = cv2.cvtColor(self.crop, cv2.COLOR_BGR2GRAY)
        self.ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
        # 네 발은 다리 사이 틈을 되돌린 마스크를 쓴다
        self.mask_q, self.alpha_q = q.open_gaps(self.crop, self.mask, self.alpha, self.ink)
        self.skel = Skel(self.mask)

    def masks(self, kind):
        return (self.mask_q, self.alpha_q) if kind == "quadruped" else (self.mask, self.alpha)


# ---------- 도우미 ----------
def snap(mask, p):
    """점이 그림 밖이면 가장 가까운 그림 안 픽셀로."""
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = mask.shape
    if 0 <= x < w and 0 <= y < h and mask[y, x]:
        return np.array([x, y], np.float32)
    pts = np.argwhere(mask > 0)
    k = np.argmin((pts[:, 0] - y) ** 2 + (pts[:, 1] - x) ** 2)
    return np.array([pts[k, 1], pts[k, 0]], np.float32)


def ridge(mask, a, b):
    """a → b 몸 한가운데 능선을 따르는 길 (x, y). 최단 경로는 가장자리에 붙는다."""
    a, b = snap(mask, a), snap(mask, b)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    mcp = MCP_Geometric(np.where(mask > 0, 1.0 / (dt + 1.0), np.inf))
    ay, ax, by, bx = int(a[1]), int(a[0]), int(b[1]), int(b[0])
    costs, _ = mcp.find_costs([(ay, ax)], [(by, bx)])
    if not np.isfinite(costs[by, bx]):
        # 떨어져 그린 조각(머리 · 날개)이라 몸 안으로 이어지지 않으면 곧은 선으로
        n = max(2, int(np.linalg.norm(b - a)))
        return np.linspace(a, b, n).astype(np.float32)
    return np.array(mcp.traceback((by, bx)), np.float32)[:, ::-1]


def at(path, f):
    return path[min(len(path) - 1, int(round(f * (len(path) - 1))))]


def r1(p):
    return None if p is None else [round(float(p[0]), 1), round(float(p[1]), 1)]


def nearest_px(mask, x, y):
    return snap(mask, (x, y))


def protrusion(mask, dt, centre, tip):
    """몸 가운데 → 끝 길에서, 끝 쪽 굵기의 2배를 넘는 곳 = 뿌리. 반환: 길, 뿌리 번호, 튀어나온 길이, 굵기.
    뼈대 가지가 길어도 몸통 안을 지나는 부분은 튀어나온 길이에 안 들어간다 (울퉁불퉁한 덩어리 괴물)."""
    path = ridge(mask, centre, tip)
    w = np.array([dt[int(y), int(x)] for x, y in path])
    tip_w = float(np.median(w[-max(3, len(w) // 6):]))
    root_i = 0
    for i in range(len(w) - 1, -1, -1):
        if w[i] > max(2.0 * tip_w, 0.2 * dt.max()):
            root_i = i
            break
    seg = np.linalg.norm(np.diff(path[root_i:], axis=0), axis=1).sum() if root_i < len(path) - 1 else 0.0
    return path, root_i, float(seg), 2 * tip_w


# ---------- 점 미리 놓기 ----------
def propose(d):
    mask, H, W = d.mask, *d.mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx = (x0 + x1) / 2
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    size = max(H, W)
    cy0, cx0 = np.unravel_index(np.argmax(dt), dt.shape)
    centre = np.array([cx0, cy0], np.float32)
    # 가지친 뼈대 끝마다 실제로 몸 밖으로 튀어나온 길이를 잰다 (shape_rig 팔다리 기준: 길이 ≥ 긴 변 12% · 굵기 2배)
    tips = []
    for t in d.skel.tips():
        _, _, plen, width = protrusion(mask, dt, centre, t["tip"])
        tips.append({**t, "len": plen, "width": width})
    tips.sort(key=lambda t: -t["len"])
    long_tips = [t for t in tips if t["len"] >= 0.12 * size and t["len"] >= 2.0 * t["width"]]
    pts, why = {}, {}

    # 네 발: 이미 있는 다리 찾기(quadruped.build_rig)를 먼저, 안 되면 바닥에 고르게
    quad_ok, n_feet, y_belly = False, 0, None
    try:
        pins0, rig = q.build_rig(d.mask_q, d.ink)
        feet = [pins0[l["i"] + 2] for l in rig["legs"]]
        # quadruped 는 머리 방향은 잘 고르지만 끝점이 목에서 멈출 때가 있다 (머리를 동그라미 선으로만 그린 개).
        # 미리 놓는 점은 머리 쪽에서 몸 안 거리로 가장 먼 곳으로
        c = pins0[0]
        mcp = MCP_Geometric(np.where(d.mask > 0, 1.0, np.inf))  # open_gaps 는 떨어져 그린 머리를 잇는 다리까지 끊을 수 있다
        D, _ = mcp.find_costs([(int(c[1]), int(c[0]))])
        xx = np.arange(W)[None, :]
        legpix = np.zeros_like(d.mask, bool)
        legpix[rig["y_belly"]:] = True
        cand = np.where(((xx - c[0]) * rig["forward"] > 0) & np.isfinite(D) & ~legpix, D, -1)
        hy, hx = np.unravel_index(np.argmax(cand), cand.shape)
        head_tip = (hx, hy) if cand.max() > 0 else pins0[rig["head"] + 2]
        pts["quadruped"] = {"head": r1(head_tip), "tail": r1(pins0[rig["tail"] + 2]) if rig["tail"] is not None else None,
                            "feet": [r1(f) for f in feet]}
        n_feet, y_belly = len(feet), rig["y_belly"]
        quad_ok = n_feet >= 3
        why["quadruped"] = f"다리 {len(feet)}개 찾음"
    except Exception as e:
        bw = x1 - x0
        feet = [nearest_px(mask, x0 + f * bw, y1) for f in (0.15, 0.38, 0.62, 0.85)]
        pts["quadruped"] = {"head": r1(nearest_px(mask, x0, y0 + 0.3 * (y1 - y0))), "tail": r1(nearest_px(mask, x1, y0 + 0.4 * (y1 - y0))),
                            "feet": [r1(f) for f in feet]}
        why["quadruped"] = f"자동으로 못 찾음 ({e}) → 바닥에 고르게 놓음"
        if looks_face_only(mask):
            pts["quadruped"]["no_body"] = True  # animate 가 네 발 대신 몸통만으로 움직인다 (사람이 점을 옮기면 화면이 지운다)
            why["quadruped"] = f"다리를 못 찾았고 앞모습 얼굴처럼 보임 ({e}) → 그대로 두면 몸통만으로 움직임. 네 발로 움직이려면 발 점을 옮겨 주세요"

    # 긴 몸: shape_rig 의 긴 몸 판정이 되면 그 머리 · 꼬리, 아니면 가지친 뼈대의 가장 먼 두 끝 (굵은 쪽이 머리)
    axis, _ = sr.find_body_axis(mask, d.ink)
    if axis is not None:
        pts["long"] = {"head": r1(axis[-1][::-1]), "tail": r1(axis[0][::-1])}
        why["long"] = "긴 몸으로 보임"
    else:
        dia = d.skel.diameter()
        if dia is not None and len(dia) > 4:
            a, b = dia[0], dia[-1]
            wa, wb = dt[int(a[1]), int(a[0])], dt[int(b[1]), int(b[0])]
            head, tail = (a, b) if wa >= wb else (b, a)
        else:
            head, tail = nearest_px(mask, x0, (y0 + y1) / 2), nearest_px(mask, x1, (y0 + y1) / 2)
        pts["long"] = {"head": r1(head), "tail": r1(tail)}
        why["long"] = "가장 먼 두 끝에 놓음"

    # 날개: 몸 가운데를 사이에 두고 양쪽에서 가장 긴, 아래로 뻗지 않은 튀어나온 곳 (아래로 뻗은 것은 다리). 없으면 좌우 끝
    def sideways(t):
        dx, dy = t["tip"][0] - t["base"][0], t["tip"][1] - t["base"][1]
        return dy < 1.2 * abs(dx)

    side_tips = [t for t in long_tips if sideways(t)]
    left = [t for t in side_tips if t["tip"][0] < cx0]
    right = [t for t in side_tips if t["tip"][0] >= cx0]
    # 날개: 날개마다 몸에 붙은 곳의 위 끝 · 아래 끝 두 점 (= 경첩 선). 선 바깥이 통째로 날개다.
    # 먼저 맨 왼쪽 · 맨 오른쪽 끝(날개 끝)과 두 끝 가운데 세로줄 맨 위(머리)로 몸 축을 잡는다.
    # 나비 날개처럼 넓은 날개는 「가는 갈래」가 아니라서 튀어나온 곳으로 고르면 더듬이가 잡힌다.
    tips_lr = []
    for xe in (x0, x1):
        col = np.nonzero(mask[:, xe])[0]
        tips_lr.append(np.array([xe, col.mean()], np.float32))
    mx = int(round((x0 + x1) / 2))
    band = mask[:, max(0, mx - 2): mx + 3].any(1)
    head = nearest_px(mask, mx, np.nonzero(band)[0][0] + 4) if band.any() else nearest_px(mask, cx, y0)
    wp = ridge(mask, tips_lr[0], tips_lr[1])
    wc = wp[np.argmin(((wp - head) ** 2).sum(1))]  # 몸통 = 두 날개 끝을 잇는 길에서 머리와 가장 가까운 곳
    u = head - wc
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 3 else np.array([0, -1], np.float32)
    n = np.array([-u[1], u[0]], np.float32)
    # 몸통 반 폭: 굵기(dt)로 재면 날개와 몸통이 한 덩어리인 나비는 너무 크게 나온다 → 날개 끝 거리의 18% 를 넘지 않게
    reach = float(np.mean([abs((t - wc) @ n) for t in tips_lr]))
    bw = max(4.0, min(float(dt[int(wc[1]), int(wc[0])]), 0.18 * reach))
    wings, ratios = [], []
    for tip in tips_lr:
        s = 1.0 if (tip - wc) @ n >= 0 else -1.0
        base = wc + n * s * (bw + 2)
        ends = []
        for dirn in (1, -1):  # 몸 축을 따라 머리 쪽 · 꼬리 쪽으로 걸어가며 그림이 끝나는 곳
            last, miss = base, 0
            for k in range(1, size):
                p = base + u * dirn * k
                x, y = int(round(p[0])), int(round(p[1]))
                if not (0 <= x < W and 0 <= y < H):
                    break
                if mask[y, x]:
                    last, miss = p, 0
                else:
                    miss += 1
                    if miss > 3:
                        break
            ends.append(last)
        wings.append([r1(ends[0]), r1(ends[1]), r1(tip)])  # [뿌리 위(머리 쪽), 뿌리 아래, 날개 끝]
        # 날개짓 방식: 날개가 길이에 비해 넓으면(나비) 접었다 펴기, 가늘면(앞모습 새) 위아래
        p = ridge(mask, wc, tip)
        p = p[len(p) // 4:]
        if len(p) > 4:
            seg = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            ratios.append(2 * float(np.mean([dt[int(y), int(x)] for x, y in p])) / max(seg, 1.0))
    style = "fold" if ratios and float(np.mean(ratios)) > 0.45 else "updown"
    pts["wings"] = {"wings": wings, "style": style}
    why["wings"] = f"몸 축 양옆 경첩 선, 날개 폭/길이 {np.mean(ratios) if ratios else 0:.2f} → {'접었다 폈다' if style == 'fold' else '위아래'}"
    pts["blob"] = {}

    # AI 추측 (카드에 「AI 추측」 표시만 한다. 고르는 건 사람)
    feat = {"quad_ok": quad_ok, "n_feet": n_feet, "y_belly": y_belly, "feet": pts["quadruped"]["feet"], "axis": axis is not None,
            "tips": tips, "long_tips": long_tips, "left": left, "right": right, "centre": centre,
            "bbox": (x0, y0, x1, y1), "size": size}
    d._guess_feat = feat
    return {"guess": guess_kind(d, feat), "points": pts, "why": why}


FACE_SYM_LR = 0.4  # 좌우 대칭이 이 이상이고
FACE_ASPECT = 1.3  # 가로 ÷ 세로가 이 이하면 앞모습 얼굴


def looks_face_only(mask):
    """몸 없이 얼굴만 그린 그림(앞모습 고양이 · 돼지 얼굴)인가. 다리 찾기가 실패한 그림에만 쓴다.
    네 발 그림은 옆모습이라 좌우가 다르고 가로로 길다. 얼굴만 그린 그림은 앞모습이라 좌우가 비슷하고 동그랗다.
    QuickDraw 학습용 480장(eval_real/QUAD_STRETCH.md)에서 진짜 네 발 250장 중 1장만 걸리고, 얼굴만 184장 중 93장이 걸린 기준.
    억지로 다리를 만들면 얼굴 안이 소용돌이치며 늘어났다 (REPORT.md 실패 1번의 대부분)."""
    ys, xs = np.nonzero(mask)
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
    sym_lr = (sub & sub[:, ::-1]).sum() / max(1, (sub | sub[:, ::-1]).sum())
    aspect = sub.shape[1] / sub.shape[0]
    return bool(sym_lr >= FACE_SYM_LR and aspect <= FACE_ASPECT)


GUESS_MODEL = Path(__file__).resolve().parent / "guess_model.json"
GUESS_MIN_P = 0.8  # 이보다 자신 없으면 「모르겠음」: 틀리게 확신하는 것이 모르겠음보다 나쁘다
_guess_model = None


def guess_features(d, f):
    """종류 추측에 쓰는 숫자 24개 (그림 크기와 상관없게 긴 변 · 몸 넓이로 나눈 값). eval_real/train_guess.py 가 이것으로 배운다."""
    m = d.mask
    x0, y0, x1, y1 = f["bbox"]
    size = f["size"]
    sub = m[y0:y1 + 1, x0:x1 + 1] > 0
    area = sub.sum()
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    e = {"fill": area / sub.size,  # 테두리 상자 중 그림이 차지한 비율
         "solidity": area / max(1, cv2.contourArea(cv2.convexHull(c))),  # 볼록 껍질 중 그림 비율 (구름 ↑, 별 · 다리 ↓)
         "aspect": (x1 - x0 + 1) / (y1 - y0 + 1),
         "sym_lr": (sub & sub[:, ::-1]).sum() / max(1, (sub | sub[:, ::-1]).sum()),  # 좌우 대칭 (나비 · 얼굴 ↑)
         "sym_ud": (sub & sub[::-1]).sum() / max(1, (sub | sub[::-1]).sum()),
         "n_long": len(f["long_tips"]), "n_tips": len(f["tips"]), "n_regions": int(region_labels(d)[0].max()),
         "n_comp": len(cnts), "inkd": ((d.ink > 0) & (m > 0)).sum() / max(1, area)}  # 몸 안 선 밀도 (수염 · 줄무늬 ↑)
    cx, cy = f["centre"]
    lt = f["long_tips"]
    ang = sorted(float(np.arctan2(t["tip"][1] - cy, t["tip"][0] - cx)) for t in lt)
    # 튀어나온 곳 사이 가장 큰 빈 각: 별처럼 사방으로 고르게 뻗으면 작다
    e["maxgap"] = float(np.diff(ang + [ang[0] + 2 * np.pi]).max()) if len(ang) >= 2 else 2 * np.pi
    L = [t["len"] / size for t in lt]
    e["l0"] = L[0] if L else 0
    e["l1"] = L[1] if len(L) > 1 else 0
    e["lcv"] = float(np.std(L) / np.mean(L)) if len(L) >= 2 else 1  # 갈래 길이가 고른지 (별 ↓)
    up = side = down = 0
    for t in lt:
        a = math.degrees(math.atan2(-(t["tip"][1] - t["base"][1]), abs(t["tip"][0] - t["base"][0])))
        up, side, down = up + (a > 45), side + (-45 <= a <= 45), down + (a < -45)
    e.update(up=up, side=side, down=down, n_feet=f["n_feet"], axis=int(f["axis"]))
    if f["n_feet"]:
        fx = np.array([p[0] for p in f["feet"]])
        fy = np.array([p[1] for p in f["feet"]])
        e["feet_spread"] = (fx.max() - fx.min()) / (x1 - x0 + 1)
        e["feet_low"] = float(np.mean(fy > y0 + 0.75 * (y1 - y0)))
        e["belly"] = (f["y_belly"] - y0) / (y1 - y0 + 1)
    else:
        e["feet_spread"] = e["feet_low"] = 0
        e["belly"] = 1
    try:  # 볼록 껍질에서 깊이 패인 곳 수 (별 꼭짓점 사이 골)
        dfc = cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False))
        e["n_defect"] = int((dfc.reshape(-1, 4)[:, 3] / 256 > 0.08 * size).sum()) if dfc is not None else 0
    except cv2.error:
        e["n_defect"] = 0
    e["dtmax"] = float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max() / size)  # 가장 굵은 곳
    return {k: float(v) for k, v in e.items()}


def guess_proba(x):
    """작은 결정 나무 묶음(gradient boosting) → 종류마다 확률. 나무는 guess_model.json (eval_real/train_guess.py 가 QuickDraw 학습용 그림으로 만듦)."""
    global _guess_model
    if _guess_model is None:
        import json
        _guess_model = json.loads(GUESS_MODEL.read_text(encoding="utf-8"))
    M = _guess_model
    v = np.array([x[k] for k in M["features"]], np.float32).astype(np.float64)  # scikit-learn 처럼 float32 로 비교
    raw = np.array(M["init"], np.float64)
    for stage in M["trees"]:
        for k, (feat, thr, left, right, value) in enumerate(stage):
            i = 0
            while left[i] >= 0:
                i = left[i] if v[feat[i]] <= thr[i] else right[i]
            raw[k] += M["lr"] * value[i]
    p = np.exp(raw - raw.max())
    return dict(zip(M["classes"], p / p.sum()))


def guess_kind(d, f):
    if not GUESS_MODEL.exists():
        return guess_rules(f)
    d._guess_x = guess_features(d, f)
    p = guess_proba(d._guess_x)
    k = max(p, key=p.get)
    return k if p[k] >= GUESS_MIN_P else None


def guess_rules(f):
    """예전 추측 (2026-10 까지). 비교용으로 남긴다."""
    if f["quad_ok"]:
        guess = "quadruped"
    elif f["axis"]:
        guess = "long"
    elif not f["long_tips"]:
        guess = "blob"
    elif (f["left"] and f["right"] and f["long_tips"][0] in (f["left"][0], f["right"][0])
          and abs(f["left"][0]["len"] - f["right"][0]["len"]) < 0.35 * max(f["left"][0]["len"], f["right"][0]["len"])):
        guess = "wings"  # 가장 긴 두 갈래가 양옆으로 비슷하게 뻗었을 때만
    else:
        guess = None
    return guess


# ---------- 사용자가 고친 점 → 기준점 · 동작 ----------
def snap_to_tips(d, pts, keys_single=("head", "tail"), key_multi="feet"):
    """점을 그림의 「끝」(가지친 뼈대의 끝점: 발끝 · 꼬리 끝 · 귀 끝 · 코끝)에 맞춘다. 반경 안에 끝점이 없으면 그대로.
    발은 서로 다른 끝점에 하나씩: 막대 다리가 붙어 있는 개 그림에서 발 점이 10px 만 빗나가도 옆 다리로 붙어
    두 발이 한 다리에 몰리거나 다리 뿌리를 엉뚱하게 찾았다 (check_points_jitter 개 3/8 → 맞춘 뒤 다시 잼)."""
    tips = [np.asarray(t["tip"], np.float32) for t in d.skel.tips()]
    if not tips:
        return pts
    R = max(15.0, 0.06 * max(d.mask.shape))
    out = dict(pts)
    used = set()
    feet = [np.asarray(f, np.float32) for f in (pts.get(key_multi) or [])]
    if feet:
        # 발은 한꺼번에 최적으로 짝짓는다: 그림 아래쪽 40% 의 끝점(발 후보)과 거리 합이 가장 작게, 서로 다른 끝에 하나씩.
        # 가까운 순서로 하나씩 붙였더니 많이 빗나간 발이 반경 밖에 남는 사이 옆 발이 그 끝을 가져가 두 발이 한 다리에 몰렸다
        from scipy.optimize import linear_sum_assignment
        ys = np.nonzero(d.mask.any(1))[0]
        low = ys[0] + 0.6 * (ys[-1] - ys[0])
        cand = [j for j, t in enumerate(tips) if t[1] >= low]
        snapped = list(feet)
        if len(cand) >= len(feet):
            cost = np.array([[np.linalg.norm(f - tips[j]) for j in cand] for f in feet])
            rows, cols = linear_sum_assignment(cost)
            for i, c in zip(rows, cols):
                if cost[i, c] <= 3 * R:  # 너무 먼 끝은 다른 다리가 아니라 엉뚱한 곳
                    snapped[i] = tips[cand[c]]
                    used.add(cand[c])
        out[key_multi] = [p.tolist() for p in snapped]
    for key in keys_single:
        p = pts.get(key)
        if p is None:
            continue
        p = np.asarray(p, np.float32)
        cand = [(float(np.linalg.norm(p - t)), j) for j, t in enumerate(tips) if j not in used]
        if cand:
            dist, j = min(cand)
            if dist <= R:
                out[key] = tips[j].tolist()
                used.add(j)
    return out


def walk_leg(mask, f, top):
    """발에서 위로 한 줄씩 올라가며, 지금 잡은 구간 폭이 다리 폭의 2.5배를 넘는 곳 = 몸통에 닿은 곳.
    반환: root · knee · foot, found(몸통에 닿는 곳을 찾았나). 못 찾으면 발과 그림 맨 위 사이 절반을 뿌리로 둔다."""
    H, W = mask.shape
    x, trail, widths, root_y = float(f[0]), [], [], None
    for yy in range(int(f[1]), top - 1, -1):
        row = mask[yy] > 0
        xi = int(round(x))
        if not row[xi]:
            near = [xi + dx for dx in (-1, 1, -2, 2, -3, 3) if 0 <= xi + dx < W and row[xi + dx]]
            if not near:
                break
            xi = near[0]
        s = xi
        while s > 0 and row[s - 1]:
            s -= 1
        e = xi
        while e < W - 1 and row[e + 1]:
            e += 1
        widths.append(e - s + 1)
        base = float(np.median(widths[:5]))
        if len(widths) > 5 and widths[-1] > max(2.5 * base, 6):
            root_y = yy
            break
        x = 0.7 * x + 0.3 * (s + e) / 2
        trail.append((x, yy))
    leg_w = float(np.median(widths[:5])) if widths else 4.0
    found = root_y is not None and bool(trail)
    if not found:
        root_y = f[1] - 0.5 * (f[1] - top)
        trail = [(f[0], f[1] - k * (f[1] - root_y) / 10) for k in range(11)]
    root = snap(mask, (trail[-1][0], root_y - 0.6 * leg_w))
    knee = snap(mask, trail[len(trail) // 2])
    return {"root": root, "knee": knee, "foot": f, "found": found, "leg_w": leg_w}


def merge_same_leg(mask, feet):
    """한 다리에 붙은 발 둘을 하나로. 낙서의 막대 다리는 한 다리를 두 줄(U 자)로 그려서 발 점 둘이 한 다리 양쪽에 붙고,
    서로 다른 박자로 당겨 다리가 찢어졌다 (QuickDraw 말 · 개 늘어남의 대부분).
    그림 안으로 걸어서 잰 거리가 곧은 거리와 거의 같으면(= 아래에서 바로 이어짐) 같은 다리다.
    따로 선 두 다리는 몸통까지 올라갔다 내려와야 해서 그림 안 거리가 훨씬 길다."""
    if len(feet) < 2:
        return feet
    size = max(mask.shape)
    cost = np.where(mask > 0, 1.0, np.inf)
    feet = list(feet)
    merged = True
    while merged and len(feet) > 1:
        merged = False
        for i in range(len(feet)):
            D, _ = MCP_Geometric(cost).find_costs([(int(feet[i][1]), int(feet[i][0]))])
            for j in range(i + 1, len(feet)):
                eu = float(np.linalg.norm(feet[i] - feet[j]))
                geo = float(D[int(feet[j][1]), int(feet[j][0])])
                if eu <= 0.15 * size and geo <= 1.5 * eu + 10:
                    lo = feet[i] if feet[i][1] >= feet[j][1] else feet[j]
                    feet[i] = snap(mask, ((feet[i][0] + feet[j][0]) / 2, lo[1]))
                    del feet[j]
                    merged = True
                    break
            if merged:
                break
    return feet


def build_quadruped(mask, pts):
    H, W = mask.shape
    feet = [snap(mask, f) for f in pts["feet"]]
    if len(feet) < 2:
        raise ValueError("발을 2개 이상 놓아 주세요")
    feet = merge_same_leg(mask, feet)  # 하나만 남아도 그 다리는 움직인다
    top = np.nonzero(mask.any(1))[0][0]
    legs = [walk_leg(mask, f, top) for f in feet]
    y_belly = int(min(l["root"][1] for l in legs))
    rx = sorted(float(l["root"][0]) for l in legs)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    band = np.zeros_like(mask, bool)
    band[: y_belly + 1, int(rx[0]): int(rx[-1]) + 1] = True
    cand = np.where(band & (mask > 0), dt, 0)
    if cand.max() <= 0:
        cand = dt
    cy, cx = np.unravel_index(np.argmax(cand), cand.shape)
    centre = np.array([cx, cy], np.float32)
    head = snap(mask, pts["head"])
    forward = 1 if head[0] >= centre[0] else -1
    hp = ridge(mask, centre, head)
    head_pins = [at(hp, 0.45), at(hp, 0.75), hp[-1]]
    tail_pins = None
    if pts.get("tail"):
        tp = ridge(mask, centre, pts["tail"])
        w = np.array([dt[int(y), int(x)] for x, y in tp])
        tip_w = np.median(w[-max(3, len(w) // 4):])
        root_i = int(0.6 * (len(w) - 1))
        for i in range(len(w) - 1, -1, -1):
            if w[i] > 2.0 * max(tip_w, 2.0):
                root_i = min(i, len(w) - 3)
                break
        root_i = max(0, root_i)
        tail_pins = [tp[root_i], tp[(root_i + len(tp) - 1) // 2], tp[-1]]
    # 걸음 박자: quadruped.build_rig 와 같은 규칙 (뒷다리 0 · 앞다리 1/4, 같은 쪽 안에서 번갈아)
    hind = sorted([l for l in legs if (l["root"][0] - centre[0]) * forward < 0], key=lambda l: l["root"][0])
    front = sorted([l for l in legs if (l["root"][0] - centre[0]) * forward >= 0], key=lambda l: l["root"][0])
    for group, start in ((hind, 0.0), (front, 0.25 if q.PATTERN == "walk" else 0.5)):
        for k, l in enumerate(group):
            l["phase"] = start + 0.5 * (k % 2)
    if not hind or not front:
        for k, l in enumerate(sorted(legs, key=lambda l: l["root"][0])):
            l["phase"] = 0.5 * (k % 2)
    pins = [centre]
    rig = {"forward": forward, "legs": [], "head": None, "tail": None, "y_belly": y_belly}
    for l in legs:
        rig["legs"].append({"i": len(pins), "phase": l["phase"]})
        pins += [l["root"], l["knee"], l["foot"]]
    rig["head"] = len(pins)
    pins += head_pins
    if tail_pins is not None:
        rig["tail"] = len(pins)
        pins += tail_pins
    pins0 = np.array(pins, np.float32)
    keep = drop_crowded_pins(pins0, rig, max(mask.shape))
    return pins0[keep], lambda t, H: q.pose(pins0, rig, t, H)[keep]


def drop_crowded_pins(pins0, rig, size):
    """서로 다른 줄(다리마다 · 머리 · 꼬리 · 몸 가운데)의 기준점이 거의 같은 자리에 있으면 덜 중요한 쪽을 ARAP 에서 뺀다.
    칠 없는 선 그림은 머리 · 꼬리 길이 다리 선을 지나가고, 다리 선 둘이 맞닿기도 해서 두 기준점이 같은 곳을
    서로 다른 쪽으로 당겨 그 사이가 찢어졌다. 중요도: 발 > 다리 뿌리 > 몸 가운데 > 무릎 · 머리 끝 > 머리 중간 > 꼬리.
    반환: 남길 기준점 번호 (pose 결과도 같은 번호만 쓴다)."""
    chain = np.zeros(len(pins0), int)  # 0 = 몸 가운데
    prio = np.full(len(pins0), 3.5)
    for k, leg in enumerate(rig["legs"]):
        i = leg["i"]
        chain[i:i + 3] = k + 1
        prio[i:i + 3] = (4, 3, 5)
    if rig["head"] is not None:
        h = rig["head"]
        chain[h:h + 3] = 100
        prio[h:h + 3] = (2, 2, 3)
    if rig["tail"] is not None:
        t = rig["tail"]
        chain[t:t + 3] = 101
        prio[t:t + 3] = 1
    R = max(12.0, CROWD_R * size)
    alive = np.ones(len(pins0), bool)
    order = sorted(range(len(pins0)), key=lambda j: (-prio[j], j))
    for a_i, a in enumerate(order):
        if not alive[a]:
            continue
        for b in order[a_i + 1:]:
            if alive[b] and chain[b] != chain[a] and np.linalg.norm(pins0[a] - pins0[b]) < R:
                alive[b] = False
    return np.nonzero(alive)[0]


CROWD_R = 0.04  # 기준점끼리 이보다 가까우면(긴 변 비율) 겹친 것으로 본다


def build_long(mask, pts):
    path = ridge(mask, pts["tail"], pts["head"])  # 꼬리 → 머리
    if len(path) < 12:
        raise ValueError("머리와 꼬리가 너무 가까워요")
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    med = float(np.median([dt[int(y), int(x)] for x, y in path]))
    k = max(3, int(med))
    pad = np.pad(path, ((k, k), (0, 0)), mode="edge")
    path = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in (0, 1)], 1)
    pins0, _ = sr.make_body_pins(path[:, ::-1])  # make_body_pins 는 (y, x)
    return pins0, lambda t, H: sr.pose_body(pins0, t)


WING_SOFT = 0.04  # 점 3개 날개 무게를 흐리는 폭 (그림 긴 변 비율, 가우스 시그마)


def soft_in_mask(w, dom, sigma):
    """dom 안에서만 흐리기 (바깥 0 이 섞여 가장자리 무게가 줄지 않게 나눠 준다). dom 밖은 0."""
    if sigma <= 0:
        return w
    dm = dom.astype(np.float32)
    num = cv2.GaussianBlur(w * dm, (0, 0), sigma)
    den = cv2.GaussianBlur(dm, (0, 0), sigma)
    return np.where(dom, num / np.maximum(den, 1e-6), 0).astype(np.float32)


def build_wings(mask, pts):
    """날개마다 [뿌리 위, 뿌리 아래, 날개 끝] 점 3개. 뿌리 두 점을 잇는 선이 경첩이다.
    움직이는 정도(무게)는 그림 안 거리로 정한다: 날개 끝에 가까우면 1, 경첩에 가까우면 0.
    거리는 경첩 바깥쪽 그림 안으로만 잰다 → 날개와 떨어진 머리 · 더듬이는 날개 끝에서 닿지 않아 0.
    선으로 자르는 경계(예전 부채꼴)가 없어서 경계에서 꺾이거나 찢기지 않는다.
    ARAP 는 쓰지 않는다 — 모양을 지키려 버텨서 날개를 좁히면 말려 들어갔다."""
    wings = []
    for w in pts.get("wings") or []:
        if len(w) < 3:
            continue
        a, b, tip = (np.asarray(p, np.float32) for p in w[:3])
        L = float(np.linalg.norm(b - a))
        if L < 4:
            continue
        l = (b - a) / L
        m = np.array([-l[1], l[0]], np.float32)
        mid = (a + b) / 2
        if (tip - mid) @ m < 0:  # 바깥 = 날개 끝 쪽
            m = -m
        yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
        e_pix = (xx - a[0]) * m[0] + (yy - a[1]) * m[1]
        dom = (mask > 0) & (e_pix > 0)
        if not dom.any():
            continue
        cost = np.where(dom, 1.0, np.inf)
        tip_s = snap(dom.astype(np.uint8), tip)
        d_tip, _ = MCP_Geometric(cost).find_costs([(int(tip_s[1]), int(tip_s[0]))])
        # 몸에서 이 픽셀까지 오는 길이 경첩 구간(위 끝 ~ 아래 끝)을 지나 오면 날개, 구간 밖의 선을 넘어 오면 날개가 아니다.
        # 옆모습 새의 머리는 경첩 끝 옆에서 몸과 바로 이어져 있어 「구간 밖」으로 판정된다.
        # (모양으로 긋는 부채꼴은 경계에서 기괴하게 꺾였고, 한 가지 거리 비율만으로는 머리와 앞모습 새 어깨를 같이 맞출 수 없었다)
        al_pix = (xx - a[0]) * l[0] + (yy - a[1]) * l[1]
        near_line = dom & (e_pix <= 3)
        seg_seeds = near_line & (al_pix >= -2) & (al_pix <= L + 2)
        out_seeds = near_line & ((al_pix < -2) | (al_pix > L + 2))
        if seg_seeds.any():
            d_seg, _ = MCP_Geometric(cost).find_costs(list(zip(*np.nonzero(seg_seeds))))
        else:  # 경첩 바로 바깥이 빈 종이면(날개를 몸 안에 그린 옆모습 새) 곧은 거리
            d_seg = np.where(dom, np.maximum(e_pix, 0), np.inf).astype(np.float64)
        if out_seeds.any():
            d_out, _ = MCP_Geometric(cost).find_costs(list(zip(*np.nonzero(out_seeds))))
        else:
            d_out = np.full(mask.shape, np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            belong = smooth(np.where(np.isfinite(d_out), (d_out - d_seg) / (0.1 * L + 4) + 0.5, 1.0))
            ok = np.isfinite(d_tip) & np.isfinite(d_seg)
            r = np.where(ok, d_seg / np.maximum(d_seg + d_tip, 1e-6), 0.0)  # 0 = 경첩, 1 = 날개 끝
        # 경첩 바로 바깥부터 돌기 시작해 날개 끝까지 25% 지점부터 통째로 (가만히 있는 구간이 길면 어깨에서 꺾여 보인다)
        wmap = (belong * smooth((r - 0.03) / 0.22) * smooth(e_pix / 4.0)).astype(np.float32)
        # 무게가 몇 px 안에서 0 → 1 로 바뀌면 경첩에서 먼 곳일수록 이웃 꼭짓점이 크게 엇갈려 삼각형이 뒤집혔다
        # (QuickDraw 날개 뒤집힘 > 1% 는 모두 이 점 3개 방식에서 나왔다). 그림 안에서만 흐리게 해 바뀌는 폭을 넓힌다
        wmap = soft_in_mask(wmap, dom, WING_SOFT * max(mask.shape))
        wmap = cv2.dilate(wmap, np.ones((5, 5), np.uint8))  # 윤곽선 위 꼭짓점도 값을 받게
        up = -1.0 if (tip - mid)[0] >= 0 else 1.0  # + 각은 오른쪽으로 뻗은 날개를 내린다
        wings.append({"a": a, "b": b, "tip": tip, "m": m, "mid": mid, "wmap": wmap, "up": up})
    if not wings:
        raise ValueError("날개마다 뿌리 위 · 뿌리 아래 · 날개 끝 세 점을 놓아 주세요")
    style = pts.get("style") or "updown"

    def deform(v0, t, H):
        w = 2 * math.pi * t
        v = v0.astype(np.float64).copy()
        for wg in wings:
            wm = wg["wmap"]
            ix = np.clip(np.round(v0[:, 0]).astype(int), 0, wm.shape[1] - 1)
            iy = np.clip(np.round(v0[:, 1]).astype(int), 0, wm.shape[0] - 1)
            wt = wm[iy, ix].astype(np.float64)
            if style == "fold":
                # 위에서 본 나비: 경첩을 따라 접혔다 펴진다 → 경첩에서 먼 거리만 cos 만큼 줄어든다
                th = WING_FOLD * (0.5 + 0.5 * math.sin(w))
                e = np.maximum((v0 - wg["a"]) @ wg["m"], 0)
                v -= np.outer(e * (1 - math.cos(th)) * wt, wg["m"])
            else:
                # 앞모습 새 · 옆모습 새: 경첩 가운데를 축으로 위아래
                # 위치를 섞으면 휘는 곳이 S 자로 꺾인다 → 각도를 무게만큼 나눠 돌린다 (어깨에서 둥글게 휜다)
                ang = wg["up"] * WING_FLAP * math.sin(w) * wt
                c, si = np.cos(ang), np.sin(ang)
                d = v0 - wg["mid"]
                rot = np.stack([d[:, 0] * c - d[:, 1] * si, d[:, 0] * si + d[:, 1] * c], 1) + wg["mid"]
                v += rot - v0
        v[:, 1] += (0.02 if style != "fold" else 0.008) * H * math.sin(w)  # 날개를 내려칠 때 몸이 뜬다
        return v.astype(np.float32)

    marks = np.array([x for wg in wings for x in (wg["a"], wg["b"], wg["tip"])], np.float32)
    wmap = np.max([wg["wmap"] for wg in wings], axis=0) * (mask > 0)
    return {"deform": deform, "marks": lambda t, H: deform(marks, t, H), "wmap": wmap}


# ---------- 날개: 칸 고르기 ----------
def wing_deformer(wings, style):
    """wings: [{"pivot", "m"(바깥 방향), "wmap"(픽셀별 무게), "up"}] → 프레임별 꼭짓점 함수."""
    def deform(v0, t, H, vw=None):
        """vw: 날개마다 꼭짓점별 무게 (animate 가 메시 위에서 다듬은 것). 없으면 픽셀 무게에서 바로 읽는다 (뿌리 표시 등)."""
        w = 2 * math.pi * t
        v = v0.astype(np.float64).copy()
        for wi, wg in enumerate(wings):
            if vw is not None:
                wt, mem = vw[wi] if isinstance(vw[wi], tuple) else (vw[wi], None)
            else:
                mem = None
                wm = wg["wmap"]
                ix = np.clip(np.round(v0[:, 0]).astype(int), 0, wm.shape[1] - 1)
                iy = np.clip(np.round(v0[:, 1]).astype(int), 0, wm.shape[0] - 1)
                wt = wm[iy, ix].astype(np.float64)
            if style == "fold":
                # 경첩(뿌리를 지나 바깥 방향에 수직인 선)에서 먼 거리만 cos 만큼 줄어든다
                th = WING_FOLD * (0.5 + 0.5 * math.sin(w))
                # 경첩 선 안쪽에 걸친 날개 부분도 선 쪽으로 줄인다 (바깥만 줄이면 비스듬한 나비의 앞날개 위쪽이
                # 선을 경계로 한쪽만 줄어 뭉개졌다). 무게가 0 인 몸은 그대로
                e = (v0 - wg["pivot"]) @ wg["m"]
                if "G" in wg:  # 칸 고르기: 누적표로 (뒤집힘 없음)
                    if mem is None:
                        wm2 = wg["member"]
                        ix = np.clip(np.round(v0[:, 0]).astype(int), 0, wm2.shape[1] - 1)
                        iy = np.clip(np.round(v0[:, 1]).astype(int), 0, wm2.shape[0] - 1)
                        mem = wm2[iy, ix].astype(np.float64)
                    g = np.sign(e) * np.interp(np.abs(e), wg["grid"], wg["G"])
                    v -= np.outer(g * (1 - math.cos(th)) * mem, wg["m"])
                else:
                    v -= np.outer(e * (1 - math.cos(th)) * wt, wg["m"])
            else:
                # 뿌리를 축으로 위아래, 무게만큼 각도를 나눠 돌린다
                ang = wg["up"] * WING_FLAP * math.sin(w) * wt
                c, si = np.cos(ang), np.sin(ang)
                d = v0 - wg["pivot"]
                rot = np.stack([d[:, 0] * c - d[:, 1] * si, d[:, 0] * si + d[:, 1] * c], 1) + wg["pivot"]
                v += rot - v0
        v[:, 1] += (0.02 if style != "fold" else 0.008) * H * math.sin(w)  # 날개를 내려칠 때 몸이 뜬다
        return v.astype(np.float32)
    return deform


def region_labels(d):
    """선으로 닫힌 칸 (페인트 통으로 채우는 칸). 색칠 공부 그림 · 윤곽선으로 그린 아이 그림에서 날개 · 몸 · 머리가 서로 다른 칸이 된다.
    선이 끊긴 곳은 3px 까지 메우고, 더 큰 틈은 선 끝을 가까운 선에 이어(bridge_gaps) 새로 닫힌 칸만 더한다.
    마스크가 안 채워진 열린 낙서는 여전히 칸이 안 생긴다 (그때는 점 3개 방식)."""
    if getattr(d, "_regions", None) is None:
        # 선은 결을 지운 그림에서 찾고 점 같은 작은 조각은 버린다: 크레파스 · 색연필 결의 어두운 점이 선으로 잡혀
        # 날개가 자잘한 칸으로 쪼개졌다 (안 고른 조각이 가만히 있어 날개가 얼룩덜룩하게 움직였다)
        g = cv2.cvtColor(cv2.medianBlur(d.crop, 5), cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_i, lab_i, st_i, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        ink = np.isin(lab_i, np.nonzero(st_i[:, cv2.CC_STAT_AREA] >= 30)[0][1:])
        # 검은 선이 없어도 색이 크게 바뀌는 곳은 칸 경계 (윤곽선 없이 크레파스로 칠한 날개는 선으로 닫힌 칸이 안 생긴다).
        # 크레파스 결 · 연한 그림자는 경계가 아니게: 먼저 가운데값으로 결을 지우고, 이웃 3px 안의 변화만 본다
        lab_img = cv2.cvtColor(cv2.medianBlur(d.crop, 5), cv2.COLOR_BGR2LAB)
        k3 = np.ones((3, 3), np.uint8)
        grad = [cv2.morphologyEx(lab_img[..., c], cv2.MORPH_GRADIENT, k3).astype(np.int32) for c in range(3)]
        color_edge = ((np.maximum(grad[1], grad[2]) > 18) | (grad[0] > 45)).astype(np.uint8)
        # 크레파스 결이 남긴 짧은 경계 조각은 버린다 (날개 안이 자잘한 칸으로 쪼개져 안 고른 조각이 가만히 있다가 찢겼다)
        n_e, lab_e, st_e, _ = cv2.connectedComponentsWithStats(color_edge, connectivity=8)
        long_e = np.nonzero(st_e[:, cv2.CC_STAT_AREA] >= max(40, int(0.1 * max(d.mask.shape))))[0]
        color_edge = np.isin(lab_e, long_e[long_e > 0])
        ink = ink | color_edge
        lab, ink_l = _closed_regions(d.mask, ink)
        # 닫히지 않은 선 낙서: 선 끝을 가까운 선에 이어 본다. 이어서 「새로」 생긴 칸만 더한다 —
        # 이미 있던 칸은 그대로 둔다 (색칠 그림 · 윤곽 그림의 칸이 바뀌지 않게)
        bridge = bridge_gaps(ink, d.mask)
        if bridge.any():
            lab_b, ink_b = _closed_regions(d.mask, ink | bridge)
            n0 = int(lab.max())
            for rid in set(np.unique(lab_b).tolist()) - {0}:
                R = lab_b == rid
                if (lab[R] == 0).mean() < 0.9:
                    continue
                R &= lab == 0
                n0 += 1
                lab[R] = n0
            ink_l = (ink_l | (bridge & (d.mask > 0))) & ~(lab > 0)
        d._regions, d._ink_r = lab, ink_l
    return d._regions, d._ink_r


def _closed_regions(mask, ink):
    free = ((mask > 0) & ~ink).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    small = np.nonzero(st[:, cv2.CC_STAT_AREA] < 15)[0]
    lab[np.isin(lab, small)] = 0
    # 마스크는 선보다 2~3px 넓게 잡혀서, 그림 바깥 둘레에 선 바깥쪽 얇은 띠가 칸 하나로 생긴다.
    # 이 띠를 칸으로 두면 날개 바깥 테두리 전체가 「몸에 붙은 곳」이 되어 테두리만 안 움직이고 번진다 → 선으로 친다
    edge = (mask > 0) & (cv2.erode(mask, np.ones((3, 3), np.uint8)) == 0)
    halo = np.unique(lab[edge])
    halo = halo[halo > 0]
    ink = ink | np.isin(lab, halo)
    lab[np.isin(lab, halo)] = 0
    return lab, ink


BRIDGE_MAX = 0.08  # 선 끝을 이을 수 있는 가장 먼 거리 (그림 긴 변 비율)


def bridge_gaps(ink, mask):
    """선 끝(뼈대 끝점)마다, 끝이 향하는 쪽(±60°)의 가까운 다른 선까지 잇는 짧은 선을 그린다.
    「다른 선」= 선을 따라가면 멀리 돌아가거나 이어지지 않는 선 (자기 자신 바로 옆은 안 됨).
    이은 선은 마스크 안에만 (그림 밖 종이를 가로지르지 않는다). 짧은 조각(눈 · 점)에서는 잇지 않는다."""
    from skimage.morphology import skeletonize
    H, W = ink.shape
    out = np.zeros((H, W), bool)
    D = BRIDGE_MAX * max(H, W)
    sk = skeletonize(ink > 0)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    ends = np.argwhere(sk & (nb == 1))
    if len(ends) == 0 or len(ends) > 200:
        return out
    n_c, comp, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    R = int(np.ceil(3 * D))
    for y, x in ends:
        if st[comp[y, x], cv2.CC_STAT_AREA] < 15:
            continue
        y0, y1, x0, x1 = max(0, y - R), min(H, y + R + 1), max(0, x - R), min(W, x + R + 1)
        win = ink[y0:y1, x0:x1] > 0
        # 끝이 향하는 쪽: 같은 뼈대 위 10px 안 점들의 가운데 → 끝
        sy, sx = np.nonzero(sk[y0:y1, x0:x1] & (comp[y0:y1, x0:x1] == comp[y, x]))
        near = (sy + y0 - y) ** 2 + (sx + x0 - x) ** 2 <= 100
        v = np.array([x - (sx[near] + x0).mean(), y - (sy[near] + y0).mean()])
        if np.linalg.norm(v) < 1:
            continue
        v /= np.linalg.norm(v)
        geo, _ = MCP_Geometric(np.where(win, 1.0, np.inf)).find_costs([(y - y0, x - x0)])
        ty, tx = np.nonzero(win)
        dx, dy = tx + x0 - x, ty + y0 - y
        e = np.hypot(dx, dy)
        g = geo[ty, tx]
        cos = (dx * v[0] + dy * v[1]) / np.maximum(e, 1e-6)
        ok = (e > 2) & (e <= D) & (cos >= 0.5) & ~(g <= 2.5 * e + 10)
        if not ok.any():
            continue
        score = np.where(ok, e * (2 - cos), np.inf)
        j = int(np.argmin(score))
        p, q = (int(x), int(y)), (int(tx[j] + x0), int(ty[j] + y0))
        n = max(2, int(e[j]))
        ly = np.linspace(p[1], q[1], n).round().astype(int)
        lx = np.linspace(p[0], q[0], n).round().astype(int)
        if (mask[ly, lx] > 0).mean() < 0.9:
            continue
        line = np.zeros((H, W), np.uint8)
        cv2.line(line, p, q, 1, 3)
        out |= line > 0
    return out


def region_at(d, x, y, r=6):
    """누른 곳의 칸 번호. 선 위를 눌렀으면 r px 안에서 가장 가까운 칸."""
    lab, _ = region_labels(d)
    H, W = lab.shape
    x, y = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
    win = lab[y0:y1, x0:x1]
    if not (win > 0).any():
        return 0
    yy, xx = np.nonzero(win > 0)
    j = np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2)
    return int(win[yy[j], xx[j]])


def regions_along(d, points):
    """문지른 길(점 목록)이 지나간 칸들. 점 사이를 이어서 빈틈 없이 본다."""
    lab, _ = region_labels(d)
    H, W = lab.shape
    line = np.zeros((H, W), np.uint8)
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for p0, p1 in zip(pts, pts[1:] or pts):
        cv2.line(line, p0, p1, 1, 3)
    ids = set(np.unique(lab[line > 0]).tolist()) - {0}
    if not ids and pts:  # 선 위만 지나갔으면 가장 가까운 칸
        r = region_at(d, *pts[0])
        ids = {r} if r else set()
    return sorted(ids)


def nearest_region(d):
    """픽셀마다 가장 가까운 칸 번호 (선 · 바깥 둘레 픽셀이 어느 칸에 붙는지)."""
    if getattr(d, "_near", None) is None:
        lab, _ = region_labels(d)
        src = (lab == 0).astype(np.uint8)  # 칸 픽셀이 0 → 가장 가까운 칸 픽셀을 찾는다
        dist, idx = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(idx.max() + 1, np.int32)
        lut[idx[lab > 0]] = lab[lab > 0]
        near = lut[idx]
        # 칸에서 (선 굵기 + 바깥 띠) 보다 먼 픽셀은 어느 칸에도 안 붙인다: 검게 꽉 칠한 머리 · 몸 안쪽이 감싼 날개 것이 되지 않게.
        # line_width 는 실제 굵기의 절반쯤으로 나오고(검은 픽셀 거리의 중앙값), 바깥 테두리는 칸에서 선 하나 + 띠 3px 떨어져 있어
        # 2 × line_width + 6 으로 둔다 (잰 값: 바깥 테두리 거리 중앙값 8~12px). 선 굵기는 색 경계가 섞이지 않은 진짜 검은 선에서 잰다
        cap = 2 * line_width(d.ink > 0) + 6
        near[dist > cap] = 0
        # 검게 칠한 덩어리(머리 · 몸)는 가장자리라도 칸에 안 붙인다: 가장자리만 양쪽 날개에 붙고 가운데는 남아 머리가 끼여 뒤집혔다.
        # 덩어리 = 칸 아닌 픽셀에서 지름 2×cap 원보다 두꺼운 곳 (가는 선 · 바깥 테두리 띠는 이 연산에서 사라진다)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cap + 1, 2 * cap + 1))
        blob = cv2.morphologyEx(((lab == 0) & (d.mask > 0)).astype(np.uint8), cv2.MORPH_OPEN, disk)
        near[cv2.dilate(blob, disk) > 0] = 0
        d._near = near
    return d._near


def line_width(ink):
    """선 굵기 (px). 칸과 칸 사이를 건너는 거리로 쓴다."""
    return max(2, int(round(2 * np.median(cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)[ink])))) if ink.any() else 3


def crowd_root(lab, M, mask, comp=None, k=11):
    """날개 뿌리 자동: 날개가 몸(날개 아닌 칸)과 맞닿은 곳 중 날개 칸이 가장 많이 모인 점.
    나비 날개 판들은 몸통 쪽 한 점(진짜 뿌리)으로 모이고, 머리와 옆으로 닿기만 한 곳은 칸 한두 개뿐이다.
    (맞닿은 곳 전체의 가운데를 뿌리로 하면 앞날개 한가운데가 뿌리가 됐다)"""
    comp = M if comp is None else comp
    # 맞닿은 곳 = 날개 아닌 「칸」(몸통 · 머리)과 선 하나를 사이에 두고 닿는 곳. 선 · 바깥 둘레 픽셀은 안 센다
    # (세면 날개 바깥 테두리가 맞닿은 곳이 되어 뿌리가 날개 끝에 잡혔다)
    rest = (lab > 0) & ~M & (mask > 0)
    contact = comp & (cv2.dilate(rest.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0)
    if not contact.any():
        return None, contact
    rr = max(8, int(0.06 * max(mask.shape)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    crowd = np.zeros(mask.shape, np.float32)
    for rid in set(np.unique(lab[comp]).tolist()) - {0}:  # 이 날개 덩어리의 칸만 (반대쪽 날개까지 세면 뿌리가 몸 가운데로 갔다)
        crowd += cv2.dilate((lab == rid).astype(np.uint8), disk)
    cy, cx = np.nonzero(contact)
    sc = crowd[cy, cx]
    best = sc >= sc.max() - 0.5
    j = np.argmin((cx - cx[best].mean()) ** 2 + (cy - cy[best].mean()) ** 2)
    return np.array([cx[j], cy[j]], np.float32), contact


def wing_regions(d, ids):
    """고른 칸 → 날개 범위. 고른 칸에 갇힌 칸(날개 안 줄무늬)과 그 둘레 선까지 더한다."""
    lab, ink = region_labels(d)
    mask = d.mask > 0
    S = np.isin(lab, list(ids)) & mask
    if not S.any():
        return S
    lw = line_width(ink)
    k = 2 * lw + 1
    # 고른 칸 + 둘레 선으로 막힌 안쪽(구멍 = 날개 안 무늬)을 채운다. 날개 덩어리마다 따로:
    # 양쪽 날개를 합쳐서 보면 그 사이 몸통이 「갇힌 칸」이 됐다 (위에서 본 나비)
    H, W = S.shape
    M = S.copy()
    n_g, groups = cv2.connectedComponents((cv2.dilate(S.astype(np.uint8), np.ones((k, k), np.uint8)) > 0).astype(np.uint8))
    for g in range(1, n_g):
        Sg = S & (groups == g)
        shell = cv2.dilate(Sg.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        shell = (shell & (ink | Sg)) | Sg
        ff = np.pad(shell.astype(np.uint8) * 255, 1)
        cv2.floodFill(ff, np.zeros((H + 4, W + 4), np.uint8), (0, 0), 128)
        inside = (ff[1:-1, 1:-1] != 128) & mask
        M |= Sg
        # 갇힌 칸이라도 날개 칸 「하나」에 둘러싸인 것만 (날개 무늬 · 크레파스 결 조각).
        # 양쪽 윗날개가 감싼 머리처럼 여러 날개 칸에 걸쳐 둘러싸인 칸은 몸이다 (위에서 본 나비)
        ring_k = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        for rid in set(np.unique(lab[inside & ~Sg]).tolist()) - {0}:
            R = lab == rid
            if not (R & ~inside).sum() == 0:
                continue
            ring = (cv2.dilate(R.astype(np.uint8), ring_k) > 0) & (lab > 0) & ~R
            around = lab[ring]
            if len(around) == 0:
                continue
            vals, cnt = np.unique(around, return_counts=True)
            top = vals[np.argmax(cnt)]
            if np.isin(top, list(ids)) and cnt.max() / len(around) >= 0.7:
                M |= R
    # 둘레 선: 칸이 아닌 픽셀(선 · 바깥 둘레)은 가장 가까운 칸 것이다. 날개 바깥 테두리는 날개 것, 날개와 몸 사이 선은 반씩.
    # (선 굵기만큼만 붙이면 굵기를 잘못 잴 때 테두리가 날개에서 빠져 혼자 가만히 있다가 늘어났다)
    near = nearest_region(d)
    cap = 2 * line_width(d.ink > 0) + 6

    def border(X):
        # 날개와 몸이 맞닿은 선은 몸 것: 반씩 나누면 날개가 움직일 때 몸 윤곽선이 점선처럼 끊겼다.
        # 날개 안쪽 가장자리 선은 날개가 움직이면 몸 뒤로 들어가 가려진다 (그리는 순서: 날개 먼저, 몸 나중)
        ids_x = list(set(np.unique(lab[X]).tolist()) - {0})
        body = ((lab > 0) & ~np.isin(lab, ids_x)).astype(np.uint8)
        far_body = cv2.distanceTransform(1 - body, cv2.DIST_L2, 3) > cap if body.any() else np.ones_like(X)
        return X | (mask & (lab == 0) & np.isin(near, ids_x) & far_body)

    M = border(M)
    # (2026-10-26) 문지르다 빠진 줄무늬 칸을 자동으로 채우는 규칙은 뺐다. 「뿌리에서 날개 쪽 · 몸 쪽 칸과는 뿌리 둘레에서만 닿음」으로
    # 골랐더니 양쪽 날개가 머리 근처에서 닿는 그림(위에서 본 나비 · 크레파스 날개)에서 몸통이 날개로 들어왔다.
    # 빠진 칸은 화면에서 빨간 날개 안의 흰 칸으로 보이니 사람이 눌러 채운다. 완전히 갇힌 무늬만 위에서 자동으로 채운다.
    return M


def wing_groups(d, M, k):
    """날개 범위를 날개 하나씩으로 나눈다. 같은 날개 안의 칸(바깥 칸 · 줄무늬 · 겹친 앞뒤 날개)은 경계를 길게 나눠 갖고,
    양쪽 날개는 머리 위에서 점처럼 닿을 뿐이다 → 둘레의 12% 넘게 맞닿은 칸끼리만 묶는다.
    (픽셀로 이어졌는지만 보면 위에서 본 나비의 양쪽 날개가 한 덩어리가 되어 뿌리가 하나만 잡히고 한쪽으로 쏠려 접혔다)"""
    lab, _ = region_labels(d)
    near = nearest_region(d)
    ids = sorted(set(np.unique(lab[M]).tolist()) - {0})
    if not ids:
        return []
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    grown = {r: cv2.dilate((lab == r).astype(np.uint8), ker) > 0 for r in ids}
    rim = {r: max(1, int((grown[r] & (lab > 0) & (lab != r)).sum())) for r in ids}
    parent = {r: r for r in ids}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = int((grown[a] & (lab == b)).sum())
            if shared and shared / min(rim[a], rim[b]) >= 0.12:
                parent[find(a)] = find(b)
    out = []
    for root in sorted({find(r) for r in ids}):
        members = [r for r in ids if find(r) == root]
        comp = M & (np.isin(lab, members) | np.isin(near, members))
        if comp.sum() >= 0.005 * M.sum():
            out.append(comp)
    return out


def build_wings_regions(d, mask, pts):
    """칸 고르기: 고른 칸이 날개. 떨어진 덩어리마다 날개 하나 (앞모습 새는 양쪽 날개가 따로).
    뿌리 = 날개가 몸(안 고른 칸)과 맞닿은 곳의 가운데 (사용자가 옮길 수 있다). 바깥 방향 = 뿌리 → 날개 무게중심.
    무게 = 맞닿은 곳에서 날개 안으로 잰 거리: 맞닿은 곳은 0 (몸에 붙어 있고), 날개 길이의 25% 부터 통째로.
    고른 칸만 움직이므로 점 3개 방식처럼 몸통이 날개로 잡히는 일이 없다."""
    M = wing_regions(d, pts.get("regions") or [])
    if not M.any():
        raise ValueError("움직일 날개 칸을 눌러 골라 주세요")
    user = [np.asarray(p, np.float32) for p in pts.get("pivots") or []]
    reg, ink = region_labels(d)
    k_line = 2 * line_width(ink) + 1
    groups = wing_groups(d, M, k_line)
    # 서로 닿아 있고 같은 쪽으로 뻗은 덩어리는 합친다: 비스듬한 나비의 앞날개 · 뒷날개는 뿌리 근처에서만 짧게 닿아
    # 따로 나뉘는데, 따로 움직이면 둘 사이가 벌어져 찢겼다. 위에서 본 나비의 좌우 날개는 반대 방향이라 안 합쳐진다
    ker = np.ones((2 * k_line + 1, 2 * k_line + 1), np.uint8)

    def outward(comp):
        root, _c = crowd_root(reg, M, mask, comp, k_line)
        ys, xs = np.nonzero(comp)
        c = np.array([xs.mean(), ys.mean()])
        v = c - (root if root is not None else c)
        return v / (np.linalg.norm(v) + 1e-6)

    merged = True
    while merged and len(groups) > 1:
        merged = False
        dirs = [outward(g) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                touch = (cv2.dilate(groups[i].astype(np.uint8), ker) > 0) & groups[j]
                if touch.any() and dirs[i] @ dirs[j] > 0:
                    groups[i] = groups[i] | groups[j]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    wings = []
    for comp in groups:
        ys_c, xs_c = np.nonzero(comp)
        centroid = np.array([xs_c.mean(), ys_c.mean()], np.float32)
        auto, contact = crowd_root(reg, M, mask, comp, k_line)
        if auto is None:  # 몸과 안 닿은 날개 (따로 그린 날개): 그림 전체 무게중심 쪽 가장자리
            ys, xs = np.nonzero(mask)
            c0 = np.array([xs.mean(), ys.mean()], np.float32)
            py, px = np.nonzero(comp)
            j = np.argmin((px - c0[0]) ** 2 + (py - c0[1]) ** 2)
            auto = np.array([px[j], py[j]], np.float32)
            contact = np.zeros_like(comp)
            contact[py[j], px[j]] = True
        pivot = auto
        if user:  # 사용자가 옮긴 뿌리: 이 날개의 자동 뿌리에 가장 가까운 것
            j = int(np.argmin([np.linalg.norm(u - auto) for u in user]))
            pivot = user[j]
        m = centroid - pivot
        m = m / np.linalg.norm(m) if np.linalg.norm(m) > 1 else np.array([-1, 0], np.float32)
        # 뿌리 둘레의 맞닿은 곳만 고정점(무게 0). 나머지 맞닿은 곳(머리 옆)은 날개의 35% 만큼은 같이 움직여 덜 번지게
        py, px = np.nonzero(comp)
        ext = float(np.sqrt(((px - pivot[0]) ** 2 + (py - pivot[1]) ** 2).max()))
        cy, cx = np.nonzero(contact)
        near_root = (cx - pivot[0]) ** 2 + (cy - pivot[1]) ** 2 <= (0.2 * ext) ** 2
        if not near_root.any():
            near_root[np.argmin((cx - pivot[0]) ** 2 + (cy - pivot[1]) ** 2)] = True
        cost = np.where(comp, 1.0, np.inf)
        D, _ = MCP_Geometric(cost).find_costs(list(zip(cy[near_root], cx[near_root])))
        D = np.where(np.isfinite(D), D, 0)
        reach = max(float(D.max()), 1.0)
        # 위아래(회전) 무게: 뿌리에서 날개 길이의 45% 에 걸쳐 천천히 오른다. 25% 로 급하게 올리면 넓은 나비 날개가
        # 뿌리 근처에서 비틀려 삼각형이 뒤집혔다 (check_wings 「뒤집힌 삼각형」).
        # 뿌리 아닌 맞닿은 곳(머리 · 몸 옆)을 덜 움직이게 하던 감쇠는 뺐다: 메시를 잘라(cut_mesh) 날개가 몸 위로 미끄러지니
        # 필요 없고, 남겨 두면 몸 쪽 가장자리에서 무게가 급히 바뀌어 윗날개 안쪽이 뒤집혔다
        wmap = smooth(D / (0.45 * reach))
        # 접기용 소속: 뿌리 바로 둘레만 0 이고 날개 안은 거의 1 (경첩에서 멀어질수록 줄이는 정도는 아래 누적표가 맡는다)
        member = smooth(D / (0.06 * reach))
        wmap = (wmap * comp).astype(np.float32)
        member = (member * comp).astype(np.float32)
        # 윤곽선 위 꼭짓점도 값을 받게 바깥(빈 종이)으로만 번지게. 몸 쪽으로는 안 번진다
        grow = comp | ~(mask > 0)
        wmap = cv2.dilate(wmap, np.ones((5, 5), np.uint8)) * grow
        member = cv2.dilate(member, np.ones((5, 5), np.uint8)) * grow
        # 접기 누적표: 경첩에서 거리 u 만큼 나아갈 때 그 자리에서 줄이는 비율 ramp(u) 를 쌓는다 → 바깥 점이 안쪽 점을
        # 앞지르지 않아 삼각형이 뒤집히지 않는다 (「거리 × 무게」로 한 번에 당기면 무게가 가파른 곳에서 뒤집혔다)
        e_all = (np.stack([px, py], 1) - pivot) @ m
        e_max = max(float(np.abs(e_all).max()), 1.0)
        grid = np.linspace(0, e_max * 1.2, 256)
        ramp = smooth(grid / (0.25 * e_max))
        G = np.concatenate([[0], np.cumsum((ramp[1:] + ramp[:-1]) / 2 * np.diff(grid))])
        up = -1.0 if m[0] >= 0 else 1.0
        wings.append({"pivot": pivot, "m": m, "wmap": wmap, "member": member, "grid": grid, "G": G, "up": up,
                      "comp": comp, "root_r": 0.2 * ext})
    style = pts.get("style") or "updown"
    deform = wing_deformer(wings, style)
    marks = np.array([wg["pivot"] for wg in wings], np.float32)
    wmap = np.max([wg["wmap"] for wg in wings], axis=0) * (mask > 0)
    # 어느 칸에도 안 붙은 픽셀 중 그림 바깥 둘레 근처(넓은 바깥 띠)만 이웃을 따라가게 한다.
    # 그림 안쪽에 검게 칠한 머리 · 몸은 0 으로 고정: 따라가게 했더니 양쪽 날개 무게를 반씩 받아 머리가 끼여 찌그러졌다
    unassigned = (mask > 0) & (reg == 0) & (nearest_region(d) == 0)
    outer = (cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3) <= 2 * (2 * line_width(d.ink > 0) + 6))
    free = unassigned & outer
    return {"deform": deform, "marks": lambda t, H: deform(marks, t, H), "wmap": wmap, "M": M,
            "pivots": [r1(wg["pivot"]) for wg in wings], "wings": wings, "free": free}


def cut_mesh(rig, v0, tris):
    """날개가 다른 부분(반대쪽 날개 · 머리 · 몸)과 맞닿은 곳에서 꼭짓점을 둘로 나눠 날개가 따로 움직이게 한다.
    뿌리 둘레만 몸과 붙여 둔다 (날개가 몸에 매달려 있게). 이어진 한 장 메시로는 양쪽 윗날개가 머리 위에서 맞닿은
    나비를 위아래로 돌리면 서로 파고들어 사이 삼각형이 뒤집혔다. 자르면 겹치기만 하고, 그리는 순서로 날개가 몸 뒤로 간다.
    반환: 꼭짓점, 삼각형, 잘린 자리 꼭짓점 표시(무게를 이웃에서 받게)."""
    H, W = rig["free"].shape
    part = -np.ones((H, W), np.int32)
    for gi, wg in enumerate(rig["wings"]):
        part[wg["comp"]] = gi
    cen = v0[tris].mean(1)
    tp = part[np.clip(np.round(cen[:, 1]).astype(int), 0, H - 1), np.clip(np.round(cen[:, 0]).astype(int), 0, W - 1)]
    verts = list(v0)
    new_tris = tris.copy()
    seam = set()
    incident = {}
    for ti, t in enumerate(tris):
        for v in t:
            incident.setdefault(int(v), set()).add(int(tp[ti]))
    copy_of = {}
    for v, parts in incident.items():
        if len(parts) < 2:
            continue
        primary = -1 if -1 in parts else min(parts)
        for q in parts:
            if q == primary:
                continue
            wg = rig["wings"][q]
            if primary == -1 and np.linalg.norm(v0[v] - wg["pivot"]) <= wg["root_r"]:
                continue  # 뿌리 둘레: 몸과 붙여 둔다
            copy_of[(v, q)] = len(verts)
            verts.append(v0[v].copy())
            seam.update((v, copy_of[(v, q)]))
    for ti, t in enumerate(tris):
        q = int(tp[ti])
        for j in range(3):
            key = (int(t[j]), q)
            if key in copy_of:
                new_tris[ti, j] = copy_of[key]
    vs = np.array(verts, np.float32)
    seam_v = np.zeros(len(vs), bool)
    seam_v[list(seam)] = True
    rig["tri_part"] = tp  # 삼각형마다 날개 번호(-1 = 몸): vertex_weights 가 꼭짓점 무게를 부분에 맞춘다
    return vs, new_tris.astype(np.int32), seam_v


def vertex_weights(rig, v0, tris, iters=40, seam_v=None):
    """날개마다 꼭짓점별 무게. 어느 칸에도 안 붙은 꼭짓점(날개 끝 뾰족한 곳의 넓은 바깥 띠 등)은
    이웃 꼭짓점 무게의 평균을 따라가게 메시 위에서 퍼뜨린다 — 옆은 움직이는데 혼자 가만히 있어 늘어나던 것(찢김 지표 15~20배)을 없앤다."""
    import scipy.sparse as sp
    n = len(v0)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    free = rig["free"]
    ix = np.clip(np.round(v0[:, 0]).astype(int), 0, free.shape[1] - 1)
    iy = np.clip(np.round(v0[:, 1]).astype(int), 0, free.shape[0] - 1)
    fv = free[iy, ix]
    if seam_v is not None:
        fv = fv | seam_v  # 잘린 자리: 픽셀이 몸 쪽에 걸려도 자기 쪽(날개 · 몸) 이웃의 무게를 받는다
    out = []
    tp = rig.get("tri_part")
    for gi, wg in enumerate(rig["wings"]):
        # 이 날개 삼각형에 쓰이는 꼭짓점만 날개 무게를 받는다. 몸 삼각형에만 쓰이는 꼭짓점은 0 으로 고정:
        # 두 부분이 만나는 테두리에서 몸 쪽으로 분류된 삼각형이 날개 무게를 받아 반쯤 움직이며 늘어났다
        elig = np.ones(n, bool)
        if tp is not None:
            elig[:] = False
            elig[np.unique(tris[tp == gi])] = True
        pair = []
        for key in ("wmap", "member"):
            w = wg[key][iy, ix].astype(np.float64) * elig
            # 날개 삼각형의 꼭짓점인데 픽셀은 날개 범위 밖(배 · 머리 쪽 테두리)에 걸친 것도 이웃 날개 꼭짓점을 따라가게
            f_g = (fv | ~wg["comp"][iy, ix]) & elig
            for _ in range(iters):
                w[f_g] = (A @ w)[f_g] / deg[f_g]
            pair.append(w)
        out.append(tuple(pair))
    return out


def snap_points_to_regions(d, mask, pts):
    """점 3개 → 칸. 날개 끝 점이 있는 칸에서 시작해, 경계를 길게(둘레 12% 넘게) 나눠 갖는 이웃 칸으로 넓혀 가되
    경첩 선(뿌리 두 점) 안쪽 = 몸 쪽에 대부분 있는 칸에서는 멈춘다.
    뿌리 두 점은 「어느 쪽이 몸인지」만 알려 주면 되니 몇 px 빗나가도 결과가 같다.
    (처음엔 점 3개 무게 지도에서 칸 넓이 절반 넘게 움직이면 날개로 쳤는데, 그 무게 지도가 날개가 몸에 붙은 폭이
    뿌리 두 점보다 조금만 넓어도 날개 대부분을 0 으로 만들어 정답 점으로도 실패했다)
    날개 끝이 칸 안에 없거나(선이 크게 끊긴 그림) 칸이 안 잡히면 None → 점 3개 그대로."""
    lab, ink = region_labels(d)
    k = 2 * line_width(ink) + 1
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    rim_cache, grown_cache = {}, {}

    def grown(r):
        if r not in grown_cache:
            grown_cache[r] = cv2.dilate((lab == r).astype(np.uint8), ker) > 0
            rim_cache[r] = max(1, int((grown_cache[r] & (lab > 0) & (lab != r)).sum()))
        return grown_cache[r]

    picked, per_wing = set(), []
    for w in pts.get("wings") or []:
        if len(w) < 3:
            continue
        A, B, T = (np.asarray(p, np.float32) for p in w[:3])
        if np.linalg.norm(B - A) < 4:
            continue
        l = (B - A) / np.linalg.norm(B - A)
        m = np.array([-l[1], l[0]], np.float32)
        if (T - (A + B) / 2) @ m < 0:
            m = -m
        # 안쪽 · 바깥쪽은 경첩 구간 둘레 띠(구간 앞뒤로 절반씩 더)의 픽셀로 본다. 선을 끝없이 늘려 보면
        # 경첩 너머로 길게 뻗은 몸(잠자리 긴 몸통)이 점이 조금만 기울어도 바깥쪽이 됐다.
        # 더해서 칸 중심이 경첩 구간 너머에 있으면, 넘어간 거리의 30% 넘게 바깥으로 나가 있어야 날개 쪽:
        # 경첩 너머 선 바로 옆에 놓인 색칠 나비 배 끝 칸은 빠지고, 멀리 퍼진 날개 칸(뒤에 살짝 보이는 날개)은 남는다.
        # (픽셀마다 부채꼴로 세면 크게 휘감긴 앞날개 바깥 칸이 날개인데도 빠졌다)
        L = float(np.linalg.norm(B - A))
        e_pix = (xx - A[0]) * m[0] + (yy - A[1]) * m[1]
        along = (xx - A[0]) * l[0] + (yy - A[1]) * l[1]
        outer = e_pix > 0
        band = (along >= -0.5 * L) & (along <= 1.5 * L)

        def wing_side(Q):
            Qb = Q & band
            frac = outer[Qb].mean() if Qb.sum() >= 20 else outer[Q].mean()
            if frac < 0.7:
                return False
            e_c, a_c = float(e_pix[Q].mean()), float(along[Q].mean())
            over = max(0.0, -a_c, a_c - L)
            return e_c > 0 and e_c >= 0.3 * over

        # 날개 끝은 보통 그림 맨 끝 테두리 선 위에 찍힌다 (테두리 선 + 바깥 띠 8~10px) → 선 굵기 3배 안에서 가장 가까운 칸
        start = region_at(d, *T, r=3 * k)
        if not start:
            continue
        seen, todo = {start}, [start]
        while todo:
            r = todo.pop()
            g = grown(r)
            for q in set(np.unique(lab[g]).tolist()) - {0} - seen:
                Q = lab == q
                # 칸의 70% 넘게가 경첩 바깥이어야 날개. 절반으로 두면 가는 몸통(잠자리)이 뿌리 점이 10px 만 빗나가도
                # 경첩 선이 몸통 한가운데로 들어가 몸통이 날개로 들어왔다. 날개 칸은 거의 다 바깥이라 영향 없다
                if not wing_side(Q):  # 몸 쪽 칸
                    continue
                shared = int((g & Q).sum())
                grown(q)
                if shared / min(rim_cache[r], rim_cache[q]) < 0.12:  # 점처럼만 닿은 칸 (옆모습 새 머리 · 반대쪽 날개)
                    continue
                seen.add(q)
                todo.append(q)
        per_wing.append(seen)
    # 양쪽 날개 끝에서 모두 닿은 칸은 몸이다 (옆모습 새 낙서: 몸 전체가 칸 하나라 양쪽 끝 점이 같은 칸에 들어가
    # 몸통 통째로 날개가 됐다). 같은 뿌리를 쓰는 앞 · 뒤 날개(비스듬한 나비)는 끝이 서로 다른 칸이라 걸리지 않는다
    for i, s in enumerate(per_wing):
        others = set().union(*(t for j, t in enumerate(per_wing) if j != i))
        picked |= s - others
    if not picked:
        return None
    return sorted(int(r) for r in picked)


def resolve_wings(d, mask, pts):
    """날개 점 → 실제로 쓸 방식. 칸 고르기면 그대로, 점 3개면 칸에 맞춰 칸 고르기로 (안 되면 점 3개 그대로).
    반환: (rig, 쓴 방식 "regions" | "points-snapped" | "points")"""
    if pts.get("mode") == "regions":
        return build_wings_regions(d, mask, pts), "regions"
    if pts.get("snap", True):
        ids = snap_points_to_regions(d, mask, pts)
        if ids:
            return build_wings_regions(d, mask, {"mode": "regions", "regions": ids, "style": pts.get("style")}), "points-snapped"
    return build_wings(mask, pts), "points"


def wing_overlay(d, pts):
    """점 고치기 화면에 겹칠 「움직일 범위」 그림 (빨갈수록 많이 움직인다, 칸 고르기는 고른 칸을 연한 주황으로).
    점 3개 방식은 뿌리 점을 몸 안쪽에 두면 몸통도 날개로 잡히는데, 점만으로는 막을 수 없어 어른이 보고 고치게 한다."""
    mask, _ = d.masks("wings")
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), np.uint8)
    out = {"pivots": []}
    if pts.get("mode") == "regions":
        if not pts.get("regions"):
            wm = np.zeros(mask.shape, np.float32)
        else:
            rig = build_wings_regions(d, mask, pts)
            wm = rig["wmap"]
            out["pivots"] = rig["pivots"]
            rgba[rig["M"]] = (60, 150, 250, 70)  # 고른 칸: 연한 주황 (BGRA)
    else:
        rig, used = resolve_wings(d, mask, pts)
        wm = rig["wmap"]
        out["used"] = used
        if used == "points-snapped":
            rgba[rig["M"]] = (60, 150, 250, 70)  # 칸에 맞춘 범위: 연한 주황
    a = np.clip(wm * 150, 0, 255).astype(np.uint8)
    red = a > 0
    rgba[red, 0], rgba[red, 1], rgba[red, 2] = 60, 60, 235  # BGR 빨강
    rgba[..., 3] = np.maximum(rgba[..., 3], a)
    ok, png = cv2.imencode(".png", rgba)
    out["moving"] = round(float((wm > 0.5).sum() / max(1, (mask > 0).sum())), 3)
    out["overlay"] = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    return out


def build_blob(mask, pts):
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
    pins0, rigs = sr.make_pins([], (cy, cx), mask)  # 가운데 · 왼 · 오른 · 위 · 아래

    def pose(t, H):
        p = sr.pose(pins0, rigs, t, H, False)
        # 2차 움직임: 위쪽은 조금 늦게 따라와 말랑하게 출렁인다 (스프링 대신 늦은 자세를 섞는다)
        late = sr.pose(pins0, rigs, (t - 0.08) % 1.0, H, False)
        for j, f in ((3, 0.6), (1, 0.25), (2, 0.25)):
            if j < len(p):
                p[j] = p[j] + f * (late[j] - p[j])
        return p

    return pins0, pose


BUILD = {"quadruped": build_quadruped, "long": build_long, "wings": build_wings, "blob": build_blob}


def animate(d, kind, pts):
    """프레임마다 꼭짓점 위치를 만든다. 그리는 것은 브라우저(Pixi MeshSimple)가 한다."""
    if kind not in BUILD:
        raise ValueError(f"모르는 종류: {kind}")
    notice = None
    if kind == "quadruped" and pts.get("no_body"):
        # 몸 · 다리가 없는 그림(얼굴만)을 네 발로 억지로 움직이면 크게 늘어난다 → 안전한 몸통만으로
        kind, pts = "blob", {}
        notice = "다리를 못 찾아서(얼굴만 그린 그림 같아요) 몸통만으로 움직였어요. 네 발로 움직이려면 발 점을 직접 놓아 주세요."
    mask, alpha = d.masks(kind)
    v0, tris, comps, mesh_kind = build_mesh(mask)
    H, W = mask.shape
    frames, pin_frames = [], []
    used = None
    if kind == "wings":
        rig, used = resolve_wings(d, mask, pts)
    else:
        if kind == "quadruped" and pts.get("snap", True):
            pts = snap_to_tips(d, pts)
        rig = BUILD[kind](mask, pts)
    if isinstance(rig, dict):  # 날개: ARAP 없이 꼭짓점을 직접 옮긴다
        vw = None
        if "free" in rig:
            v0, tris, seam_v = cut_mesh(rig, v0, tris)
            vw = vertex_weights(rig, v0, tris, seam_v=seam_v)
        if vw is not None:
            # 그리는 순서: 움직이는 날개 삼각형을 먼저, 몸 · 머리를 나중에 → 날개가 돌다 겹치면 몸 뒤로 들어간다
            move = np.max([np.maximum(a, b) for a, b in vw], axis=0)
            tris = tris[np.argsort(-move[tris].max(1), kind="stable")]
        for f in range(FRAMES):
            frames.append(rig["deform"](v0, f / FRAMES, H, vw) if vw is not None else rig["deform"](v0, f / FRAMES, H))
            pin_frames.append(rig["marks"](f / FRAMES, H))
    else:
        pins0, pose = rig
        pins_all, followers = sr.add_followers(pins0, v0, comps, tris)
        arap = ARAP2D(pins_all, tris, v0)
    for f in range(FRAMES if not isinstance(rig, dict) else 0):
        p = pose(f / FRAMES, H)
        pin_frames.append(p.copy())
        extra = [pins_all[k] + (p[lead] - pins0[lead]) for k, lead in followers]
        if extra:
            p = np.vstack([p, np.array(extra, np.float32)])
        v1 = arap.solve(p)
        if not np.isfinite(v1).all():
            raise RuntimeError("ARAP 결과에 NaN")
        frames.append(v1.astype(np.float32))
    a = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba = np.dstack([d.crop, a])
    ok, png = cv2.imencode(".png", rgba)

    def b64(arr):
        return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()

    return {
        "w": W, "h": H, "pad": int(0.25 * max(H, W)), "frame_ms": FRAME_MS, "frames": FRAMES,
        "mesh": mesh_kind, "n_vertices": len(v0), "n_triangles": len(tris),
        "texture": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
        "uvs": b64((v0 / np.array([W, H], np.float32)).astype(np.float32)),
        "indices": b64(tris.astype(np.uint32)),
        "positions": b64(np.stack(frames)),  # (프레임, 꼭짓점, 2) float32
        "pins": [[r1(x) for x in pf] for pf in pin_frames],
        "wing_mode": used,
        "used_kind": kind,
        "notice": notice,
    }
