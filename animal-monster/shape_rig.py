"""무엇을 그렸는지 몰라도 움직이는 3단계 시험판.

마스크 → 뼈대 선(skeletonize) → 끝점 = 튀어나온 곳 → 끝점마다 기준점 3개 → ARAP 로 변형 → GIF.
관절 모델을 쓰지 않는다.

    python shape_rig.py                          # drawings/ 전체
    python shape_rig.py monster_blob worm        # drawings/ 에서 이름으로
    python shape_rig.py 사진.jpg 폴더/ --out 결과   # 아무 그림 파일 · 폴더
"""
import argparse
import heapq
import math
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial import Delaunay
from skimage.morphology import skeletonize

from arap2d import ARAP2D

HERE = Path(__file__).parent
OUT = HERE / "shape"
MAX_SIDE = 400
MAX_INPUT = 1600  # 휴대폰 사진처럼 큰 그림은 먼저 줄인다 (시험 그림 17장은 모두 이보다 작다)
FRAMES = 32
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image(path: Path):
    """한글 경로 · 투명 PNG · 흑백 · 16비트도 BGR 8비트로 읽는다. 투명한 곳은 흰 종이로 본다."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("그림 파일로 읽을 수 없다")
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[..., 3:].astype(np.float32) / 255
        img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    s = MAX_INPUT / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


# ---------- 1. 자르기 · 마스크 ----------
def is_straight(lab, k):
    """종이 가장자리처럼 곧은 선인가 (가장 작게 감싸는 사각형이 아주 가늘다)."""
    pts = np.argwhere(lab == k).astype(np.float32)
    _, (a, b), _ = cv2.minAreaRect(pts)
    return min(a, b) < 0.08 * max(a, b, 1)


def crop_and_mask(img, notes):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 종이 가장자리 · 책상 그림자처럼 사진 테두리에 닿는 선은 그림이 아니다
    # 단, 닿는 선이 안 닿는 선보다 많으면 그림이 종이를 꽉 채운 것이다. 그때는 곧은 선만 뺀다.
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    area = st[:, cv2.CC_STAT_AREA]
    edge = {k for k in range(1, n) if area[k] >= 30 and (st[k, 0] <= 1 or st[k, 1] <= 1 or st[k, 0] + st[k, 2] >= w - 1 or st[k, 1] + st[k, 3] >= h - 1)}
    inner = sum(area[k] for k in range(1, n) if area[k] >= 30 and k not in edge)
    full = inner < sum(area[k] for k in edge)
    if full:
        notes.append("그림이 사진 가장자리에 닿음 → 곧은 선만 뺌")
    for k in range(1, n):
        if area[k] < 30 or (k in edge and (not full or is_straight(lab, k))):
            ink[lab == k] = 0
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        raise ValueError("그림 선을 찾지 못했다 (너무 연하거나 빈 그림)")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = img[y0:y1, x0:x1]
    s = MAX_SIDE / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)

    # Meta segment() 와 같은 생각: 선을 굵게 이어 붙이고, 바깥에서 flood fill 해서 안을 채운다
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    raw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H0, W0 = raw.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw)
    for k in range(1, n):
        x, y, cw, ch = st[k, :4]
        if (x <= 1 or y <= 1 or x + cw >= W0 - 1 or y + ch >= H0 - 1) and (cw > 0.8 * W0 or ch > 0.8 * H0):
            if not full or is_straight(lab, k):
                raw[lab == k] = 0
    H, W = raw.shape
    # 윤곽이 끊겨 있으면 안이 안 채워진다. 채워질 때까지 틈 메우기를 키운다.
    for k in (5, 9, 13, 17):
        ink = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=2)
        ff = np.pad(ink, 1)
        fmask = np.zeros((H + 4, W + 4), np.uint8)
        cv2.floodFill(ff, fmask, (0, 0), 128)
        filled = np.where(ff[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)
        if (filled > 0).sum() > 1.8 * (ink > 0).sum():
            break
    # 색칠한 그림은 얇은 윤곽이 끊겨도 칠한 곳이 몸이다. 종이와 색(밝기 말고 a·b)이 다르고 이미 잡은 몸에 붙은 곳을 더한다.
    # 밝기를 안 보니 회색 그림자는 안 들어온다.
    lab_img = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    outside = cv2.dilate(filled, np.ones((9, 9), np.uint8)) == 0
    if outside.sum() > 100:
        paper = np.median(lab_img[outside], axis=0)
        colored = (np.hypot(lab_img[..., 1] - paper[1], lab_img[..., 2] - paper[2]) > 20).astype(np.uint8) * 255
        colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(colored)
        near = cv2.dilate(filled, np.ones((5, 5), np.uint8)) > 0
        add = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > 0.002 * H * W and near[lab == k].any()]
        new = np.isin(lab, add) & (filled == 0)
        if new.sum() > 0.005 * H * W:
            filled[new] = 255
            ff = np.pad(filled, 1)  # 더한 뒤 막힌 안쪽(눈 둘레 등)도 채운다
            cv2.floodFill(ff, np.zeros((H + 4, W + 4), np.uint8), (0, 0), 128)
            filled = np.where(ff[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)
            notes.append("색칠한 곳을 몸에 더함")
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + np.argmax(areas)
    # 몸에서 떨어져 그린 조각(다리, 더듬이)도 버리지 않는다.
    # 그림(alpha)에는 조각 그대로, 메시에는 가장 가까운 곳을 잇는 다리를 놓는다.
    alpha = np.zeros_like(filled)
    mesh = np.where(lab == big, 255, 0).astype(np.uint8)
    main_pts = np.argwhere(mesh > 0)
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
    return crop, mesh, alpha


# ---------- 2. 뼈대 선 · 끝점 · 팔다리 ----------
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def find_limbs(mask):
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skel = skeletonize(mask > 0)
    H, W = mask.shape
    pts = set(zip(*np.nonzero(skel)))
    cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
    # 몸 중심에서 가장 가까운 뼈대 점
    arr = np.array(list(pts))
    start = tuple(arr[np.argmin(((arr - [cy, cx]) ** 2).sum(1))])
    # 뼈대 위 최단 경로
    dist = {start: 0.0}
    par = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, p = heapq.heappop(pq)
        if d > dist[p]:
            continue
        for dy, dx in NB:
            q = (p[0] + dy, p[1] + dx)
            if q in pts:
                nd = d + math.hypot(dy, dx)
                if nd < dist.get(q, 1e9):
                    dist[q] = nd
                    par[q] = p
                    heapq.heappush(pq, (nd, q))
    ends = [p for p in pts if p in dist and sum((p[0] + dy, p[1] + dx) in pts for dy, dx in NB) == 1]
    size = max(H, W)
    dtmax = dt.max()

    limbs = []
    for e in ends:
        path = [e]
        while par[path[-1]] is not None:
            path.append(par[path[-1]])
        # 끝에서 몸 쪽으로 걸어가며 굵기가 확 커지는 곳을 뿌리로 본다
        tip_w = np.median([dt[p] for p in path[: max(3, len(path) // 6)]])
        base_i = len(path) - 1
        for i, p in enumerate(path):
            if i > 3 and (dt[p] > max(2.0 * tip_w, 0.2 * dtmax)):
                base_i = i
                break
        length = sum(math.dist(path[i], path[i + 1]) for i in range(base_i))
        width = 2 * np.mean([dt[p] for p in path[:base_i]]) if base_i else 1
        # 짧거나, 길이에 비해 굵으면(몸통 안 뼈대 선의 끝) 팔다리가 아니다
        if length < 0.12 * size or length < 2.0 * width:
            continue
        limbs.append({"path": path, "base_i": base_i, "len": length, "width": width})

    # 같은 팔다리에서 갈라진 가지(손가락 등)는 가장 긴 것 하나만 남긴다
    limbs.sort(key=lambda l: -l["len"])
    kept, owned = [], set()
    for l in limbs:
        outer = l["path"][: l["base_i"]]
        hit = next((i for i, p in enumerate(outer) if p in owned), None)
        if hit is not None and hit < l["base_i"] * 0.85:
            continue
        kept.append(l)
        owned.update(outer)
    return kept, (cy, cx), dt


def skeleton_paths(skel_pts, start):
    """뼈대 위 최단 거리 · 부모."""
    dist, par, pq = {start: 0.0}, {start: None}, [(0.0, start)]
    while pq:
        d, p = heapq.heappop(pq)
        if d > dist[p]:
            continue
        for dy, dx in NB:
            q = (p[0] + dy, p[1] + dx)
            if q in skel_pts:
                nd = d + math.hypot(dy, dx)
                if nd < dist.get(q, 1e9):
                    dist[q], par[q] = nd, p
                    heapq.heappush(pq, (nd, q))
    return dist, par


def find_body_axis(mask, ink):
    """뱀 · 지렁이처럼 몸 전체가 길쭉한 관인지 본다 (2026-10-19).
    뼈대에서 가장 긴 길(지름)을 찾고, 그 길이 굵기에 비해 길며 굵기가 고르고 그림 대부분을 덮으면 몸 축으로 쓴다.
    반환: 꼬리 → 머리 순서의 (y, x) 점 배열, 판정 수치. 긴 몸이 아니면 점 배열은 None."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    pts = set(zip(*np.nonzero(skeletonize(mask > 0))))
    if len(pts) < 10:
        return None, {}
    any_p = next(iter(pts))
    d0, _ = skeleton_paths(pts, any_p)
    a = max(d0, key=d0.get)
    d1, par = skeleton_paths(pts, a)
    b = max(d1, key=d1.get)
    path = [b]
    while par[path[-1]] is not None:
        path.append(par[path[-1]])
    path = np.array(path, float)
    L = d1[b]
    wd = np.array([dt[int(y), int(x)] for y, x in path])
    inner = wd[len(wd) // 10: len(wd) - len(wd) // 10] if len(wd) > 20 else wd
    med, mx = float(np.median(inner)), float(np.percentile(inner, 95))
    # 축에서 굵기의 2배 안에 드는 그림 비율 (몸통 덩어리가 따로 있으면 낮다)
    line_img = np.full(mask.shape, 255, np.uint8)
    for y, x in path.astype(int):
        line_img[y, x] = 0
    near = cv2.distanceTransform(line_img, cv2.DIST_L2, 5) <= 2.0 * mx + 2
    cover = float((near & (mask > 0)).sum() / max(1, (mask > 0).sum()))
    m = {"elong": round(L / (2 * med + 1e-6), 1), "w_ratio": round(mx / (med + 1e-6), 2), "cover": round(cover, 2)}
    if not (m["elong"] >= 6 and m["w_ratio"] <= 1.5 and m["cover"] >= 0.9):
        return None, m
    # 픽셀 계단을 고르게 펴 둔다
    k = max(3, int(med))
    pad = np.pad(path, ((k, k), (0, 0)), mode="edge")
    path = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in (0, 1)], 1)
    # 머리 쪽 고르기: 끝 둘레에 눈 · 입 · 혀처럼 「몸만 있으면 없을 선」이 더 많은 끝, 비슷하면 더 굵은 끝.
    # 몸만 있으면 그려질 선은 두 가지다 — 테두리로 그린 몸이면 마스크 가장자리, 한 줄로 그은 몸이면 축 바로 위.
    on_axis = cv2.distanceTransform(line_img, cv2.DIST_L2, 5)
    stroke = (ink > 0).sum() > 0.45 * (mask > 0).sum()
    plain = on_axis <= med + 1 if stroke else (dt <= 7) & (mask > 0)
    feat = (ink > 0) & ~plain
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    r = 0.06 * max(mask.shape) if stroke else max(1.6 * med, 0.04 * max(mask.shape))
    m["stroke"] = bool(stroke)

    def score(p):
        return int((feat & ((yy - p[0]) ** 2 + (xx - p[1]) ** 2 <= r ** 2)).sum())

    s0, s1 = score(path[0]), score(path[-1])
    w0, w1 = wd[: len(wd) // 8].mean(), wd[-(len(wd) // 8):].mean()
    head_last = s1 > s0 if abs(s1 - s0) > 0.12 * max(s0, s1) else w1 >= w0
    if not head_last:
        path = path[::-1]
    m.update(head_ink=max(s0, s1), tail_ink=min(s0, s1))
    return path, m


# ---------- 3. 메시 ----------
def build_grid_mesh(mask, cell=6):
    """마스크를 덮는 격자 메시. 칸에 그림이 조금이라도 있으면 삼각형 2개를 둔다.
    가는 선도 모서리로 이어져서 경첩이 거의 생기지 않는다."""
    H, W = mask.shape
    gh, gw = H // cell + 1, W // cell + 1
    occ = np.zeros((gh, gw), bool)
    ys, xs = np.nonzero(mask)
    occ[ys // cell, xs // cell] = True
    vid = -np.ones((gh + 1, gw + 1), np.int64)
    verts, tris = [], []

    def vtx(r, c):
        if vid[r, c] < 0:
            vid[r, c] = len(verts)
            verts.append((c * cell, r * cell))
        return vid[r, c]

    for r, c in zip(*np.nonzero(occ)):
        a, b, d, e = vtx(r, c), vtx(r, c + 1), vtx(r + 1, c), vtx(r + 1, c + 1)
        tris += [(a, b, e), (a, e, d)]
    v = np.array(verts, np.float32)
    keep = np.array(tris, np.int32)
    parent = list(range(len(keep)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for ti, (a, b, cc) in enumerate(keep):
        for e in ((a, b), (b, cc), (cc, a)):
            e = (min(e), max(e))
            if e in owner:
                ra, rb = find(ti), find(owner[e])
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[e] = ti
    roots = np.array([find(i) for i in range(len(keep))])
    _, tri_comp = np.unique(roots, return_inverse=True)
    comps = [np.unique(keep[tri_comp == c]) for c in range(tri_comp.max() + 1)]
    return v, keep, (comps, tri_comp)


def build_mesh(mask, step=9):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :]
    pts = [tuple(p) for p in c[::3]]
    ys, xs = np.nonzero(cv2.erode(mask, np.ones((5, 5), np.uint8)))
    for x, y in zip(xs, ys):
        if x % step == 0 and y % step == 0:
            pts.append((x, y))
    v = np.unique(np.array(pts, np.float32), axis=0)
    tri = Delaunay(v).simplices
    keep = []
    for t in tri:
        a, b, cc = v[t]
        # 납작한 삼각형은 ARAP 의 모서리별 회전 계산을 망가뜨린다
        area = abs((b[0] - a[0]) * (cc[1] - a[1]) - (cc[0] - a[0]) * (b[1] - a[1])) / 2
        longest = max(np.linalg.norm(b - a), np.linalg.norm(cc - b), np.linalg.norm(a - cc))
        if area < 0.08 * longest ** 2:
            continue
        ok = True
        for wa, wb, wc in ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)):
            x, y = a * wa + b * wb + cc * wc
            if mask[min(int(round(y)), mask.shape[0] - 1), min(int(round(x)), mask.shape[1] - 1)] == 0:
                ok = False
                break
        if ok:
            keep.append(t)
    keep = np.array(keep, np.int32)
    # 삼각형에 안 쓰인 점을 빼고 번호를 다시 매긴다 (남으면 ARAP 행렬이 특이해진다)
    used = np.unique(keep)
    remap = -np.ones(len(v), np.int32)
    remap[used] = np.arange(len(used))
    v, keep = v[used], remap[keep]
    # 모서리를 나눠 갖는 삼각형끼리 덩어리로 묶는다.
    # 꼭짓점 하나로만 붙은 곳은 경첩이라 ARAP 가 마음대로 돌리므로 다른 덩어리로 본다.
    parent = list(range(len(keep)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for ti, (a, b, cc) in enumerate(keep):
        for e in ((a, b), (b, cc), (cc, a)):
            e = (min(e), max(e))
            if e in owner:
                ra, rb = find(ti), find(owner[e])
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[e] = ti
    roots = np.array([find(i) for i in range(len(keep))])
    _, tri_comp = np.unique(roots, return_inverse=True)
    comps = [np.unique(keep[tri_comp == c]) for c in range(tri_comp.max() + 1)]
    return v, keep, (comps, tri_comp)


def add_followers(pins0, v0, comps, tris):
    """ARAP 는 메시 조각마다 기준점이 2개 이상 있어야 풀린다 (1개면 회전이 정해지지 않는다).
    모자란 조각에는 서로 먼 꼭짓점 2개를 기준점으로 두고, 가장 가까운 기준점의 이동을 그대로 따라가게 한다."""
    followers = []
    pins = list(map(tuple, pins0))
    comps, tri_comp = comps
    hw = (int(v0[:, 1].max()) + 2, int(v0[:, 0].max()) + 2)
    lab = -np.ones(hw, np.int32)
    for t, ci in zip(tris, tri_comp):
        cv2.fillConvexPoly(lab, np.round(v0[t]).astype(np.int32), int(ci))
    count = np.zeros(len(comps), np.int32)
    for x, y in pins0:
        ci = lab[min(int(round(y)), hw[0] - 1), min(int(round(x)), hw[1] - 1)]
        if ci >= 0:
            count[ci] += 1
    for ci, c in enumerate(comps):
        if count[ci] >= 2:
            continue
        pts = v0[c]
        a = c[np.argmin(pts[:, 0] + pts[:, 1])]
        b = c[np.argmax(((pts - v0[a]) ** 2).sum(1))]
        for vid in (a, b):
            leader = int(np.argmin(((pins0 - v0[vid]) ** 2).sum(1)))
            followers.append((len(pins), leader))
            pins.append(tuple(v0[vid]))
    return np.array(pins, np.float32), followers


# ---------- 4. 동작 ----------
def rot(p, c, a):
    s, co = math.sin(a), math.cos(a)
    return (c[0] + (p[0] - c[0]) * co - (p[1] - c[1]) * s, c[1] + (p[0] - c[0]) * s + (p[1] - c[1]) * co)


def make_pins(limbs, center, mask):
    """반환: 기준점 xy 배열, 팔다리별 (뿌리, 가운데, 끝) 번호, 종류."""
    cy, cx = center
    pins = [(cx, cy)]
    rigs = []
    for l in limbs:
        p = l["path"]
        b = l["base_i"]
        tip, mid, base = p[0], p[b // 2], p[b]
        idx = len(pins)
        pins += [(base[1], base[0]), (mid[1], mid[0]), (tip[1], tip[0])]
        # 방향은 몸 중심이 아니라 자기 뿌리 기준 (머리가 제일 두꺼운 그림도 있다)
        ang = math.degrees(math.atan2(tip[0] - base[0], tip[1] - base[1]))  # 아래 = 90
        thick = l["width"] > 0.3 * l["len"]  # 머리처럼 굵고 짧은 덩어리
        kind = "thick" if thick else "leg" if 50 <= ang <= 130 else "wiggle"
        rigs.append({"i": idx, "kind": kind, "tipx": tip[1]})
    if not limbs:
        # 튀어나온 곳이 없으면 몸 둘레에 기준점을 두고 통통 · 말랑하게
        inner = np.argwhere(cv2.erode(mask, np.ones((25, 25), np.uint8)) > 0)
        if len(inner) < 4:
            inner = np.argwhere(mask > 0)
        ys, xs = inner[:, 0], inner[:, 1]
        for x, y in [(xs.min(), cy), (xs.max(), cy), (cx, ys.min()), (cx, ys.max())]:
            k = np.argmin((inner[:, 0] - y) ** 2 + (inner[:, 1] - x) ** 2)
            pins.append((inner[k, 1], inner[k, 0]))
    return np.array(pins, np.float32), rigs


def make_body_pins(axis):
    """몸 축을 따라 같은 간격으로 기준점을 둔다 (꼬리 → 머리)."""
    seg = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(np.clip(s[-1] / 28, 7, 14))
    at = np.linspace(0, s[-1], n)
    ys, xs = np.interp(at, s, axis[:, 0]), np.interp(at, s, axis[:, 1])
    return np.stack([xs, ys], 1).astype(np.float32), [{"kind": "body", "n": n}]


def pose_body(pins0, t, amp_deg=16, waves=1.0):
    """꿈틀: 마디마다 방향을 사인 물결만큼 돌리고, 물결은 머리에서 꼬리로 흐른다.
    마디 길이는 그대로라 몸이 늘어나지 않는다. 가운데 점을 기준으로 이어 붙이고 무게중심을 제자리에 둔다."""
    n = len(pins0)
    seg = np.diff(pins0, axis=0)
    s = (np.arange(n - 1) + 0.5) / (n - 1)  # 0 꼬리 · 1 머리
    # 머리 쪽은 조금 덜 흔들어 얼굴이 덜 돈다
    amp = math.radians(amp_deg) * (1 - 0.35 * s)
    d = amp * np.sin(2 * math.pi * (waves * s + t))
    co, si = np.cos(d), np.sin(d)
    new = np.stack([seg[:, 0] * co - seg[:, 1] * si, seg[:, 0] * si + seg[:, 1] * co], 1)
    mid = n // 2
    p = np.zeros_like(pins0)
    p[mid] = pins0[mid]
    for i in range(mid + 1, n):
        p[i] = p[i - 1] + new[i - 1]
    for i in range(mid - 1, -1, -1):
        p[i] = p[i + 1] - new[i]
    p += pins0.mean(0) - p.mean(0)
    return p


def pose(pins0, rigs, t, H, has_limbs):
    if rigs and rigs[0]["kind"] == "body":
        return pose_body(pins0, t)
    p = pins0.copy()
    w = 2 * math.pi * t
    legs = sorted([r for r in rigs if r["kind"] == "leg"], key=lambda r: r["tipx"])
    for k, r in enumerate(legs):
        r["phase"] = math.pi * (k % 2)
    others = [r for r in rigs if r["kind"] != "leg"]
    for k, r in enumerate(others):
        r["phase"] = k * 1.3
    for r in rigs:
        i = r["i"]
        amp = math.radians({"leg": 16, "wiggle": 22, "thick": 7}[r["kind"]])
        a = amp * math.sin(w + r["phase"])
        base = tuple(p[i])
        p[i + 1] = rot(p[i + 1], base, a)
        p[i + 2] = rot(p[i + 2], base, a)
        if r["kind"] != "thick":
            p[i + 2] = rot(p[i + 2], tuple(p[i + 1]), a * 0.6)  # 끝이 한 번 더 휘게
    if has_limbs:
        p[:, 1] -= 0.015 * H * abs(math.sin(w))  # 통통
    else:
        # 말랑: 바닥 기준으로 세로 늘이고 가로 줄이기
        bottom = p[:, 1].max()
        sq = 0.10 * math.sin(w)
        p[:, 1] = bottom - (bottom - p[:, 1]) * (1 + sq)
        p[:, 0] = p[0, 0] + (p[:, 0] - p[0, 0]) * (1 - sq * 0.6)
        p[:, 1] -= 0.03 * H * max(0, math.sin(w))
    return p


# ---------- 5. 그리기 ----------
def warp(tex, alpha, v0, v1, tris, canvas_hw, off):
    Hc, Wc = canvas_hw
    out = np.full((Hc, Wc, 3), 255, np.float32)
    acc = np.zeros((Hc, Wc, 3), np.float32)
    am = np.zeros((Hc, Wc), np.float32)
    for t in tris:
        s = v0[t].astype(np.float32)
        d = (v1[t] + off).astype(np.float32)
        x, y, w, h = cv2.boundingRect(d)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > Wc or y + h > Hc:
            continue
        M = cv2.getAffineTransform(s, (d - [x, y]).astype(np.float32))
        patch = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tm = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(tm, np.round(d - [x, y]).astype(np.int32), 1, cv2.LINE_AA)
        a = cv2.warpAffine(alpha, M, (w, h)) * tm
        region = am[y:y + h, x:x + w]
        put = (a > region)
        acc[y:y + h, x:x + w][put] = patch[put]
        region[put] = a[put]
    out = acc * am[..., None] + out * (1 - am[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def run(path: Path):
    t0 = time.time()
    img = load_image(path)
    notes = []
    crop, mask, ink_alpha = crop_and_mask(img, notes)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
    axis, axis_m = find_body_axis(mask, ink)
    v0, tris, comps = build_grid_mesh(mask)
    if axis is not None:
        limbs, center = [], None
        pins0, rigs = make_body_pins(axis)
    else:
        limbs, center, _ = find_limbs(mask)
        pins0, rigs = make_pins(limbs, center, mask)
    pins_all, followers = add_followers(pins0, v0, comps, tris)
    arap = ARAP2D(pins_all, tris, v0)
    H, W = mask.shape
    pad = int(0.25 * max(H, W))
    off = np.array([pad, pad], np.float32)
    tex = crop.astype(np.float32)
    alpha = (cv2.GaussianBlur(ink_alpha, (3, 3), 0) / 255.0).astype(np.float32)
    frames = []
    for f in range(FRAMES):
        p = pose(pins0, rigs, f / FRAMES, H, bool(limbs))
        extra = [pins_all[k] + (p[lead] - pins0[lead]) for k, lead in followers]
        if extra:
            p = np.vstack([p, np.array(extra, np.float32)])
        v1 = arap.solve(p)
        if not np.isfinite(v1).all():
            raise RuntimeError("ARAP 결과에 NaN")
        frames.append(cv2.cvtColor(warp(tex, alpha, v0, v1, tris, (H + 2 * pad, W + 2 * pad), off), cv2.COLOR_BGR2RGB))
    imageio.mimsave(OUT / f"{path.stem}.gif", frames, duration=60, loop=0)

    # 확인용: 마스크 윤곽 · 팔다리 · 기준점
    vis = crop.copy()
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (180, 180, 180), 1)
    for k, l in enumerate(limbs):
        col = (0, 0, 255) if rigs[k]["kind"] == "leg" else (255, 0, 160)
        for p in l["path"][: l["base_i"]]:
            vis[p] = col
    if axis is not None:
        cv2.polylines(vis, [axis[:, ::-1].astype(np.int32)], False, (255, 0, 160), 1)
        hx, hy = pins_all[rigs[0]["n"] - 1]
        cv2.circle(vis, (int(hx), int(hy)), 9, (0, 0, 255), 2)  # 머리로 고른 끝
    for x, y in pins_all:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 170, 0), -1)
    cv2.imwrite(str(OUT / f"{path.stem}_rig.png"), vis)
    kinds = [r["kind"] for r in rigs]
    return {"name": path.stem, "notes": notes, "mode": "body" if axis is not None else "limbs", "axis": axis_m, "limbs": len(limbs), "legs": kinds.count("leg"), "wiggle": kinds.count("wiggle"), "thick": kinds.count("thick"),
            "tris": len(tris), "pins": len(pins_all), "pieces": len(comps[0]),
            "sec": round(time.time() - t0, 1), "frames": frames}


def collect(targets):
    """파일 · 폴더 · drawings/ 안 이름을 그림 파일 목록으로."""
    files = []
    for t in targets or [str(HERE / "drawings")]:
        p = Path(t)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            files.append(p)
        else:
            hit = [f for f in sorted((HERE / "drawings").iterdir()) if f.stem == t and f.suffix.lower() in IMG_EXT]
            if not hit:
                print(f"찾을 수 없음: {t}")
            files += hit
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="모양 뼈대로 그림을 움직여 GIF 를 만든다")
    ap.add_argument("targets", nargs="*", help="그림 파일, 폴더, 또는 drawings/ 안 이름 (없으면 drawings/ 전체)")
    ap.add_argument("--out", default=str(OUT), help="결과 폴더 (기본: shape/)")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    tiles, failed = [], []
    for f in collect(args.targets):
        try:
            r = run(f)
        except Exception as e:  # 한 장이 안 돼도 나머지는 돌린다
            failed.append(f"{f.name}: {e}")
            print(f"실패 {f.name}: {e}")
            continue
        frames = r.pop("frames")
        print(r)
        rig = cv2.imread(str(OUT / f"{f.stem}_rig.png"))
        row = [rig] + [cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR) for i in (0, 8, 24)]
        row = [cv2.resize(x, (int(x.shape[1] * 200 / x.shape[0]), 200)) for x in row]
        strip = np.hstack(row)
        strip = cv2.copyMakeBorder(strip, 22, 4, 0, max(0, 1100 - strip.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :1100]
        cv2.putText(strip, f"{r['name']}  [{r['mode']}] limbs={r['limbs']} (leg {r['legs']}, wiggle {r['wiggle']}, thick {r['thick']})  {r['sec']}s" + ("  edge!" if r["notes"] else ""), (4, 16), 0, 0.5, (0, 0, 0), 1)
        tiles.append(strip)
    if tiles:
        cv2.imwrite(str(OUT / "shape_sheet.png"), np.vstack(tiles))
    print(f"\n{len(tiles)}장 완료, {len(failed)}장 실패 → {OUT.resolve()}  (그림마다 .gif · _rig.png, 모아 보기 shape_sheet.png)")
    for m in failed:
        print("  실패", m)
