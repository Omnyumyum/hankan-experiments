"""네발 동물(옆모습) 전용 걷기. 그림마다 코드를 고치지 않고 돌리는 것이 목표.

마스크 · 메시 · ARAP · 그리기는 shape_rig 것을 쓰고, 부위 찾기와 동작만 따로 한다.
- 다리: 배 아래로 내려와 바닥 근처까지 닿는 덩어리. 아래에서 위로 한 줄씩 잘라 보며 다리가 가장 많이 갈라져 있는 가장 높은 줄 = 배.
  뼈대 선 가지 모양에 기대지 않으니 굵은 다리 · 색칠한 다리 · 비스듬한 다리도 같은 방법으로 잡힌다.
- 머리 · 꼬리: 몸통 가운데에서 몸 안으로 가장 멀리 가는 두 끝. 눈 · 입 같은 안쪽 선이 많은 쪽이 머리 (비슷하면 굵은 쪽).
- 걸음: 네발 걷기 순서(뒷발 → 앞발 → 반대 뒷발 → 반대 앞발, 1/4 박자씩). 앞으로 나가는 다리는 무릎을 접어 발을 든다.
  몸통은 한 걸음에 두 번 살짝 오르내리고, 머리는 끄덕, 꼬리는 흔든다.
네발로 안 보이면(다리 2개 미만 · 7개 이상) shape_rig 일반 방식으로 넘긴다.

    python quadruped.py                        # drawings/ 전체
    python quadruped.py 그림.png 폴더/ --out 결과
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from skimage.graph import MCP_Geometric

import shape_rig as sr
from arap2d import ARAP2D

FRAMES = 32
LEG_SWING = math.radians(20)  # 다리 앞뒤로 젓는 각
KNEE_FOLD = math.radians(25)  # 발 들 때 무릎 접는 각
KNEE_MAX = math.radians(40)
HEAD_NOD = math.radians(4)
TAIL_WAG = math.radians(14)
BOB = 0.012  # 몸통 오르내림 (그림 높이 비율)

# 사람 걸음 곡선 (CMU mocap 02_01 · 35_01 평균, 한 걸음을 50칸, 앞으로 뻗은 순간부터). 옆에서 본 허벅지 각(앞 +) · 무릎 굽힘 각.
# 사람도 네발 짐승도 한 다리는 「천천히 뒤로 밀고(디딤 약 60%) 빠르게 앞으로(뜀 약 40%)」, 무릎은 발 들 때만 잠깐 크게 접는다.
# 모양만 쓰고 크기는 LEG_SWING · KNEE_FOLD 로 맞춘다 (아이 그림 막대 다리는 70° 접으면 꺾여 보인다).
WALK_THIGH = [31, 31, 30, 30, 30, 29, 29, 28, 27, 27, 26, 25, 23, 21, 19, 18, 16, 14, 12, 11, 9, 7, 5, 3, 0, -2, -5, -8, -11, -13,
              -14, -15, -16, -16, -15, -12, -10, -6, -2, 3, 7, 11, 14, 18, 21, 24, 27, 29, 30, 31]
WALK_KNEE = [35, 30, 26, 23, 23, 23, 23, 24, 27, 29, 30, 30, 31, 31, 30, 30, 30, 29, 29, 29, 29, 28, 27, 26, 24, 21, 19, 16, 14, 13,
             14, 15, 17, 21, 26, 33, 39, 46, 53, 60, 65, 69, 71, 72, 71, 70, 67, 64, 59, 53]
GAIT = "cmu"  # "cmu" 사람 걸음 곡선, "sine" 예전 사인
# 걷기(내장 곡선) 대비 이 동작의 크기 · 빠르기. 뛰기는 더 크게 젓고 무릎을 더 접고 더 빨리
AMP_THIGH, AMP_KNEE, STRIDE_SEC = 1.0, 1.0, 1.07
REF_THIGH_RANGE, REF_KNEE_RANGE, REF_STRIDE = 47.0, 59.0, 1.07  # 내장 걷기 곡선의 허벅지 · 무릎 폭(도), 한 걸음(초)
PATTERN = "walk"  # 다리 박자: "walk" 네발 걷기(1/4 씩), "trot" 속보(대각선 두 다리가 함께)


def use_bvh(path):
    """BVH 동작 파일 하나에서 곡선을 바로 뽑아 쓴다 (사람 그림이 Meta 로 쓰는 것과 같은 파일).
    한 걸음이 0.9초보다 빠르면(뛰기) 속보 박자로."""
    global WALK_THIGH, WALK_KNEE, PATTERN, AMP_THIGH, AMP_KNEE, STRIDE_SEC
    from cmu_gait_curves import cycle
    t, k, n, period = cycle(path)
    if not n or n < 0.8:
        raise SystemExit(f"{path}: 걸음 주기를 못 찾음 (걷기 · 뛰기처럼 되풀이되는 동작만)")
    WALK_THIGH, WALK_KNEE = list(t), list(k)
    AMP_THIGH = float(np.ptp(t)) / REF_THIGH_RANGE
    AMP_KNEE = float(np.ptp(k)) / REF_KNEE_RANGE
    STRIDE_SEC = float(period)
    PATTERN = "trot" if period < 0.9 else "walk"
    return period, PATTERN


def gait_angles(u):
    """한 걸음 안 위치 u(0~1) → (허벅지 각, 무릎 접는 각) 라디안."""
    if GAIT == "sine":
        w = 2 * math.pi * u
        return LEG_SWING * math.sin(w), KNEE_FOLD * max(0.0, math.cos(w))
    th, kn = np.array(WALK_THIGH, float), np.array(WALK_KNEE, float)
    x = np.arange(len(th) + 1) / len(th)
    t = np.interp(u % 1.0, x, np.append(th, th[0])) - th.mean()
    k = np.interp(u % 1.0, x, np.append(kn, kn[0])) - kn.min()
    knee = min(KNEE_FOLD * AMP_KNEE, KNEE_MAX)  # 막대 다리가 꺾여 보이지 않게 상한
    return LEG_SWING * AMP_THIGH * t / np.abs(th - th.mean()).max(), knee * k / (kn.max() - kn.min())


class NotQuadruped(Exception):
    pass


# ---------- 부위 찾기 ----------
def inner_marks(mask, ink):
    """윤곽선과 떨어져 있는 안쪽 선(눈 · 입 · 무늬). 윤곽선 덩어리는 마스크 가장자리 띠에 닿는다."""
    edge = (mask > 0) & (cv2.erode(mask, np.ones((11, 11), np.uint8)) == 0)
    n, lab = cv2.connectedComponents((ink > 0).astype(np.uint8) & (mask > 0))
    touch = np.unique(lab[edge & (lab > 0)])
    return (lab > 0) & ~np.isin(lab, touch)


def open_gaps(crop, mask, alpha, ink):
    """마스크를 만들 때 틈 메우기가 가까이 붙은 다리 사이까지 메운다.
    종이 색이면서 바깥 종이와 이어진 곳은 몸이 아니니 되돌린다. 윤곽 안쪽 흰 몸은 선에 막혀 바깥과 안 이어지니 그대로."""
    lab_img = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    outside = cv2.dilate(mask, np.ones((9, 9), np.uint8)) == 0
    if outside.sum() < 100:
        return mask, alpha
    paper = np.median(lab_img[outside], axis=0)
    like = (np.linalg.norm(lab_img - paper, axis=2) < 18) & (cv2.dilate(ink, np.ones((3, 3), np.uint8)) == 0)
    n, lab = cv2.connectedComponents(like.astype(np.uint8))
    open_ids = np.unique(lab[outside & like])
    gap = np.isin(lab, open_ids[open_ids > 0]) & (mask > 0)
    if gap.sum() < 20:
        return mask, alpha
    gap = cv2.dilate(gap.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # 틈 가장자리 번짐까지
    gap &= cv2.dilate(ink, np.ones((3, 3), np.uint8)) == 0
    mask = np.where(gap, 0, mask).astype(np.uint8)
    alpha = np.where(gap, 0, alpha).astype(np.uint8)
    return mask, alpha


def find_legs(mask, ink):
    """배 높이와 다리 덩어리들.
    1) 아래에서 위로 한 줄씩 잘라 보다 몸통 폭의 35% 를 넘는 덩어리가 나오면 거기가 몸통 밑. 그 위 몇 줄의 가장 넓은 구간 = 몸통 폭.
    2) 그 아래 각 줄에서 다리를 센다: 몸통 폭 아래에서 내려오고, 잘린 줄에서 바닥까지 절반 넘게 내려오고(턱 · 귀는 빠짐),
       눈 · 입 같은 안쪽 선이 없다(땅까지 숙인 머리는 빠짐). 몸통 밖에서 내려온 것(코끼리 코)은 머리 쪽에 남는다.
    3) 다리가 가장 많은 줄 중 가장 높은 줄 = 배. 배 근처에서 붙어 버린 다리도 조금 아래에서는 따로 센다."""
    inner = inner_marks(mask, ink)
    ys, xs = np.nonzero(mask)
    top, ground = ys.min(), ys.max()
    body_w = xs.max() - xs.min() + 1
    y_break = top
    for y in range(ground - 4, top, -2):
        _, _, st, _ = cv2.connectedComponentsWithStats(mask[y:])
        if (st[1:, cv2.CC_STAT_WIDTH] > 0.35 * body_w).any():
            y_break = y
            break
    tx0, tx1 = mask.shape[1], 0
    for r in range(max(top, y_break - int(0.15 * (ground - top))), y_break + 1):
        d = np.diff(np.concatenate([[0], mask[r] > 0, [0]]).astype(np.int8))
        s0, s1 = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        if len(s0):
            k = np.argmax(s1 - s0)
            tx0, tx1 = min(tx0, s0[k]), max(tx1, s1[k])
    # 몸통 바로 밑에서 자른 덩어리마다 눈 · 입이 있는지. 숙인 머리는 낮게 자르면 눈이 잘려 나가도 여기서 걸린다.
    y_hi = min(y_break + 2, ground)
    n_hi, lab_hi = cv2.connectedComponents(mask[y_hi:])
    head_hi = {k for k in range(1, n_hi) if inner[y_hi:][lab_hi == k].sum() >= 12}
    best = (0, None, None)
    for y in range(ground - 4, y_break, -2):
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask[y:])
        legs = []
        for k in range(1, n):
            x, yy, w, h, area = st[k]
            if area < 20 or yy + h < 0.5 * (ground - y):
                continue
            top_x = np.nonzero((lab[:3] == k).any(0))[0]
            cx = top_x.mean() if len(top_x) else cen[k][0]
            if not tx0 - 0.5 * w <= cx <= tx1 + 0.5 * w:
                continue
            if inner[y:][lab == k].sum() >= max(12, 0.005 * area):
                continue
            if head_hi and set(np.unique(lab_hi[y - y_hi:][lab == k])) & head_hi:
                continue
            legs.append(k)
        if len(legs) >= best[0]:  # 같으면 더 높은 줄(더 긴 다리)
            best = (len(legs), y, [lab == k for k in legs])
    _, y_belly, comps = best
    if y_belly is None:
        raise NotQuadruped("다리 0개")
    n = len(comps)
    if not 2 <= n <= 6:
        raise NotQuadruped(f"다리 {n}개")
    if ground - y_belly < 0.1 * (ground - top):
        raise NotQuadruped("다리가 너무 짧음 (서 있는 그림?)")
    legs = []
    for m in comps:
        full = np.zeros_like(mask, bool)
        full[y_belly:] = m
        rows = np.nonzero(full.any(1))[0]
        def centre(r0, r1):
            yy, xx = np.nonzero(full[r0:r1 + 1])
            return np.array([xx.mean(), r0 + yy.mean()], np.float32)
        h = rows[-1] - rows[0] + 1
        width = np.median(full.sum(1)[rows])
        top_c = centre(rows[0], rows[0] + max(1, h // 10))
        root = top_c - [0, 0.6 * width]  # 몸통 안에 경첩
        while root[1] < y_belly and not mask[int(root[1]), int(np.clip(root[0], 0, mask.shape[1] - 1))]:
            root[1] += 1
        legs.append({"root": root, "knee": centre(rows[0] + h // 2 - 1, rows[0] + h // 2 + 1),
                     "foot": centre(rows[-1] - max(1, h // 10), rows[-1]), "pix": full, "len": h})
    return y_belly, legs


def path_to(mcp, end):
    return np.array(mcp.traceback(tuple(end)), np.float32)[:, ::-1]  # (x, y)


def find_head_tail(mask, ink, y_belly, legs):
    body = mask.copy()
    for l in legs:
        body[l["pix"]] = 0
    dt = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    rx = sorted(l["root"][0] for l in legs)
    band = np.zeros_like(body, bool)
    band[:, int(rx[0]):int(rx[-1]) + 1] = True
    c = np.unravel_index(np.argmax(np.where(band, dt, 0) if (band & (body > 0)).any() else dt), dt.shape)
    # 끝 고르기는 몸 안 거리(D), 끝까지 가는 길은 몸 한가운데 능선을 따르게(ridge). 최단 경로는 가장자리에 붙어 굵기를 못 잰다.
    mcp = MCP_Geometric(np.where(body > 0, 1.0, np.inf))
    D, _ = mcp.find_costs([c])
    D = np.where(np.isfinite(D), D, -1)
    ridge = MCP_Geometric(np.where(body > 0, 1.0 / (dt + 1.0), np.inf))
    ridge.find_costs([c])
    xx = np.arange(body.shape[1])[None, :]
    ends = []
    for side in (-1, 1):  # 왼쪽 끝, 오른쪽 끝
        ref = rx[0] if side < 0 else rx[-1]
        cand = np.where(((xx - ref) * side > 0) & (body > 0), D, -1)
        if cand.max() <= 0:
            cand = np.where(((xx - c[1]) * side > 0) & (body > 0), D, -1)
        tip = np.unravel_index(np.argmax(cand), cand.shape)
        ends.append(path_to(ridge, tip))  # 몸통 가운데 → 끝
    # 머리 = 몸통 가운데에서 먼 곳에 안쪽 선(눈 · 입)이 많은 끝. 몸통 무늬(기린 점)는 가운데 가까이라 안 센다. 비슷하면 굵은 끝
    inner = inner_marks(mask, ink) & (body > 0)
    bx0, bx1 = np.nonzero(body.any(0))[0][[0, -1]]
    # 양쪽 끝마다, 안쪽 선 덩어리가 끝까지 거리의 몇 할 지점에 있는지 가장 큰 값. 눈은 끝 가까이, 무늬는 가운데 가까이.
    d_tip = [D[tuple(np.round(e[-1][::-1]).astype(int))] for e in ends]
    score = [0.0, 0.0]
    n_m, lab_m, st_m, cen_m = cv2.connectedComponentsWithStats(inner.astype(np.uint8))
    for k in range(1, n_m):
        if st_m[k, cv2.CC_STAT_AREA] < 6:
            continue
        side = 0 if cen_m[k][0] < c[1] else 1
        dk = D[lab_m == k]
        if (dk > 0).any() and d_tip[side] > 0:
            score[side] = max(score[side], float(dk[dk > 0].mean()) / d_tip[side])
    thick = [dt[tuple(np.round(p[::-1]).astype(int))] for p in (ends[0][len(ends[0]) * 3 // 4], ends[1][len(ends[1]) * 3 // 4])]
    if abs(score[0] - score[1]) > 0.1:
        head_i = int(np.argmax(score))
    else:
        head_i = int(np.argmax(thick))
    head, tail = ends[head_i], ends[1 - head_i]
    forward = -1 if head_i == 0 else 1
    # 꼬리 뿌리: 끝에서 거슬러 오다 굵기가 끝 쪽 굵기의 2배를 넘는 곳
    w = np.array([dt[int(round(y)), int(round(x))] for x, y in tail])
    tip_w = np.median(w[-max(3, len(w) // 4):])
    root_i = 0
    for i in range(len(w) - 1, -1, -1):
        if w[i] > 2.0 * max(tip_w, 2.0):
            root_i = i
            break
    tail_len = len(w) - root_i
    L = int(0.12 * (bx1 - bx0))
    tail_pins = None
    if tail_len > L and len(w) > L and w[-L] < 0.35 * dt[c]:
        tail_pins = [tail[root_i], tail[(root_i + len(w) - 1) // 2], tail[-1]]
    head_pins = [head[int(len(head) * 0.45)], head[int(len(head) * 0.75)], head[-1]]
    return np.array([c[1], c[0]], np.float32), forward, head_pins, tail_pins


def build_rig(mask, ink):
    y_belly, legs = find_legs(mask, ink)
    centre, forward, head_pins, tail_pins = find_head_tail(mask, ink, y_belly, legs)
    # 걸음 박자: 뒷다리 · 앞다리로 나누고, 같은 쪽 안에서 번갈아
    hind = sorted([l for l in legs if (l["root"][0] - centre[0]) * forward < 0], key=lambda l: l["root"][0])
    front = sorted([l for l in legs if (l["root"][0] - centre[0]) * forward >= 0], key=lambda l: l["root"][0])
    for group, start in ((hind, 0.0), (front, 0.25 if PATTERN == "walk" else 0.5)):
        for k, l in enumerate(group):
            l["phase"] = start + 0.5 * (k % 2)  # trot: 뒤 0 · 앞 0.5 / 뒤 0.5 · 앞 0 = 대각선 짝
    if not hind or not front:  # 한쪽에 다 몰리면 번갈아
        for k, l in enumerate(sorted(legs, key=lambda l: l["root"][0])):
            l["phase"] = 0.5 * (k % 2)
    pins, rig = [centre], {"forward": forward, "legs": [], "head": None, "tail": None, "y_belly": y_belly}
    for l in legs:
        rig["legs"].append({"i": len(pins), "phase": l["phase"]})
        pins += [l["root"], l["knee"], l["foot"]]
    rig["head"] = len(pins)
    pins += head_pins
    if tail_pins is not None:
        rig["tail"] = len(pins)
        pins += tail_pins
    return np.array(pins, np.float32), rig


# ---------- 동작 ----------
def pose(pins0, rig, t, H):
    p = pins0.copy()
    f = rig["forward"]
    for leg in rig["legs"]:
        i = leg["i"]
        swing, knee = gait_angles(t + leg["phase"])
        a = -f * swing  # 양수 = 발이 앞으로
        fold = f * knee  # 무릎 접어 발 들기 (발이 뒤로 접힌다)
        root = tuple(p[i])
        p[i + 1] = sr.rot(p[i + 1], root, a)
        p[i + 2] = sr.rot(p[i + 2], root, a)
        p[i + 2] = sr.rot(p[i + 2], tuple(p[i + 1]), fold)
    h = rig["head"]
    nod = HEAD_NOD * math.sin(4 * math.pi * t) * f
    p[h + 1] = sr.rot(p[h + 1], tuple(p[h]), nod)
    p[h + 2] = sr.rot(p[h + 2], tuple(p[h]), nod)
    if rig["tail"] is not None:
        k = rig["tail"]
        a = TAIL_WAG * math.sin(4 * math.pi * t)
        p[k + 1] = sr.rot(p[k + 1], tuple(p[k]), a)
        p[k + 2] = sr.rot(p[k + 2], tuple(p[k]), a)
        p[k + 2] = sr.rot(p[k + 2], tuple(p[k + 1]), a * 0.5)
    # 몸통 오르내림: 발은 땅에 두고 나머지만
    feet = {leg["i"] + 2 for leg in rig["legs"]}
    lift = BOB * max(1.0, AMP_KNEE) * H * abs(math.sin(2 * math.pi * t * 2))
    for j in range(len(p)):
        if j not in feet:
            p[j, 1] -= lift
    return p


# ---------- 돌리기 ----------
def run(path: Path, out: Path):
    t0 = time.time()
    img = sr.load_image(path)
    notes = []
    crop, mask, ink_alpha = sr.crop_and_mask(img, notes)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
    mask, ink_alpha = open_gaps(crop, mask, ink_alpha, ink)
    try:
        n_legs = len(find_legs(mask, ink)[1])
    except NotQuadruped:
        n_legs = 0
    if n_legs < 3 and sr.find_body_axis(mask, ink)[0] is not None:
        raise NotQuadruped("긴 몸 (뱀 · 지렁이)")
    pins0, rig = build_rig(mask, ink)
    v0, tris, comps = sr.build_grid_mesh(mask)
    pins_all, followers = sr.add_followers(pins0, v0, comps, tris)
    arap = ARAP2D(pins_all, tris, v0)
    H, W = mask.shape
    pad = int(0.25 * max(H, W))
    off = np.array([pad, pad], np.float32)
    tex = crop.astype(np.float32)
    alpha = (cv2.GaussianBlur(ink_alpha, (3, 3), 0) / 255.0).astype(np.float32)
    frames = []
    for fi in range(FRAMES):
        p = pose(pins0, rig, fi / FRAMES, H)
        extra = [pins_all[k] + (p[lead] - pins0[lead]) for k, lead in followers]
        if extra:
            p = np.vstack([p, np.array(extra, np.float32)])
        v1 = arap.solve(p)
        if not np.isfinite(v1).all():
            raise RuntimeError("ARAP 결과에 NaN")
        frames.append(cv2.cvtColor(sr.warp(tex, alpha, v0, v1, tris, (H + 2 * pad, W + 2 * pad), off), cv2.COLOR_BGR2RGB))
    imageio.mimsave(out / f"{path.stem}.gif", frames, duration=round(60 * STRIDE_SEC / REF_STRIDE), loop=0)  # 뛰기는 더 빨리

    # 확인용: 배 높이(회색) · 다리(빨강) · 머리(파랑) · 꼬리(보라) · 기준점(초록)
    vis = crop.copy()
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (180, 180, 180), 1)
    cv2.line(vis, (0, rig["y_belly"]), (W - 1, rig["y_belly"]), (160, 160, 160), 1)
    P = lambda j: tuple(int(v) for v in pins0[j])
    for leg in rig["legs"]:
        i = leg["i"]
        cv2.polylines(vis, [np.array([P(i), P(i + 1), P(i + 2)])], False, (0, 0, 255), 2)
        cv2.putText(vis, f"{leg['phase']:.2f}", P(i + 2), 0, 0.35, (0, 0, 200), 1)
    h = rig["head"]
    cv2.polylines(vis, [np.array([P(h), P(h + 1), P(h + 2)])], False, (255, 80, 0), 2)
    if rig["tail"] is not None:
        k = rig["tail"]
        cv2.polylines(vis, [np.array([P(k), P(k + 1), P(k + 2)])], False, (255, 0, 160), 2)
    for x, y in pins_all:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 170, 0), -1)
    cv2.imwrite(str(out / f"{path.stem}_rig.png"), vis)
    return {"name": path.stem, "mode": "quadruped", "notes": notes, "legs": len(rig["legs"]),
            "head": "left" if rig["forward"] < 0 else "right", "tail": rig["tail"] is not None,
            "sec": round(time.time() - t0, 1), "frames": frames}


def sheet_row(out, r, frames):
    rig = cv2.imread(str(out / f"{r['name']}_rig.png"))
    row = [rig] + [cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR) for i in (0, 8, 16, 24)]
    row = [cv2.resize(x, (int(x.shape[1] * 200 / x.shape[0]), 200)) for x in row]
    strip = np.hstack(row)
    strip = cv2.copyMakeBorder(strip, 22, 4, 0, max(0, 1300 - strip.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :1300]
    if r["mode"] == "quadruped":
        label = f"{r['name']}  [quadruped] legs={r['legs']} head={r['head']} tail={'yes' if r['tail'] else 'no'}  {r['sec']}s"
    else:
        label = f"{r['name']}  [fallback: {r['why']}] shape_rig {r['mode']} limbs={r['limbs']}  {r['sec']}s"
    cv2.putText(strip, label + ("  edge!" if any("가장자리" in n for n in r["notes"]) else ""), (4, 16), 0, 0.5, (0, 0, 0), 1)
    return strip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="네발 동물 그림을 걷게 해 GIF 를 만든다 (네발이 아니면 shape_rig 일반 방식)")
    ap.add_argument("targets", nargs="*", help="그림 파일, 폴더, 또는 drawings/ 안 이름 (없으면 drawings/ 전체)")
    ap.add_argument("--out", default=str(sr.HERE / "quadruped"), help="결과 폴더 (기본: quadruped/)")
    ap.add_argument("--gait", choices=["cmu", "sine"], default=GAIT, help="다리 곡선: cmu 사람 걸음 곡선(기본), sine 예전 사인")
    ap.add_argument("--bvh", help="이 BVH 동작에서 곡선을 바로 뽑아 쓴다 (예: 뛰기). 없으면 내장 걷기 곡선")
    args = ap.parse_args()
    GAIT = args.gait
    if args.bvh:
        period, PATTERN = use_bvh(args.bvh)
        print(f"동작 {Path(args.bvh).name}: 한 걸음 {period:.2f}초 → {PATTERN}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr.OUT = out
    tiles, failed = [], []
    for f in sr.collect(args.targets):
        try:
            try:
                r = run(f, out)
            except NotQuadruped as e:
                r = sr.run(f)
                r["why"] = str(e)
        except Exception as e:  # 한 장이 안 돼도 나머지는 돌린다
            failed.append(f"{f.name}: {e}")
            print(f"실패 {f.name}: {e}")
            continue
        frames = r.pop("frames")
        print({k: v for k, v in r.items() if k in ("name", "mode", "why", "legs", "head", "tail", "notes", "sec")})
        tiles.append(sheet_row(out, r, frames))
    if tiles:
        cv2.imwrite(str(out / "sheet.png"), np.vstack(tiles))
    print(f"\n{len(tiles)}장 완료, {len(failed)}장 실패 → {out.resolve()}  (그림마다 .gif · _rig.png, 모아 보기 sheet.png)")
    for m in failed:
        print("  실패", m)
