"""연한 연필 그림 · 공책 줄 위 그림 사진 정리 (rig_types.clean_photo 앞에 붙는 갈래).

    import faint_ruled as fr
    img2 = fr.clean_photo(img, notes)   # rig_types.clean_photo 와 같은 자리에 쓴다 (크기 · 좌표 그대로)

두 갈래는 해당할 때만 켠다. 아무것도 안 켜지면 rig_types.clean_photo 와 결과가 같다 (같은 함수를 그대로 부른다).
1) 줄 · 모눈 지우기 (rules): 같은 방향의 가는 선이 사진 전체에 고른 간격으로 있으면(공책) 그 선만 종이 색으로 메운다.
   - 방향 · 간격: 긴 변 256px 그림을 -12~12° 돌려 가며 줄마다 「가운데 값」을 모아(그림 선은 한 줄의 일부만 덮어서 공책 줄만 남는다)
     주기를 찾는다. 주기가 세고(RULE_MIN_S) · 진하고(RULE_MIN_AMP) · 고르고(RULE_MIN_EVEN) · 사진 왼쪽 30%와 오른쪽 30%에서도 같은 주기일 때
     (RULE_MIN_SIDE, 가운데 그림의 무늬 · 줄무늬 옷은 여기서 걸러진다) 줄로 본다. 모눈은 다른 방향이 같은 간격이면 느슨하게 받는다.
   - 줄 픽셀: 줄 방향으로 길게 이어진(열기) 가는 선 중, 세로 띠 8개마다 찾은 줄 자리에 있는 것.
     다음은 줄 위에 있어도 남긴다: 같은 줄 · 같은 띠의 줄 진하기(가운데 값)보다 훨씬 진한 곳, 줄 방향에서 30° 넘게 벗어난 방향으로
     줄 굵기보다 길게 이어진 곳(줄을 지나는 그림 선), 줄과 색이 다른 곳(색 펜 · 색연필).
   - 메우기는 cv2.inpaint(Telea): 줄을 가로지르는 그림 선은 위 · 아래 선 색으로 이어진다.
2) 연한 선 진하게 (faint): 종이 밝기(넓은 중앙값)보다 어두운 정도를 짧은 선 모양 필터(8 방향 평균 중 최대)로 거른다.
   점 같은 잡티 · 종이 결은 줄고 곧게 이어진 선은 남는다. 이 값을 「종이 색 − 키운 값」으로 다시 그려 rig_types.clean_photo 에 넘긴다.
   사진 전체에서 가장 진한 선(99.7%)이 FAINT_MAX 보다 연하고, 잡음보다는 뚜렷하고, 색칠이 거의 없을 때만 켠다.
   사진 테두리에 닿는 넓고 어두운 덩어리(책상 · 손가락)는 재기에서 빼고 종이 색으로 지운다.
그다음 rig_types.clean_photo 가 종이 찾기 · 칠한 곳 · 줄 지우기 · 그림 고르기 · 선 진하게를 그대로 한다.
"""
import cv2
import numpy as np

import rig_types as rt

BASE = rt.clean_photo  # 지금 사진 정리 (시험에서 rt.clean_photo 를 이 파일 것으로 바꿔 끼워도 원래 것을 부른다)

WORK = 768  # 이 크기(긴 변)에서 계산하고 결과를 원래 크기로 되돌린다
RULE_MIN_S = 80.0  # 주기 세기(봉우리 힘 ÷ 띠 안 가운데 힘)가 이 이상이고
RULE_MIN_AMP = 0.4  # 그 주기 진폭(작은 그림에서 밝기)이 이 이상이고
RULE_MIN_SIDE = 25.0  # 왼쪽 · 오른쪽 띠에서도 그 주기 세기가 이 이상이고
RULE_MIN_EVEN = 0.7  # 주기마다 봉우리가 있는 칸 비율이 이 이상이면 공책 줄로 본다
RULE_STRONG_S = 1000.0  # 단, 주기 세기가 이 이상이고
RULE_STRONG_AMP = 1.0  # 진폭이 이 이상이면 양옆 · 고름은 안 본다 (기울여 찍은 종이는 사진 일부에만 줄이 있다)
FAINT_MAX = 32.0  # 가장 진한 선(선 필터 99.7%, 테두리의 넓은 어두운 덩어리 빼고)이 종이보다 이만큼 덜 어두우면 연한 그림


def _disk(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _bg_dark(L, size):
    k = max(15, int(0.05 * size) | 1)
    bg = cv2.medianBlur(np.clip(L, 0, 255).astype(np.uint8), k).astype(np.float32)
    return bg, np.maximum(bg - L, 0)


def _period(prof):
    """1차원 모양에서 주기 세기(봉우리 힘 ÷ 띠 안 가운데 힘) · 주기(px) · 고른 정도(주기 칸마다 봉우리가 있는 비율) · 진폭."""
    n = len(prof)
    if n < 48:
        return 0.0, 0.0, 0.0, 0.0
    k = np.ones(41) / 41  # 넓게 흐린 것을 빼서 밝기 변화만 없앤다 (줄 간격보다 넓게)
    p = prof - np.convolve(np.pad(prof, 20, mode="edge"), k, mode="valid")
    p = p - p.mean()
    F = np.abs(np.fft.rfft(p * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n)
    band = (fr > 1 / 45) & (fr < 1 / 3.5)
    if band.sum() < 5 or F[1:].sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0
    i = int(np.argmax(np.where(band, F, 0)))
    s = float(F[i] / max(1e-9, np.median(F[band])))
    per = 1 / fr[i]
    amp = float(2 * np.sqrt(F[i]) / (np.hanning(n).sum()))  # 그 주기 성분의 진폭 (밝기 단위)
    m = int(n // per)
    if m < 4:
        return s, float(per), 0.0, amp
    thr = np.percentile(p, 70)
    even = float(np.mean([p[int(j * per):int((j + 1) * per)].max() > thr for j in range(m)]))
    return s, float(per), even, amp


def _rot(a, ang, flags=cv2.INTER_LINEAR):
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0), M


def _row_median(r, v):
    """줄마다 (사진 안쪽만) 가운데 값. 그림 선은 한 줄의 일부만 덮어서 공책 줄만 남는다."""
    rr = np.where(v > 0, r, np.nan)
    ok = (v > 0).sum(1) > 0.3 * v.shape[1]
    with np.errstate(all="ignore"):
        med = np.nanmedian(rr[ok], axis=1) if ok.any() else np.zeros(0)
    return med, ok


def find_rules(dark):
    """dark: 종이보다 어두운 정도 (작업 크기). 돌아오는 것: 축마다 가장 센 주기 [(축, 각도, 주기 px, 세기, 고름, 진폭, 양옆 세기)]."""
    h, w = dark.shape
    s = 256 / max(h, w)
    sm = cv2.resize(dark, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    one = np.ones_like(sm)
    best = []
    for axis in (0, 1):
        a0 = sm if axis == 0 else sm.T.copy()
        o0 = one if axis == 0 else one.T.copy()
        cand = (0.0, 0.0, 0.0, 0.0, 0.0)
        for ang in np.arange(-12, 12.1, 1.5):
            r, _ = _rot(a0, ang)
            v, _ = _rot(o0, ang, cv2.INTER_NEAREST)
            med, ok = _row_median(r, v)
            sc, per, even, amp = _period(med)
            if sc > cand[0]:
                cand = (sc, per, even, ang, amp)
        if cand[0] > 0:  # 각도 다듬기
            for ang in cand[3] + np.array([-0.75, -0.4, 0.4, 0.75]):
                r, _ = _rot(a0, ang)
                v, _ = _rot(o0, ang, cv2.INTER_NEAREST)
                sc, per, even, amp = _period(_row_median(r, v)[0])
                if sc > cand[0]:
                    cand = (sc, per, even, ang, amp)
        side = 0.0
        if cand[1] > 0:  # 공책 줄은 종이 끝까지 간다: 왼쪽 30% · 오른쪽 30% 따로 봐도 같은 주기가 세야 한다 (가운데 그림의 무늬는 아님)
            r, _ = _rot(a0, cand[3])
            v, _ = _rot(o0, cand[3], cv2.INTER_NEAREST)
            cw = r.shape[1]
            side = min(_power_at(_row_median(r[:, sl], v[:, sl])[0], cand[1]) for sl in (slice(0, int(0.3 * cw)), slice(int(0.7 * cw), cw)))
        best.append((axis, cand[3], cand[1] / s, cand[0], cand[2], cand[4], side))
    return best


def _power_at(prof, per):
    """주기 per 의 힘 ÷ 띠 안 가운데 힘."""
    n = len(prof)
    if n < 48:
        return 0.0
    k = np.ones(41) / 41
    p = prof - np.convolve(np.pad(prof, 20, mode="edge"), k, mode="valid")
    p = (p - p.mean()) * np.hanning(n)
    F = np.abs(np.fft.rfft(p)) ** 2
    fr = np.fft.rfftfreq(n)
    band = (fr > 1 / 45) & (fr < 1 / 3.5)
    i = int(np.argmin(np.abs(fr - 1 / per)))
    return float(F[max(1, i - 1):i + 2].max() / max(1e-9, np.median(F[band])))


def rules_ok(b, size):
    if b[3] >= RULE_STRONG_S and b[5] >= RULE_STRONG_AMP and 4 <= b[2] <= 0.15 * size:
        return True
    return b[3] >= RULE_MIN_S and b[4] >= RULE_MIN_EVEN and b[5] >= RULE_MIN_AMP and b[6] >= RULE_MIN_SIDE and 4 <= b[2] <= 0.15 * size


def rule_mask(dark, axis, ang, per):
    """한 방향 줄 픽셀 (작업 크기 uint8 0/1). 둘째 값: 그림 선을 살리기 전의 줄 띠 (모눈 교차점용)."""
    a0 = dark if axis == 0 else dark.T.copy()
    hh, ww = a0.shape
    r, M = _rot(a0, ang)
    v, _ = _rot(np.ones_like(a0), ang, cv2.INTER_NEAREST)
    lk = max(11, int(0.04 * max(hh, ww)))
    op = cv2.morphologyEx(r, cv2.MORPH_OPEN, np.ones((1, lk), np.uint8))  # 줄 방향으로 길게 이어진 것만
    # 줄을 가로지르는 그림 선: 줄 방향에서 30° 넘게 벗어난 방향으로 줄 굵기보다 길게 이어진다 (줄 자체는 그 방향으로 몇 px 뿐)
    vk = max(7, int(0.35 * per)) | 1
    cross = np.zeros_like(r)
    c = vk // 2
    for deg in (30, 45, 60, 90, 120, 135, 150):
        t = np.radians(deg)
        kern = np.zeros((vk, vk), np.uint8)
        cv2.line(kern, (int(round(c - c * np.cos(t))), int(round(c - c * np.sin(t)))), (int(round(c + c * np.cos(t))), int(round(c + c * np.sin(t)))), 1, 1)
        cross = np.maximum(cross, cv2.morphologyEx(r, cv2.MORPH_OPEN, kern))
    band = np.zeros(r.shape, np.uint8)
    rowd = np.zeros(r.shape, np.float32)
    rdm = np.full(r.shape, np.inf, np.float32)  # 띠의 줄 진하기 (모든 줄 자리의 가운데 값)
    nstrip = 8
    xs = np.linspace(0, ww, nstrip + 1).astype(int)
    rad = max(1, int(round(0.18 * per)))
    for j in range(nstrip):
        sl = slice(xs[j], xs[j + 1])
        med, ok = _row_median(op[:, sl], v[:, sl])
        if len(med) < 8:
            continue
        prof = np.zeros(hh, np.float32)
        prof[ok] = med
        pk = np.where((prof > np.roll(prof, 1)) & (prof >= np.roll(prof, -1)) & (prof > 0))[0]  # 평평한 봉우리(굵은 그림 선)는 한 번만
        if len(pk) < 3:
            continue
        # 띠의 줄 진하기: 줄 개수(높이 ÷ 간격)만큼의 센 봉우리 가운데 값 (잡티 봉우리는 안 섞는다)
        nexp = max(3, int(v[:, sl].any(1).sum() / per))
        rd = max(2.0, float(np.median(np.sort(prof[pk])[-nexp:])))
        pk = pk[prof[pk] > 0.4 * rd]
        rows = np.zeros(hh, bool)
        for y in pk:
            rows[max(0, y - rad):y + rad + 1] = True
        band[:, sl] = rows[:, None] & (op[:, sl] > 0.35 * rd)
        rdm[:, sl] = rd
        # 이 띠 · 이 줄의 줄 진하기 = 줄 픽셀의 가운데 값 (그림 선은 줄 일부만 덮어서 가운데 값에 안 끼어든다)
        rr = np.where(band[:, sl] > 0, r[:, sl], np.nan)
        has = band[:, sl].any(1)
        if has.any():
            with np.errstate(all="ignore"):
                rowd[has, sl] = np.nanmedian(rr[has], axis=1)[:, None]
    # 이 줄만 띠의 다른 줄보다 훨씬 진하면 그림 선이 줄을 따라 띠를 다 덮은 것: 띠의 줄 진하기로 견준다
    rowd = np.where(rowd > 1.8 * rdm + 5, rdm, rowd)
    # 줄보다 훨씬 진한 곳(줄 위에 겹쳐 그린 선)과 비스듬히 이어진 곳(줄을 지나는 선)은 남긴다
    keep = (r > 1.5 * rowd + 8) | (cross > 0.8 * rowd + 2)
    rm = band & ~keep
    rm = cv2.dilate(rm.astype(np.uint8), np.ones((3, 1), np.uint8))
    band = cv2.dilate(band, np.ones((3, 1), np.uint8))
    Mi = cv2.invertAffineTransform(M)
    out = []
    for a in (rm, band):
        b = cv2.warpAffine(a, Mi, (ww, hh), flags=cv2.INTER_NEAREST)
        out.append(b if axis == 0 else b.T.copy())
    return out


def remove_rules(work, L):
    """작업 그림에서 공책 줄 · 모눈을 지운다. 돌아오는 것: (지운 그림, 줄 방향 수)."""
    size = max(L.shape)
    _, dark = _bg_dark(L, size)
    allb = find_rules(dark)
    found = [b for b in allb if rules_ok(b, size)]
    if not found:
        return work, 0
    if len(found) == 1:  # 모눈: 한 방향이 확실하고 다른 방향도 비슷한 간격이면 느슨한 기준으로 받는다 (세로줄은 종이가 휘어 덜 곧다)
        o = [b for b in allb if b[0] != found[0][0]][0]
        if abs(o[2] - found[0][2]) < 0.25 * found[0][2] and o[3] >= 0.5 * RULE_MIN_S and o[5] >= 0.75 * RULE_MIN_AMP and o[6] >= 0.3 * RULE_MIN_SIDE:
            found.append(o)
    m = np.zeros(L.shape, np.uint8)
    bands = []
    for axis, ang, per, *_ in found:
        rm, band = rule_mask(dark, axis, ang, per)
        m |= rm
        bands.append(band)
    if len(bands) == 2:  # 모눈 교차점: 두 방향 줄 띠가 만나는 곳 (서로를 「지나는 선」으로 보고 남기지 않게)
        m |= bands[0] & bands[1]
    # 줄과 색이 다른 곳(색 펜 · 색연필 그림 선)은 줄 위에 있어도 남긴다
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    on = (np.maximum.reduce(bands) > 0) & (dark > 3)
    if on.sum() > 50:
        rc = np.median(lab[on][:, 1:], axis=0)
        m &= ~(np.hypot(lab[..., 1] - rc[0], lab[..., 2] - rc[1]) > 14)
    if m.mean() < 0.005:
        return work, 0
    return cv2.inpaint(work, m * 255, 3, cv2.INPAINT_TELEA), len(found)


def _line_filter(d, ln=9, nang=8):
    """짧은 곧은 선 모양으로 평균한 것 중 가장 큰 값 (선은 커지고 점 같은 잡티는 줄어든다)."""
    out = np.zeros_like(d)
    c = ln // 2
    for i in range(nang):
        t = np.pi * i / nang
        k = np.zeros((ln, ln), np.float32)
        cv2.line(k, (int(round(c - c * np.cos(t))), int(round(c - c * np.sin(t)))), (int(round(c + c * np.cos(t))), int(round(c + c * np.sin(t)))), 1.0, 1)
        out = np.maximum(out, cv2.filter2D(d, -1, k / k.sum()))
    return out


def faint_stats(work, L):
    size = max(L.shape)
    bg, dark = _bg_dark(L, size)
    R = _line_filter(cv2.GaussianBlur(dark, (3, 3), 0))
    # 사진 테두리에 닿는 넓고 어두운 덩어리(책상 · 손가락 · 종이 밖 그림자)는 선이 아니다: 재기에서 빼고 선으로도 안 고른다
    k = max(5, int(0.02 * size) | 1)
    thick = (cv2.morphologyEx(dark, cv2.MORPH_OPEN, _disk(k)) > 12).astype(np.uint8)
    n, lab_t, st, _ = cv2.connectedComponentsWithStats(thick)
    hh, ww = dark.shape
    edge = (st[:, 0] <= 1) | (st[:, 1] <= 1) | (st[:, 0] + st[:, 2] >= ww - 1) | (st[:, 1] + st[:, 3] >= hh - 1)
    edge[0] = False
    blob = cv2.dilate(edge[lab_t].astype(np.uint8), _disk(2 * k + 1)) > 0
    R = np.where(blob, 0, R)
    b = max(4, int(0.03 * size))
    inner = R[b:-b, b:-b][~blob[b:-b, b:-b]]
    if inner.size < 1000:
        inner = R[b:-b, b:-b].ravel()
    noise = 1.48 * float(np.median(np.abs(inner - np.median(inner)))) + 0.3
    top = float(np.percentile(inner, 99.7))
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    ch = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    chb = cv2.medianBlur(np.clip(ch, 0, 255).astype(np.uint8), max(15, int(0.05 * size) | 1)).astype(np.float32)
    color = float(((ch - chb) > 18).mean())
    return dict(bg=bg, dark=dark, R=R, noise=noise, top=top, color=color, blob=blob)


def enhance_faint(work, st):
    """연한 그림: 밝기를 「종이 − 선 필터 값」으로 바꾼다 (rig_types.clean_photo 가 선을 더 잘 찾는다).
    잡음 수준(2σ) 아래는 0, 위는 가장 진한 선이 종이보다 90 어둡게 되도록 키운다.
    사진 테두리에 닿는 넓고 어두운 덩어리(책상 · 손가락)는 종이 색으로 지운다 (그것이 그림으로 골라져 연한 그림이 사라졌다)."""
    R, noise, blob = st["R"], st["noise"], st["blob"]
    size = max(work.shape[:2])
    pcol = cv2.medianBlur(work, max(15, int(0.05 * size) | 1)).astype(np.float32)
    d = np.maximum(R - 2 * noise, 0)
    gain = float(np.clip(90 / max(1.0, st["top"] - 2 * noise), 1, 8))
    tint = work.astype(np.float32) - pcol  # 선 색(연필 회색 · 색연필 색)을 조금 남긴다
    tint = tint / np.maximum(np.abs(tint).max(-1, keepdims=True), 1)
    out = pcol - (gain * d)[..., None] * (1 - 0.3 * np.clip(tint, -1, 0) - 0.3 * np.clip(tint, 0, 1))
    out = np.where(blob[..., None], pcol, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_photo(img, notes=None):
    h, w = img.shape[:2]
    s = min(1.0, WORK / max(h, w))
    work = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    work = rt.paper_only(work)
    L = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    did = []
    work2, nr = remove_rules(work, L)
    if nr:
        did.append("모눈 지움" if nr == 2 else "공책 줄 지움")
        L = cv2.cvtColor(work2, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    st = faint_stats(work2, L)
    if st["top"] < FAINT_MAX and st["top"] > 4 * st["noise"] and st["color"] < 0.02:
        work2 = enhance_faint(work2, st)
        did.append(f"연한 선 진하게 (가장 진한 선 {st['top']:.0f})")
    if not did:
        return BASE(img, notes)
    if notes is not None:
        notes.append("연한 선 · 줄 갈래: " + ", ".join(did))
    out = cv2.resize(work2, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else work2
    return BASE(out, notes)
