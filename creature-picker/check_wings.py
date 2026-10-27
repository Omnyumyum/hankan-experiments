"""날개 칸 고르기 경우의 수 자동 검사.

그림마다 정해 둔 칸을 누르고 · 문지른 뒤
- 움직여야 할 점: 무게 ≥ 0.6
- 가만히 있어야 할 점: 무게 ≤ 0.15
- 찢김 지표: 프레임 중 가장 많이 늘어난 메시 모서리 배율 (5 넘으면 실패)
- 뒤집힌 삼각형: 1% 넘으면 실패 (뭉개짐)
- 날개 덩어리 수 · 날개 안 빈 곳
을 보고 통과 · 실패를 표로 낸다. 모아 보기: out/wing_cases.png

    venv/Scripts/python gen_wing_cases.py   # 새 경우 그림 (처음 한 번)
    venv/Scripts/python check_wings.py
"""
from pathlib import Path

import cv2
import numpy as np

import check
import rig_types as rt
from gen_wing_cases import CASES, OUT as CASE_DIR

HERE = Path(__file__).parent
MOVE_MIN, STAY_MAX = 0.6, 0.15
# 날개가 몇 덩어리(뿌리 몇 개)로 나뉘어야 하나 (최소, 최대). 양쪽 날개가 한 덩어리면 한쪽으로 쏠려 접힌다
GROUPS = {"coloring_butterfly_3q": (1, 1), "front_bird": (2, 2), "top_butterfly": (2, 2), "side_bird": (1, 1),
          "lineart_butterfly": (2, 4), "broken_bird": (1, 1), "fairy": (2, 2), "crayon_wings": (2, 2), "dragonfly": (2, 4)}

# 기존 그림: (파일, 좌표계, 누르기, 문지르기, 움직여야 할 점, 가만히 있어야 할 점, 날개짓)
# 좌표계 "crop" = 잘라 낸 그림 좌표(긴 변 400), "orig" = 원래 그림 좌표
EXISTING = {
    "coloring_butterfly_3q": (HERE / "drawings/test_butterfly_coloring.png", "crop", [],
                              # 사람이 날개를 칠하듯 지그재그로 문지른다 (직선 한 줄은 부채꼴 줄무늬를 다 못 지난다)
                              [[(40, 80), (120, 55), (60, 150), (150, 120), (80, 220), (175, 225)], [(175, 40), (205, 130)],
                               [(30, 300), (120, 280), (50, 350), (160, 310)]],
                              [(60, 60), (110, 150), (60, 330), (190, 60)], [(250, 215), (160, 350)], "fold"),
    "front_bird": (HERE / "drawings/wings_bird_front.png", "orig", [(130, 170), (430, 170)], [],
                   [(60, 140), (500, 140)], [(280, 250), (280, 140)], "updown"),
    "top_butterfly": (HERE / "drawings/wings_butterfly.png", "orig", [(165, 170), (355, 170), (190, 285), (330, 285)], [],
                      [(110, 150), (410, 150), (170, 300), (350, 300)], [(260, 230), (260, 112)], "fold"),
    "side_bird": (rt.AM / "drawings/animal_bird.png", "crop", [(150, 75)], [],
                  [(118, 58)], [(270, 80), (200, 170), (70, 160)], "updown"),
}


def ink_bbox(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(ink)
    return xs.min(), ys.min(), xs.max(), ys.max()


def to_crop(d, orig_img):
    """원래 그림 좌표 → 잘라 낸 그림 좌표 (두 그림의 선 테두리 상자를 맞춘다)."""
    ox0, oy0, ox1, oy1 = ink_bbox(orig_img)
    cx0, cy0, cx1, cy1 = ink_bbox(d.crop)
    sx, sy = (cx1 - cx0) / max(1, ox1 - ox0), (cy1 - cy0) / max(1, oy1 - oy0)
    return lambda p: (cx0 + (p[0] - ox0) * sx, cy0 + (p[1] - oy0) * sy)


def weight_at(wm, p):
    x, y = int(round(p[0])), int(round(p[1]))
    return float(wm[max(0, y - 1): y + 2, max(0, x - 1): x + 2].max())


def flipped(anim):
    """프레임 중 뒤집힌(접혀 겹친) 삼각형 비율의 최댓값. 뭉개짐은 늘어남이 아니라 이것으로 나타난다."""
    import base64
    n = anim["n_vertices"]
    pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
    tris = np.frombuffer(base64.b64decode(anim["indices"]), np.uint32).reshape(-1, 3).astype(np.int64)
    uv = np.frombuffer(base64.b64decode(anim["uvs"]), np.float32).reshape(n, 2) * [anim["w"], anim["h"]]

    def area(p):
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])

    s0 = np.sign(area(uv))
    return max(float((np.sign(area(pos[f])) != s0).mean()) for f in range(anim["frames"]))


def stretch(anim):
    n = anim["n_vertices"]
    import base64
    pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
    tris = np.frombuffer(base64.b64decode(anim["indices"]), np.uint32).reshape(-1, 3).astype(np.int64)
    uv = np.frombuffer(base64.b64decode(anim["uvs"]), np.float32).reshape(n, 2) * [anim["w"], anim["h"]]
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    l0 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1) + 1e-3
    worst = 0.0
    for f in range(anim["frames"]):
        l1 = np.linalg.norm(pos[f, e[:, 0]] - pos[f, e[:, 1]], axis=1)
        worst = max(worst, float((l1 / l0).max()))
    return worst


def run_case(name, img_path, coord, taps, swipes, move, stay, style, expect="ok"):
    raw = rt.sr.load_image(img_path)
    d = rt.Drawing(raw)
    f = (lambda p: p) if coord == "crop" else to_crop(d, raw)
    taps_c = [f(p) for p in taps]
    swipes_c = [[f(p) for p in sw] for sw in swipes]
    ids = sorted(({rt.region_at(d, *p) for p in taps_c} | {i for sw in swipes_c for i in rt.regions_along(d, sw)}) - {0})
    res = {"name": name, "ids": len(ids), "fails": [], "stretch": None}
    mask, _ = d.masks("wings")
    pts = {"mode": "regions", "regions": ids, "style": style}
    if not ids:
        res["fails"].append("칸이 하나도 안 잡힘")
        return res, d, None, None, taps_c, swipes_c, f
    try:
        rig = rt.build_wings_regions(d, mask, pts)
        anim = rt.animate(d, "wings", pts)
    except Exception as e:
        res["fails"].append(f"오류 {e}")
        return res, d, None, None, taps_c, swipes_c, f
    wm = rig["wmap"]
    g = len(rig["pivots"])
    lo, hi = GROUPS.get(name.split("/")[0], (1, 99))
    if not lo <= g <= hi:
        res["fails"].append(f"날개 덩어리 {g}개 (기대 {lo}~{hi})")
    # 날개 안 빈 곳: 날개 덩어리마다 구멍을 메운 모양과 비교해 안 골라진 곳이 3% 넘으면 (얼룩덜룩 움직임)
    for gi, comp in enumerate(rt.wing_groups(d, rig["M"], 2 * rt.line_width(rt.region_labels(d)[1]) + 1)):
        H, W = comp.shape
        ff = np.pad(comp.astype(np.uint8) * 255, 1)
        cv2.floodFill(ff, np.zeros((H + 4, W + 4), np.uint8), (0, 0), 128)
        holes = (ff[1:-1, 1:-1] != 128) & ~comp & (mask > 0)
        if holes.sum() > 0.03 * comp.sum():
            res["fails"].append(f"날개 안 빈 곳 {holes.sum() / comp.sum():.0%}")
    for p in move:
        w = weight_at(wm, f(p))
        if w < MOVE_MIN:
            res["fails"].append(f"움직여야 할 점 {p} 무게 {w:.2f}")
    for p in stay:
        w = weight_at(wm, f(p))
        if w > STAY_MAX:
            res["fails"].append(f"가만히 있어야 할 점 {p} 무게 {w:.2f}")
    res["stretch"] = stretch(anim)
    if res["stretch"] > 5:
        res["fails"].append(f"늘어남 {res['stretch']:.1f}배 (찢겨 보임)")
    fl = flipped(anim)
    if fl > 0.01:
        res["fails"].append(f"뒤집힌 삼각형 {fl:.1%} (뭉개짐)")
    return res, d, rig, anim, taps_c, swipes_c, f


def tile(d, rig, anim, taps_c, swipes_c, f, move, stay, res):
    S = 200
    img = d.crop.copy()
    if rig is not None:
        a = (rig["wmap"] * 0.6)[..., None]
        heat = np.zeros_like(img, np.float32)
        heat[..., 2] = 255
        img = (img * (1 - a) + heat * a).astype(np.uint8)
        for p in rig["pivots"]:
            cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 160, 0), -1)
    for sw in swipes_c:
        cv2.polylines(img, [np.array(sw, np.int32)], False, (255, 0, 0), 2)
    for p in taps_c:
        cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
    for p in move:
        cv2.circle(img, tuple(int(v) for v in f(p)), 4, (0, 200, 0), 2)
    for p in stay:
        cv2.drawMarker(img, tuple(int(v) for v in f(p)), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 9, 2)
    tiles = [cv2.resize(img, (S, S))]
    if anim is not None:
        fr = check.frames_of(d, "wings", anim)
        tiles += [cv2.resize(fr[i], (S, S)) for i in (0, 8, 24)]
    row = np.hstack(tiles + [np.full((S, S, 3), 255, np.uint8)] * (4 - len(tiles)))
    ok = "PASS" if not res["fails"] else "FAIL"
    cv2.putText(row, f"{res['name']} {ok} ids={res['ids']} stretch={res['stretch'] and round(res['stretch'], 1)}", (4, 14), 0, 0.45,
                (0, 140, 0) if ok == "PASS" else (0, 0, 220), 1)
    return row


def main():
    rows, n_fail = [], 0
    cases = [(k, *v) for k, v in EXISTING.items()]
    for k, (fn, taps, swipes, move, stay, style, expect) in CASES.items():
        cases.append((k, CASE_DIR / f"{k}.png", "orig", taps, swipes, move, stay, style, expect))
    print(f"{'그림/날개짓':30s} 결과  칸  늘어남  실패 이유")
    # 사용자는 어느 그림에서든 날개짓 방식을 바꿀 수 있으니 두 방식 모두 검사한다
    cases = [(f"{c[0]}/{st}", *c[1:7], st, *c[8:]) for c in cases for st in ("updown", "fold")]
    for name, path, coord, taps, swipes, move, stay, style, *exp in cases:
        res, d, rig, anim, taps_c, swipes_c, f = run_case(name, path, coord, taps, swipes, move, stay, style, *exp)
        ok = not res["fails"]
        n_fail += not ok
        st = f"{res['stretch']:.1f}" if res["stretch"] else "-"
        print(f"{name:30s} {'통과' if ok else '실패'}  {res['ids']:2d}  {st:>5s}  {'; '.join(res['fails'])}")
        rows.append(tile(d, rig, anim, taps_c, swipes_c, f, move, stay, res))
    cv2.imwrite(str(HERE / "out/wing_cases.png"), np.vstack(rows))
    print(f"\n{len(cases) - n_fail}/{len(cases)} 통과 → out/wing_cases.png (초록 동그라미 = 움직여야 할 점, 검은 X = 가만히 있어야 할 점)")


if __name__ == "__main__":
    main()
