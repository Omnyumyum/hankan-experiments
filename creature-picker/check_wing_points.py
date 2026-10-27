"""날개 「점 3개」 흔들기 검사: 점이 조금 빗나가도 결과가 괜찮은가.

그림마다 맞는 점(뿌리 위 · 뿌리 아래 · 날개 끝)을 정해 두고, 점마다 반경 R(잘라 낸 그림 px) 안에서 무작위로 흔들어 N 번 돌린다.
점 3개 그대로(raw)와 칸에 맞춘 것(snap)을 check_wings.py 와 같은 기준으로 판정해 통과율을 낸다.
(움직여야 할 점 ≥ 0.6 · 가만히 있어야 할 점 ≤ 0.15 · 늘어남 ≤ 5배 · 뒤집힌 삼각형 ≤ 1%)

    venv/Scripts/python check_wing_points.py            # R=10, N=8
    venv/Scripts/python check_wing_points.py --r 16 --n 12
"""
import argparse

import numpy as np

import check_wings as cw
import rig_types as rt

# 그림: 날개마다 [뿌리 위, 뿌리 아래, 끝] (좌표계는 check_wings 와 같다), 날개짓
TRUTH = {
    "front_bird": ([[(240, 200), (235, 245), (35, 140)], [(320, 200), (325, 245), (525, 140)]], "updown"),
    "top_butterfly": ([[(244, 130), (244, 305), (70, 150)], [(276, 130), (276, 305), (450, 150)]], "fold"),
    "side_bird": ([[(130, 96), (232, 106), (105, 50)]], "updown"),
    # 비스듬한 나비는 한쪽에 날개가 두 장(앞 · 뒤)이라 날개 끝을 둘 찍는다 (같은 뿌리)
    "coloring_butterfly_3q": ([[(205, 240), (165, 330), (40, 30)], [(205, 240), (165, 330), (25, 340)]], "fold"),
    "lineart_butterfly": ([[(242, 130), (242, 310), (65, 150)], [(278, 130), (278, 310), (455, 150)]], "fold"),
    "fairy": ([[(200, 215), (208, 250), (75, 180)], [(320, 215), (312, 250), (445, 180)]], "updown"),
    "crayon_wings": ([[(240, 140), (240, 250), (70, 175)], [(280, 140), (280, 250), (450, 175)]], "fold"),
    "dragonfly": ([[(266, 105), (266, 180), (55, 140)], [(294, 105), (294, 180), (505, 140)]], "updown"),
}


def case_info(name):
    if name in cw.EXISTING:
        path, coord, _t, _s, move, stay, _st = cw.EXISTING[name]
    else:
        _fn, _t, _s, move, stay, _st, _e = cw.CASES[name]
        path, coord = cw.CASE_DIR / f"{name}.png", "orig"
    return path, coord, move, stay


def judge(d, mask, f, pts, move, stay):
    """반환: (통과 여부, 이유, 쓴 방식)"""
    try:
        rig, used = rt.resolve_wings(d, mask, pts)
        anim = rt.animate(d, "wings", pts)
    except Exception as e:
        return False, f"오류 {e}", "-"
    wm = rig["wmap"]
    why = []
    for p in move:
        if cw.weight_at(wm, f(p)) < cw.MOVE_MIN:
            why.append("날개 안 움직임")
            break
    for p in stay:
        if cw.weight_at(wm, f(p)) > cw.STAY_MAX:
            why.append("몸통 딸려 옴")
            break
    if cw.stretch(anim) > 5:
        why.append("찢김")
    if cw.flipped(anim) > 0.01:
        why.append("뭉개짐")
    return not why, ",".join(why), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=10, help="흔드는 반경 (잘라 낸 그림 px)")
    ap.add_argument("--n", type=int, default=8, help="그림마다 몇 번")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    tot = {"raw": 0, "snap": 0}
    count = 0
    print(f"반경 {args.r}px · 그림마다 {args.n}번 (+ 흔들지 않은 1번)\n")
    print(f"{'그림':24s} {'점 3개 그대로':>14s} {'칸에 맞춤':>10s}  실패 이유 (칸에 맞춤)")
    for name, (wings, style) in TRUTH.items():
        path, coord, move, stay = case_info(name)
        raw_img = rt.sr.load_image(path)
        d = rt.Drawing(raw_img)
        mask, _ = d.masks("wings")
        f = (lambda p: p) if coord == "crop" else cw.to_crop(d, raw_img)
        base = [[np.array(f(p), np.float32) for p in w] for w in wings]
        ok = {"raw": 0, "snap": 0}
        reasons = {}
        modes = {}
        for t in range(args.n + 1):
            jit = [[p + (0 if t == 0 else rng.uniform(-1, 1, 2) * args.r) for p in w] for w in base]
            jit = [[np.clip(p, 0, [mask.shape[1] - 1, mask.shape[0] - 1]).tolist() for p in w] for w in jit]
            for kind, snap in (("raw", False), ("snap", True)):
                pts = {"wings": jit, "style": style, "snap": snap}
                good, why, used = judge(d, mask, f, pts, move, stay)
                ok[kind] += good
                if kind == "snap":
                    modes[used] = modes.get(used, 0) + 1
                    if not good:
                        reasons[why] = reasons.get(why, 0) + 1
        n = args.n + 1
        count += n
        for k in ok:
            tot[k] += ok[k]
        rs = "; ".join(f"{k} {v}" for k, v in reasons.items())
        print(f"{name:24s} {ok['raw']:>8d}/{n:<5d} {ok['snap']:>6d}/{n:<3d}  {rs}  (쓴 방식 {modes})")
    print(f"\n전체: 점 3개 그대로 {tot['raw']}/{count} ({tot['raw'] / count:.0%}) → 칸에 맞춤 {tot['snap']}/{count} ({tot['snap'] / count:.0%})")


if __name__ == "__main__":
    main()
