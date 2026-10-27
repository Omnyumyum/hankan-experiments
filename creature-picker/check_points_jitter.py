"""네 발 · 긴 몸 점 흔들기 검사: 점이 조금 빗나가도 움직임이 크게 바뀌지 않는가 (날개는 check_wing_points.py).

정답 점(자동 제안 점 중 눈으로 확인한 것)으로 만든 움직임과, 점마다 반경 R 안에서 흔든 점으로 만든 움직임을
꼭짓점별로 비교한다 (같은 그림이면 메시가 같아서 꼭짓점끼리 맞댈 수 있다).
- 어긋남: 프레임 · 꼭짓점별 거리의 상위 5% 평균 ÷ 그림 긴 변. 3% 넘으면 「엉뚱하게 바뀜」
- 늘어남 > 5배 · 뒤집힌 삼각형 > 1% 도 실패
정답 점 확인용: out/jitter_truth.png

    venv/Scripts/python check_points_jitter.py           # R=10, N=8
"""
import argparse
import base64

import cv2
import numpy as np

import check_wings as cw
import rig_types as rt

CASES = [
    ("quadruped", "drawings/animal_dog.png"),
    ("quadruped", "drawings_quad/q_horse_long_legs.png"),
    ("quadruped", "drawings/real_pig.png"),
    ("quadruped", "drawings_quad/q_dachshund_short_legs.png"),
    ("quadruped", "drawings_quad/q_elephant_trunk_to_ground.png"),
    ("quadruped", "drawings_quad/q_tail_outline_only.png"),
    ("long", "drawings/snake_s.png"),
    ("long", "drawings/snake_line.png"),
    ("long", "drawings/worm.png"),
]


def frames(anim):
    n = anim["n_vertices"]
    return np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)


def jitter(pts, rng, r):
    def j(p):
        return None if p is None else (np.asarray(p, np.float32) + rng.uniform(-1, 1, 2) * r).tolist()
    out = {}
    for k, v in pts.items():
        if isinstance(v, list) and v and isinstance(v[0], list):
            out[k] = [j(p) for p in v]
        elif isinstance(v, list):
            out[k] = j(v)
        else:
            out[k] = v
    return out


def draw_truth(d, kind, pts):
    img = d.crop.copy()
    col = {"head": (233, 125, 43), "tail": (214, 79, 163), "feet": (58, 69, 224)}
    for k, v in pts.items():
        vs = v if (isinstance(v, list) and v and isinstance(v[0], list)) else [v]
        for p in vs:
            if p is not None:
                cv2.circle(img, (int(p[0]), int(p[1])), 6, col.get(k, (0, 160, 0)), -1)
    return cv2.resize(img, (220, int(220 * img.shape[0] / img.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=10)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(11)
    tiles, total, passed = [], 0, 0
    print(f"반경 {args.r}px · 그림마다 {args.n}번\n")
    print(f"{'그림':34s} {'종류':10s} 통과   어긋남(최대)  실패 이유")
    for kind, rel in CASES:
        d = rt.Drawing(rt.sr.load_image(rt.AM / rel))
        truth = rt.propose(d)["points"][kind]
        tiles.append(draw_truth(d, kind, truth))
        base = rt.animate(d, kind, truth)
        F0 = frames(base)
        size = max(d.mask.shape)
        ok, worst, why = 0, 0.0, {}
        for _ in range(args.n):
            pts = jitter(truth, rng, args.r)
            try:
                a = rt.animate(d, kind, pts)
            except Exception as e:
                why[f"오류 {e}"] = why.get(f"오류 {e}", 0) + 1
                continue
            dev = np.linalg.norm(frames(a) - F0, axis=2).ravel()
            dev = float(np.sort(dev)[-max(1, len(dev) // 20):].mean()) / size
            worst = max(worst, dev)
            fails = []
            if dev > 0.03:
                fails.append("엉뚱하게 바뀜")
            if cw.stretch(a) > 5:
                fails.append("찢김")
            if cw.flipped(a) > 0.01:
                fails.append("뭉개짐")
            if fails:
                k = ",".join(fails)
                why[k] = why.get(k, 0) + 1
            else:
                ok += 1
        total += args.n
        passed += ok
        print(f"{rel:34s} {kind:10s} {ok}/{args.n}   {worst:6.1%}       {'; '.join(f'{k} {v}' for k, v in why.items())}")
    h = max(t.shape[0] for t in tiles)
    cv2.imwrite(str(cw.HERE / "out/jitter_truth.png"),
                np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]))
    print(f"\n전체 {passed}/{total} ({passed / total:.0%}) → 정답 점 확인: out/jitter_truth.png")


if __name__ == "__main__":
    main()
