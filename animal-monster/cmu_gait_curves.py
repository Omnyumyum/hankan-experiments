"""CMU mocap BVH 에서 한 걸음 주기 허벅지 · 무릎 곡선을 뽑는다 → quadruped.py 의 WALK_THIGH · WALK_KNEE.

BVH 는 https://github.com/una-dinosauria/cmu-mocap (CMU Graphics Lab Motion Capture Database, mocap.cs.cmu.edu 를 CGSpeed 가 BVH 로 바꾼 것).
쓴 파일: data/002/02_01.bvh, data/035/35_01.bvh (걷기 평균), data/016/16_35.bvh (뛰기). 받을 때 이 PC 는 curl --ssl-no-revoke 가 필요했다.

    python cmu_gait_curves.py 02_01.bvh 35_01.bvh
"""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def parse(path):
    lines = Path(path).read_text().split("\n")
    joints, stack, order = {}, [], []
    i = 0
    while not lines[i].startswith("MOTION"):
        t = lines[i].split()
        if t and t[0] in ("ROOT", "JOINT"):
            name = t[1]; joints[name] = {"parent": stack[-1] if stack else None}; order.append(name); cur = name
        elif t and t[0] == "End":
            cur = None
        elif t and t[0] == "OFFSET" and cur:
            joints[cur]["offset"] = np.array(list(map(float, t[1:4])))
        elif t and t[0] == "CHANNELS" and cur:
            joints[cur]["ch"] = t[2:]
        elif t and t[0] == "{":
            stack.append(cur)
        elif t and t[0] == "}":
            stack.pop()
        i += 1
    ft = float(lines[i + 2].split()[-1])
    data = np.array([list(map(float, l.split())) for l in lines[i + 3:] if l.strip()])
    return joints, order, data, ft

def fk(joints, order, data):
    pos, rot, col = {}, {}, 0
    n = len(data)
    for j in order:
        ch = joints[j]["ch"]; vals = data[:, col:col + len(ch)]; col += len(ch)
        rch = [c for c in ch if c.endswith("rotation")]
        ang = vals[:, [ch.index(c) for c in rch]]
        r = R.from_euler("".join(c[0] for c in rch), ang, degrees=True)
        p = joints[j]["parent"]
        off = joints[j]["offset"] if "offset" in joints[j] else np.zeros(3)
        if p is None:
            pos[j] = vals[:, [ch.index(c) for c in ("Xposition", "Yposition", "Zposition")]]; rot[j] = r
        else:
            pos[j] = pos[p] + rot[p].apply(np.tile(off, (n, 1))); rot[j] = rot[p] * r
    return pos

def cycle(path, side="Left"):
    joints, order, data, ft = parse(path)
    pos = fk(joints, order, data)
    hip, knee, ank = pos[f"{side}UpLeg"], pos[f"{side}Leg"], pos[f"{side}Foot"]
    root = pos["Hips"]
    up = np.array([0, 1, 0])
    fwd = root[-1] - root[0]; fwd[1] = 0
    if np.linalg.norm(fwd) < 1e-3:  # 제자리로 만든 파일: 골반 좌우 방향에 수직
        right = (pos["RightUpLeg"] - pos["LeftUpLeg"]).mean(0)
        fwd = np.cross(up, right); fwd[1] = 0
    fwd /= np.linalg.norm(fwd)
    def ang(a, b):  # a→b 벡터의 수직 아래 기준 각, 앞이 +
        v = b - a
        return np.degrees(np.arctan2(v @ fwd, -(v @ up)))
    thigh = ang(hip, knee); shank = ang(knee, ank)
    knee_flex = thigh - shank
    if knee_flex.mean() < 0:  # 앞 방향을 거꾸로 잡았다 (무릎은 뒤로만 접힌다)
        thigh, knee_flex = -thigh, -knee_flex
    # 한 걸음 길이: 허벅지 각의 자기상관이 처음 다시 커지는 곳 (0.35~2초). 짧은 클립(뛰기 1.4초)도 된다
    x = thigh - thigh.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = max(1, int(0.35 / ft)), min(len(x) - 1, int(2.0 / ft))
    if hi <= lo:
        return None, None, 0, float("nan")
    P = lo + int(np.argmax(ac[lo:hi]))
    # 포물선으로 칸 사이 값까지
    if 0 < P < len(ac) - 1:
        d = ac[P - 1] - 2 * ac[P] + ac[P + 1]
        P = P + (0.5 * (ac[P - 1] - ac[P + 1]) / d if d else 0)
    # 모든 프레임을 「한 걸음 안 위치」로 접어 50칸 평균. 0 = 허벅지가 가장 앞으로 뻗은 순간
    peak = int(np.argmax(thigh[: int(np.ceil(P)) + 1]))
    ph = ((np.arange(len(x)) - peak) / P) % 1.0
    bins = (ph * 50).astype(int)
    T, K = np.full(50, np.nan), np.full(50, np.nan)
    for i in range(50):
        if (bins == i).any():
            T[i], K[i] = thigh[bins == i].mean(), knee_flex[bins == i].mean()
    ok = ~np.isnan(T)
    idx = np.arange(50)
    T = np.interp(idx, idx[ok], T[ok], period=50)
    K = np.interp(idx, idx[ok], K[ok], period=50)
    return T, K, round(len(x) / P, 1), P * ft

if __name__ == "__main__":
  for f in sys.argv[1:]:
    t, k, n, period = cycle(f)
    print(Path(f).stem, f"걸음 {n}번 분량, 한 걸음 {period:.2f}초 | 허벅지 {t.min():.0f}~{t.max():.0f}° | 무릎 굽힘 {k.min():.0f}~{k.max():.0f}°")
    print("  허벅지:", " ".join(f"{v:.0f}" for v in t[::5]))
    print("  무릎  :", " ".join(f"{v:.0f}" for v in k[::5]))
    np.save(Path(f).with_suffix(".npy"), np.stack([t, k]))
