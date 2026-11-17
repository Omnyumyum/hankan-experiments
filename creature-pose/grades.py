"""눈으로 매긴 결과(results/grade_raw.txt)와 흔들기 결과(out/jitter_grade.jsonl)를 합친다."""
import json

import numpy as np

V = ["flip", "rot+5", "rot-5", "s0.8", "s1.2", "width"]
LEGS = ["leg_bo", "leg_bi", "leg_fi", "leg_fo"]


def load_grades(path="results/grade_raw.txt"):
    g = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        gid = int(t[0])
        if t[1] == "X":
            g[gid] = {"x": 1, "head": 0, "axis": 0, **{k: "000" for k in LEGS}, "tail": 0, "usable": 0}
        else:
            g[gid] = {"x": 0, "head": int(t[1]), "axis": int(t[2]), **dict(zip(LEGS, t[3:7])),
                      "tail": int(t[7]), "usable": int(t[8])}
    return g


def jitter_feats(rec, tol=0.05):
    """흔들기마다 5% 안에 머문 관절 비율, 그리고 묶음 숫자."""
    out = {}
    fr = []
    for v in V:
        d = rec.get(v)
        f = 0.0 if d is None else float(np.mean(np.array(d) <= tol))
        out["st_" + v] = f
        fr.append(f)
    out["st_mean"] = float(np.mean(fr))
    out["st_min"] = float(np.min(fr))
    out["st_nof"] = sum(rec.get(v) is None for v in V)          # 흔든 뒤 「못 찾음」 수
    return out


def merged():
    picks = json.load(open("out/grade_joints.json"))
    jit = {(r["cat"], r["idx"], r["style"]): r for r in map(json.loads, open("out/jitter_grade.jsonl"))}
    g = load_grades()
    rows = []
    for p in picks:
        if p["gid"] not in g:
            continue
        r = {k: p[k] for k in ("gid", "cat", "idx", "style", "split")}
        r.update(g[p["gid"]])
        r.update(jitter_feats(jit[(p["cat"], p["idx"], p["style"])]))
        rows.append(r)
    return rows
