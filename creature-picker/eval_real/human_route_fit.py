"""사람 그림 / 아닌 그림 가르는 기준 고르기 (HUMAN_ROUTE.md). human_route_eval.py 가 모은 raw_*.csv 를 읽는다.

    ../venv/bin/python -I eval_real/human_route_fit.py --hr eval_real/data/hr [--suffix ""]
기준은 학습용(learn)만 보고 고르고, 확인용(check)은 고른 기준으로 마지막에 한 번 잰다.
결과: eval_real/human_route_features.csv (그림마다 신호 · 판정), eval_real/human_route_curve.csv (곡선 표), 화면에 표.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from meta_pose import USED  # noqa: E402

L, R = {"sho": 5, "elb": 7, "wri": 9, "hip": 11, "kne": 13, "ank": 15}, {"sho": 6, "elb": 8, "wri": 10, "hip": 12, "kne": 14, "ank": 16}


def features(df):
    rows = []
    for _, r in df.iterrows():
        f = dict(found=int(r["det1"] >= 0.5))
        if f["found"] and isinstance(r["kp"], str) and r["kp"]:
            k = np.array(r["kp"].split(), float).reshape(17, 3)
            s = k[:, 2]
            u = s[USED]
            x0, y0, x1, y1 = map(int, r["box"].split())
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            p = k[:, :2]
            d = lambda a, b: float(np.linalg.norm(p[a] - p[b]))  # noqa: E731
            sho, hip = (p[5] + p[6]) / 2, (p[11] + p[12]) / 2
            torso = max(1e-3, float(np.linalg.norm(sho - hip)))
            arm = [d(S["sho"], S["elb"]) + d(S["elb"], S["wri"]) for S in (L, R)]
            leg = [d(S["hip"], S["kne"]) + d(S["kne"], S["ank"]) for S in (L, R)]
            diag = float(np.hypot(bw, bh))
            f.update(
                mean17=s.mean(), mean13=u.mean(), min13=u.min(), med13=float(np.median(u)),
                low13_05=int((u < 0.5).sum()), low13_03=int((u < 0.3).sum()),
                mean_legs=s[11:17].mean(), mean_arms=s[5:11].mean(), min_legs=s[11:17].min(), min_arms=s[5:11].min(),
                torso_rel=torso / diag, arm_rel=np.mean(arm) / diag, leg_rel=np.mean(leg) / diag,
                leg_torso=np.mean(leg) / torso, arm_torso=np.mean(arm) / torso,
                arm_sym=abs(arm[0] - arm[1]) / max(1e-3, sum(arm)), leg_sym=abs(leg[0] - leg[1]) / max(1e-3, sum(leg)),
                upright=(hip[1] - sho[1]) / torso,  # 1 = 어깨가 엉덩이 바로 위, 음수 = 거꾸로
                head_up=(sho[1] - p[0][1]) / torso,
                feet_down=((p[15][1] + p[16][1]) / 2 - hip[1]) / torso,
                box_hw=bh / bw, box_area=bw * bh / (r["w"] * r["h"]),
                spread=float(np.std(p[USED, 0]) + np.std(p[USED, 1])) / diag,  # 점이 상자 안에 퍼진 정도
            )
        rows.append(f)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def group(df):
    v = np.where(df["src"] == "qd", np.where(df["variant"] == "orig", "-원본", "-사진"), "")
    return df["label"] + "·" + df["src"] + v


FEATS = ["mean17", "mean13", "min13", "med13", "low13_05", "mean_legs", "min_legs", "mean_arms", "min_arms", "leg_torso", "arm_torso",
         "arm_sym", "leg_sym", "upright", "head_up", "feet_down", "box_hw", "box_area", "mask_box", "kp_in_mask", "spread", "det1", "det2"]


def rates(df, human):
    """human: 사람 길로 보내면 True. 사람(H) 잘못 거름 · 아님(N) 거름을 그룹마다, 그리고 그룹 평균 · 가장 나쁜 H 그룹."""
    out = {}
    for g, sub in df.groupby("grp"):
        out[g] = 1 - human[sub.index].mean()
    hg = [g for g in out if g.startswith("H")]
    ng = [g for g in out if g.startswith("N")]
    out["H 잘못 거름(평균)"] = float(np.mean([out[g] for g in hg]))  # 그룹마다 같은 무게 (Meta 가 많다고 Meta 에 끌려가지 않게)
    out["H 잘못 거름(최악)"] = max(out[g] for g in hg)
    out["N 거름(평균)"] = float(np.mean([out[g] for g in ng]))
    return out


def sweep(df, score, ths):
    base = df["found"] == 1
    return [(t, rates(df, base & (score.fillna(-9) >= t))) for t in ths]


def pick(res, max_h, max_worst):
    """H 잘못 거름 평균 ≤ max_h, 최악 그룹 ≤ max_worst 안에서 N 거름 최대."""
    ok = [(t, r) for t, r in res if r["H 잘못 거름(평균)"] <= max_h and r["H 잘못 거름(최악)"] <= max_worst]
    return max(ok, key=lambda x: (x[1]["N 거름(평균)"], -x[1]["H 잘못 거름(평균)"])) if ok else (None, None)


def fmt(r, groups):
    return " | ".join(f"{100 * r[g]:.1f}%" for g in groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max_h", type=float, default=0.01)
    ap.add_argument("--max_worst", type=float, default=0.03)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    hr = Path(a.hr)
    df = pd.concat([pd.read_csv(hr / f"raw_{s}{a.suffix}.csv", encoding="utf-8-sig") for s in ("meta", "qd", "tub")
                    if (hr / f"raw_{s}{a.suffix}.csv").exists()], ignore_index=True)
    df = features(df)
    df["grp"] = group(df)
    learn = df[(df.split == "learn") & df.label.isin(["H", "N"])]
    print("장수:", df.groupby(["grp", "split"]).size().unstack(1).fillna(0).astype(int).to_string(), sep="\n")
    cols = ["found", "det1", "mean17", "mean13", "min13", "low13_05", "mean_legs", "leg_torso", "arm_sym", "leg_sym", "upright", "box_hw", "mask_box", "kp_in_mask"]
    print("\n학습용 신호 중앙값 (찾은 것만, found 는 평균):")
    med = learn[learn.found == 1].groupby("grp")[cols].median()
    med["found"] = learn.groupby("grp")["found"].mean()
    print(med.round(3).to_string())
    groups = sorted(df.grp.unique())
    curve = []

    # 0. 지금 규칙: 찾기 < 0.3 또는 관절 평균 < 0.35 → 사람 아님 (찾기 0.5 미만은 TorchServe 가 상자를 안 줌)
    cur = {k: (df["found"] == 1) & (df[k].fillna(-9) >= 0.35) for k in ("mean17", "mean13")}
    print("\n[지금 규칙] 그룹마다 「사람 아님」으로 보낸 비율 (H 는 잘못 거름, N · A 는 거름):  " + " | ".join(groups))
    for k, h in cur.items():
        for sp in ("learn", "check"):
            sub = df[df.split == sp]
            print(f"  {k}≥0.35 {sp}: " + fmt(rates(sub, h[sub.index]), [g for g in groups if g in set(sub.grp)]))

    # 1. 문턱 하나
    ths = np.round(np.arange(0.2, 0.95, 0.01), 2)
    best = {}
    print(f"\n[문턱 하나, 학습용] H 잘못 거름 평균 ≤ {a.max_h:.1%} · 최악 ≤ {a.max_worst:.0%} 에서 N 거름 최대")
    for c in ["mean17", "mean13", "med13", "min13", "mean_legs", "min_legs"]:
        res = sweep(learn, learn[c], ths)
        for t, r in res:
            curve.append(dict(rule=c, t=t, **{k: round(v, 4) for k, v in r.items()}))
        t, r = pick(res, a.max_h, a.max_worst)
        best[c] = (t, r)
        if t is not None:
            print(f"  {c} ≥ {t}: H 평균 {r['H 잘못 거름(평균)']:.1%} 최악 {r['H 잘못 거름(최악)']:.1%} · N 거름 {r['N 거름(평균)']:.1%}")

    # 2. 문턱 둘 (mean13 와 다른 신호 하나, 같은 쪽 또는 반대쪽)
    print("\n[문턱 둘, 학습용] mean13 ≥ t1 그리고 (다른 신호 ≥ 또는 ≤ t2)")
    pair = []
    for c in [f for f in FEATS if f != "mean13"]:
        qs = np.unique(np.round(np.nanquantile(learn[c], np.linspace(0.0, 1.0, 41)), 3))
        for sign in (1, -1):
            for t2 in qs:
                extra = (sign * learn[c].fillna(sign * -1e9) >= sign * t2)
                res = sweep(learn, learn["mean13"].where(extra, -9), ths)
                t, r = pick(res, a.max_h, a.max_worst)
                if t is not None:
                    pair.append((r["N 거름(평균)"], c, sign, t2, t, r))
    pair.sort(key=lambda x: -x[0])
    for nr, c, sign, t2, t, r in pair[:8]:
        print(f"  mean13 ≥ {t} 그리고 {c} {'≥' if sign > 0 else '≤'} {t2}: H 평균 {r['H 잘못 거름(평균)']:.1%} 최악 {r['H 잘못 거름(최악)']:.1%} · N 거름 {nr:.1%}")

    # 3. 작은 분류기 (찾은 그림만 학습; 못 찾음은 언제나 「사람 아님」)
    fl = learn[learn.found == 1]
    X = fl[FEATS].fillna(0).values
    y = (fl.label == "H").values.astype(int)
    w = fl.groupby("grp")["grp"].transform(lambda s: 1 / len(s)).values  # 그룹마다 같은 무게
    mu, sd = X.mean(0), X.std(0) + 1e-9
    models = {}
    lr = LogisticRegression(C=0.3, max_iter=2000).fit((X - mu) / sd, y, sample_weight=w)
    models["logistic"] = lambda d: lr.predict_proba((d[FEATS].fillna(0).values - mu) / sd)[:, 1]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X, y, sample_weight=w)
    models["tree3"] = lambda d: tree.predict_proba(d[FEATS].fillna(0).values)[:, 1]
    print("\n[로지스틱 무게 (표준화한 신호, 큰 것부터)]")
    for c, v in sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1]))[:10]:
        print(f"  {c:12s} {v:+.2f}")
    print("[결정나무 깊이 3]\n" + export_text(tree, feature_names=FEATS, decimals=3))
    pths = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for name, f in models.items():
        sc = pd.Series(f(learn), index=learn.index)
        res = sweep(learn, sc, pths)
        for t, r in res:
            curve.append(dict(rule=name, t=t, **{k: round(v, 4) for k, v in r.items()}))
        t, r = pick(res, a.max_h, a.max_worst)
        best[name] = (t, r)
        if t is not None:
            print(f"  {name} ≥ {t}: H 평균 {r['H 잘못 거름(평균)']:.1%} 최악 {r['H 잘못 거름(최악)']:.1%} · N 거름 {r['N 거름(평균)']:.1%}")
    pd.DataFrame(curve).to_csv(HERE / "human_route_curve.csv", index=False, encoding="utf-8-sig")

    # 4. 곡선 표 (학습용, mean13 · logistic)
    print("\n[곡선, 학습용] H 잘못 거름 평균 목표 → N 거름 평균")
    for name in ["mean17", "mean13", "logistic"]:
        res = [x for x in curve if x["rule"] == name]
        line = []
        for mh in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
            ok = [x for x in res if x["H 잘못 거름(평균)"] <= mh]
            bx = max(ok, key=lambda x: x["N 거름(평균)"]) if ok else None
            line.append(f"{mh:.1%}→{bx['N 거름(평균)']:.1%}(t={bx['t']})" if bx else f"{mh:.1%}→-")
        print(f"  {name:9s} " + "  ".join(line))

    # 5. 확인용: 고른 기준을 한 번
    print("\n[확인용] 그룹마다 「사람 아님」으로 보낸 비율:  " + " | ".join(groups))
    chk = df[df.split == "check"]
    out = df.copy()
    for name, (t, r) in best.items():
        if t is None:
            continue
        sc = pd.Series(models[name](df) if name in models else df[name].values, index=df.index)
        human = (df["found"] == 1) & (sc.fillna(-9) >= t)
        out[f"human_{name}"] = human.astype(int)
        out[f"score_{name}"] = sc.round(4)
        rr = rates(chk, human[chk.index])
        print(f"  {name} ≥ {t}: " + fmt(rr, [g for g in groups if g in rr]) + f"  → H 평균 {rr['H 잘못 거름(평균)']:.1%} 최악 {rr['H 잘못 거름(최악)']:.1%} · N 거름 {rr['N 거름(평균)']:.1%}")
    keep = ["src", "cat", "label", "split", "id", "variant", "file", "grp", "det1", "det2", "n_found", "found"] + [f for f in FEATS if f not in ("det1", "det2")]
    keep += [c for c in out.columns if c.startswith(("human_", "score_"))]
    out[keep].round(4).to_csv(HERE / "human_route_features.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    main()
