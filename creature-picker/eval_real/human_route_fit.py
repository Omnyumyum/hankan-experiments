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


def rates(df, human):
    """human: 사람 길로 보내면 True. 돌려줌: 사람(H) 잘못 거름 비율, 아님(N) 거른 비율 (그룹별 · 전체)."""
    out = {}
    for g, sub in df.groupby("grp"):
        h = human[sub.index]
        out[g] = (1 - h.mean()) if g.startswith("H") else (1 - h.mean())
    H = df["label"] == "H"
    N = df["label"] == "N"
    # 사람 잘못 거름: 출처마다 같은 무게(Meta 가 많다고 Meta 에 끌려가지 않게)
    hsrc = df[H].groupby("src").apply(lambda s: 1 - human[s.index].mean(), include_groups=False)
    out["H 잘못 거름(출처 평균)"] = hsrc.mean()
    out["H 잘못 거름(가장 나쁜 출처)"] = hsrc.max()
    out["N 거름"] = 1 - human[N].mean()
    return out


def sweep_one(df, col, ths, extra=None):
    res = []
    base = df["found"] == 1
    for t in ths:
        human = base & (df[col].fillna(-1) >= t)
        if extra is not None:
            human &= extra
        res.append((t, rates(df, human)))
    return res


def pick(res, max_h=0.01, max_h_worst=0.03):
    """사람 잘못 거름(출처 평균) ≤ max_h, 가장 나쁜 출처 ≤ max_h_worst 안에서 N 거름 최대."""
    ok = [(t, r) for t, r in res if r["H 잘못 거름(출처 평균)"] <= max_h and r["H 잘못 거름(가장 나쁜 출처)"] <= max_h_worst]
    return max(ok, key=lambda x: (x[1]["N 거름"], -x[1]["H 잘못 거름(출처 평균)"])) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max_h", type=float, default=0.01)
    a = ap.parse_args()
    hr = Path(a.hr)
    df = pd.concat([pd.read_csv(hr / f"raw_{s}{a.suffix}.csv", encoding="utf-8-sig") for s in ("meta", "qd", "tub")
                    if (hr / f"raw_{s}{a.suffix}.csv").exists()], ignore_index=True)
    df = features(df)
    df["grp"] = group(df)
    learn, check = df[df.split == "learn"], df[df.split == "check"]
    print("장수:", df.groupby(["split", "grp"]).size().unstack(0).fillna(0).astype(int).to_string(), sep="\n")
    print("\n학습용 신호 중앙값:")
    cols = ["found", "det1", "mean17", "mean13", "min13", "low13_05", "leg_torso", "arm_sym", "leg_sym", "upright", "box_hw", "mask_box", "kp_in_mask", "spread"]
    print(learn.groupby("grp")[cols].median().round(3).to_string())
    return df, learn, check


if __name__ == "__main__":
    main()
