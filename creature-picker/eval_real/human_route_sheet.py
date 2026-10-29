"""사람 길 가르기 모아 보기 (HUMAN_ROUTE.md). QuickDraw · TU-Berlin(CC BY 4.0) 그림만 그린다 — Meta 그림은 넣지 않는다.

    ../venv/bin/python -I eval_real/human_route_sheet.py --hr eval_real/data/hr --rule logistic --out eval_real/sheet_human_route.png
줄마다 한 묶음: 사람인데 「사람 아님」으로 감 · 아님인데 사람 길로 감(점수 높은 순) · 아님을 잘 거름 · 애매(A). 확인용에서 무작위(seed 3).
그림 위 점 · 선 = Meta 관절 13개(초록 = 점수 0.5 이상, 빨강 = 미만), 노란 상자 = 찾은 사람 상자. 제목: 카테고리 · 변환, j = 관절 평균(17개) · s = 판정 점수.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import human_route_eval as ev  # noqa: E402
import kid_photo as kp  # noqa: E402
from meta_pose import USED  # noqa: E402

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
CELL = 220


def cell(img, raw, title):
    img = img.copy()
    if isinstance(raw["box"], str) and raw["box"]:
        x0, y0, x1, y1 = map(int, raw["box"].split())
        k = np.array(raw["kp"].split(), float).reshape(17, 3)
        k[:, 0] += x0
        k[:, 1] += y0
        t = max(2, img.shape[0] // 170)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 230), t)
        for a, b in BONES:
            cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), (200, 120, 30), t)
        for j in USED:
            cv2.circle(img, tuple(k[j, :2].astype(int)), 2 * t, (40, 170, 40) if k[j, 2] >= 0.5 else (40, 40, 220), -1)
    s = CELL / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((CELL + 34, CELL, 3), 255, np.uint8)
    out[34:34 + img.shape[0], :img.shape[1]] = img
    for i, line in enumerate(title):
        cv2.putText(out, line, (3, 13 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--rule", default="logistic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    hr = Path(a.hr)
    feat = pd.read_csv(HERE / "human_route_features.csv", encoding="utf-8-sig")
    raw = pd.concat([pd.read_csv(hr / f"raw_{s}.csv", encoding="utf-8-sig") for s in ("qd", "tub")], ignore_index=True)
    key = ["src", "cat", "split", "id", "variant"]
    df = feat[feat.src.isin(["qd", "tub"]) & (feat.split == "check")].merge(raw[key + ["box", "kp"]], on=key)
    hum, sc = df[f"human_{a.rule}"], df[f"score_{a.rule}"]
    rng = np.random.default_rng(3)
    rows = [("사람인데 사람 아님", df[(df.label == "H") & (hum == 0)]),
            ("아님인데 사람 길", df[(df.label == "N") & (hum == 1)].assign(_s=sc).sort_values("_s", ascending=False).head(3 * a.n)),
            ("아님을 거름", df[(df.label == "N") & (hum == 0)]),
            ("애매", df[df.label == "A"])]
    jobs = {}
    for meta, img in ev.qd_jobs(hr) + ev.tub_jobs(hr):
        jobs[tuple(meta[k] for k in key)] = img
    lines = []
    for name, sub in rows:
        pick = sub.iloc[rng.permutation(len(sub))[: a.n]] if len(sub) else sub
        cells = []
        for _, r in pick.iterrows():
            img = jobs[tuple(r[k] for k in key)]
            if isinstance(img, tuple):
                img, _ = kp.make(img[1], img[2], img[3])
            t = [f"{r['cat'][:14]} {r['variant'][:6]}", f"j{r.mean17:.2f} s{r[f'score_{a.rule}']:.2f}" if r.found else "not found"]
            cells.append(cell(img, r, t))
        while len(cells) < a.n:
            cells.append(np.full((CELL + 34, CELL, 3), 255, np.uint8))
        print(f"{name}: {len(sub)}장 중 {len(pick)}장 — " + ", ".join(f"{r['cat']}#{r.id}" for _, r in pick.iterrows()))
        lines.append(np.hstack(cells))
    cv2.imwrite(a.out, np.vstack(lines))


if __name__ == "__main__":
    main()
