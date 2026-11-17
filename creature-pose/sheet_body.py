"""몸 전체 옆모습 매기기용 표본 판. 종류마다 앞 3000장에서 40장을 고정 씨앗으로 뽑는다.
앞 20장(0~19) = 볼 것, 뒤 20장(20~39) = 확인용. 출력: out/body_sheets/{cat}.png, out/body_sample.csv"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import qd

OUT = Path("out/body_sheets"); OUT.mkdir(parents=True, exist_ok=True)
rows = []
for cat in qd.CATS:
    ds = qd.load(cat, 3000)
    idx = sorted(random.Random(f"body-{cat}").sample(range(len(ds)), 40), key=lambda i: random.Random(i).random())
    tiles = []
    for k, i in enumerate(idx):
        img, _ = qd.render(ds[i], "thick")
        t = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(k), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(t, (0, 0), (159, 159), (180, 180, 180), 1)
        tiles.append(t)
        rows.append({"cat": cat, "k": k, "idx": i, "key": ds[i]["key"], "split": "see" if k < 20 else "check"})
    grid = np.vstack([np.hstack(tiles[r * 10:(r + 1) * 10]) for r in range(4)])
    cv2.imwrite(str(OUT / f"{cat}.png"), grid)
with open("out/body_sample.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
