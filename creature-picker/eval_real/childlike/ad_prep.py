"""Amateur Drawings(Hugging Face 사본 keshan/amateur_drawings-controlnet-dataset)의 parquet 에서 잴 그림을 꺼낸다.

    pyarrow 가 있는 파이썬 -I ad_prep.py --pq parquet폴더 --labels ad_labels.csv --n 600 --out 꺼낼폴더
꺼내는 것: 라벨 붙인 그림 전부 + 전체에서 무작위 n 장(seed 7). out/img/<id>.png(종이 사진 원본), out/gt/<id>.png(정답 마스크), out/meta.csv.
pyarrow 는 requirements.txt 에 없다(데이터 꺼낼 때만 씀). 꺼낸 그림은 저장소에 올리지 않는다.
"""
import argparse
import csv
import glob
import os

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = {r["id"]: r["label"] for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
    files = sorted(glob.glob(os.path.join(a.pq, "*.parquet")))
    ids = []
    for f in files:
        col = pq.read_table(f, columns=["original_image"]).column("original_image").to_pylist()
        ids += [(f, i, c["path"].split(".")[0]) for i, c in enumerate(col)]
    rnd = {ids[i][2] for i in np.random.default_rng(7).permutation(len(ids))[:a.n]}
    want = {k for k in lab} | rnd
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "gt"), exist_ok=True)
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["original_image", "segment_image", "caption"])
        for ff, i, k in ids:
            if ff != f or k not in want:
                continue
            open(os.path.join(a.out, "img", k + ".png"), "wb").write(t.column("original_image")[i].as_py()["bytes"])
            open(os.path.join(a.out, "gt", k + ".png"), "wb").write(t.column("segment_image")[i].as_py()["bytes"])
            rows.append(dict(id=k, label=lab.get(k, ""), random=int(k in rnd), caption=t.column("caption")[i].as_py()))
    with open(os.path.join(a.out, "meta.csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(ids), "장 중", len(rows), "장 꺼냄 (라벨", sum(1 for r in rows if r["label"]), "· 무작위", len(rnd), ")")


if __name__ == "__main__":
    main()
