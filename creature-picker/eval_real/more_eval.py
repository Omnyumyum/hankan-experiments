"""더 많은 동물 · 괴물 그림으로 종류 추측 · 움직임 망가짐을 잰다 (MORE_ANIMALS.md).

    ../venv/bin/python -I eval_real/more_eval.py --src qd --qd 받은폴더 --split train|test|fresh [--anim] [--out 이름]
    ../venv/bin/python -I eval_real/more_eval.py --src tub --tub test.parquet,train0.parquet [--anim] [--out 이름]
--src qd : QuickDraw {category}.ndjson (앞 30MiB). guess_eval.py 와 같은 몫 나누기(seed 11, 카테고리마다 100 · 100 · 100).
--src tub: TU-Berlin PNG (Hugging Face kmewhort/tu-berlin-png 의 parquet). 확인용으로만 쓴다(학습에 안 넣음).
--anim   : 기대 종류로 움직여 늘어남 · 뒤집힘을 잰다. 기대가 「모름」인 카테고리는 AI 추측대로(추측이 None 이면 안 움직임).
결과: eval_real/more_<out>.csv (그림마다 한 줄)
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import check_wings as cw  # noqa: E402
import rig_types as rt  # noqa: E402
from guess_eval import load_split  # noqa: E402
from run_eval import EXPECT, render  # noqa: E402

# 새 카테고리의 기대 종류. 그림을 보기 전에 이름만 보고 정했다. None = 모름(여러 종류가 다 그럴듯하거나 4종 어디에도 안 맞음)
NEW_EXPECT = {
    # 네 발: 옆모습에 다리 넷이 보통인 동물
    "bear": "quadruped", "lion": "quadruped", "elephant": "quadruped", "giraffe": "quadruped", "camel": "quadruped",
    "sheep": "quadruped", "cow": "quadruped", "tiger": "quadruped", "zebra": "quadruped",
    # 긴 몸: 몸을 물결치며 헤엄치는 것
    "fish": "long", "whale": "long", "dolphin": "long", "shark": "long",
    # 날개: 앞모습에 양옆 날개
    "bat": "wings", "mosquito": "wings",
    # 몸통만: 다리 · 날개 없음
    "teapot": "blob", "snail": "blob",
    # 모름: 다리가 많음(게 · 거미 · 개미 · 바닷가재 · 전갈 · 문어), 옆모습 새(날개가 옆에 붙음), 서거나 앉은 모습(토끼 · 캥거루 · 원숭이 …),
    # 두 종류가 섞임(용 = 네 발 + 날개, 악어 = 네 발 + 긴 몸, 인어 = 사람 + 긴 꼬리, 거북 = 몸통 + 지느러미 발)
    "octopus": None, "crab": None, "spider": None, "ant": None, "lobster": None, "scorpion": None,
    "duck": None, "owl": None, "parrot": None, "flamingo": None, "penguin": None, "swan": None,
    "rabbit": None, "kangaroo": None, "monkey": None, "squirrel": None, "teddy-bear": None, "frog": None, "mouse": None,
    "hedgehog": None, "dragon": None, "crocodile": None, "mermaid": None, "sea turtle": None,
}
ALL_EXPECT = {**EXPECT, **NEW_EXPECT}
# TU-Berlin 이름 → 위 이름 (같은 동물만)
TUB_NAME = {"bear (animal)": "bear", "mouse (animal)": "mouse", "flying bird": "bird", "standing bird": None, "seagull": None,
            "pigeon": None, "rooster": None, "panda": None, "teddy-bear": "teddy-bear", "sea turtle": "sea turtle"}


def tub_render(png_bytes, size=400, thick=6):
    """TU-Berlin 그림(1111px, 가는 선)을 QuickDraw 그림과 같은 모양(흰 종이 · 검은 선 6px · 400px, 가운데)으로."""
    g = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    ink = (g < 128).astype(np.uint8)
    ys, xs = np.nonzero(ink)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = (size - 2 * thick - 4) / max(ink.shape)
    r = max(1, int(round(thick / sc / 2)))  # 줄였을 때 선 굵기가 thick 이 되게 미리 굵게
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    h, w = max(1, round(ink.shape[0] * sc)), max(1, round(ink.shape[1] * sc))
    small = cv2.resize(ink.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    img = np.full((size, size), 255, np.float32)
    oy, ox = (size - h) // 2, (size - w) // 2
    img[oy:oy + h, ox:ox + w] = 255 * (1 - np.clip(small * 1.6, 0, 1))
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def load_tub(paths):
    import pyarrow.parquet as pq
    jobs = []
    for p in paths:
        t = pq.read_table(p)
        imgs, labs = t.column("image").to_pylist(), t.column("label").to_pylist()
        for im, lab in zip(imgs, labs):
            jobs.append((lab, im["bytes"]))
    return jobs


def one(job):
    cat, split, k, src, anim_on = job
    expect = ALL_EXPECT.get(cat)
    row = dict(cat=cat, split=split, id=k, expect=expect or "모름", guess="", p="", kind="", used_kind="", wing_mode="",
               stretch="", flipped="", big_stretch="", big_flip="", broken="", n_regions="", n_feet="", no_body="",
               t_sec="", err="")
    t0 = time.time()
    try:
        img = render(src) if isinstance(src, list) else tub_render(src)
        d = rt.Drawing(img)
        pr = rt.propose(d)
        row["guess"] = pr["guess"] or "None"
        x = getattr(d, "_guess_x", None)
        if x is not None:
            row["p"] = round(max(rt.guess_proba(x).values()), 3)
        row["n_regions"] = int(rt.region_labels(d)[0].max())
        row["n_feet"] = d._guess_feat["n_feet"]
        row["no_body"] = int(bool(pr["points"]["quadruped"].get("no_body")))
        kind = expect or pr["guess"]
        if anim_on and kind:
            row["kind"] = kind
            anim = rt.animate(d, kind, pr["points"][kind])
            row["used_kind"] = anim.get("used_kind") or kind
            row["wing_mode"] = anim.get("wing_mode") or ""
            row["stretch"] = round(cw.stretch(anim), 2)
            row["flipped"] = round(cw.flipped(anim), 4)
            row["big_stretch"] = int(row["stretch"] > 5)
            row["big_flip"] = int(row["flipped"] > 0.01)
            row["broken"] = int(row["big_stretch"] or row["big_flip"])
    except Exception as e:
        row["err"] = repr(e)[:160]
    row["t_sec"] = round(time.time() - t0, 2)
    return row


def tub_names():
    """README 의 이름표 (Hugging Face 사본은 parquet 안에 이름이 없다)."""
    import re
    txt = (HERE / "data" / "tub" / "README.md").read_text(encoding="utf-8")
    return {int(a): b for a, b in re.findall(r"'(\d+)': (.+)", txt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["qd", "tub"], default="qd")
    ap.add_argument("--qd")
    ap.add_argument("--tub")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cats", default="new", help="new | old | all | 쉼표로 이름")
    ap.add_argument("--anim", action="store_true")
    ap.add_argument("--out", default="now")
    ap.add_argument("--model", help="다른 추측 모델 파일로 잰다 (채택 안 한 재학습 모델 비교용)")
    a = ap.parse_args()
    if a.model:
        rt.GUESS_MODEL = Path(a.model)
    jobs = []
    if a.src == "qd":
        cats = {"new": list(NEW_EXPECT), "old": list(EXPECT), "all": list(ALL_EXPECT)}.get(a.cats) or a.cats.split(",")
        for cat in cats:
            sp = load_split(Path(a.qd) / f"{cat}.ndjson", 100, 11)
            for s in a.split.split(","):
                jobs += [(cat, s, k, st, a.anim) for k, st in enumerate(sp[s])]
    else:
        names = tub_names()
        for lab, b in load_tub(a.tub.split(",")):
            n = names[lab].strip()
            cat = TUB_NAME.get(n, n)
            if cat in ALL_EXPECT:
                jobs.append((cat, "tub", len(jobs), b, a.anim))
    print(len(jobs), "장", flush=True)
    with Pool() as p:
        rows = p.map(one, jobs, chunksize=2)
    out = HERE / f"more_{a.out}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("→", out)


if __name__ == "__main__":
    main()
