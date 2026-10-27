"""종류 추측 모델(guess_model.json) 만들기. QuickDraw 학습용 그림(guess_eval.py 의 train 몫)만 쓴다.

    venv/bin/pip install scikit-learn   # 배울 때만 필요 (움직이는 쪽은 numpy 로 읽는다)
    venv/bin/python -I eval_real/train_guess.py --qd 받은폴더 [--n 100] [--seed 11]
기대 종류(카테고리 이름으로 정함)를 맞히도록 작은 결정 나무 80개 × 4종(깊이 3)을 배운다.
학습용 안에서 5번 나눠 맞혀 보기(교차 검증)로 「확률 몇 이상일 때만 추측할지」별 맞음/틀림도 보여 준다.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rig_types as rt  # noqa: E402
from guess_eval import load_split  # noqa: E402
from run_eval import EXPECT, render  # noqa: E402

PARAMS = dict(n_estimators=80, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=0)


def feats(job):
    cat, k, strokes = job
    d = rt.Drawing(render(strokes))
    rt.propose(d)
    return cat, rt.guess_features(d, d._guess_feat)


def main():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--more", action="store_true", help="more_eval.NEW_EXPECT 의 기대 종류가 있는 카테고리 학습용도 넣는다")
    ap.add_argument("--balance", action="store_true", help="종류마다 무게 합이 같게 (네 발 카테고리가 많아서)")
    ap.add_argument("--old-weight", type=float, default=1.0, help="예전 10 카테고리 그림의 무게 (새 카테고리 = 1)")
    ap.add_argument("--depth", type=int, default=PARAMS["max_depth"])
    ap.add_argument("--trees", type=int, default=PARAMS["n_estimators"])
    ap.add_argument("--min-p", type=float, default=rt.GUESS_MIN_P, help="이보다 자신 없으면 모르겠음 (모델 파일에 같이 적는다)")
    ap.add_argument("--cache", help="특징을 이 파일에 저장 · 있으면 다시 씀 (eval_real/data/ 아래 권장)")
    a = ap.parse_args()
    PARAMS.update(max_depth=a.depth, n_estimators=a.trees)
    old_cats = set(EXPECT)
    if a.more:
        from more_eval import NEW_EXPECT
        EXPECT.update({c: k for c, k in NEW_EXPECT.items() if k})
    jobs = [(cat, k, st) for cat in EXPECT for k, st in enumerate(load_split(Path(a.qd) / f"{cat}.ndjson", a.n, a.seed)["train"])]
    if a.cache and Path(a.cache).exists():
        got = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    else:
        with Pool() as p:
            got = p.map(feats, jobs, chunksize=4)
        if a.cache:
            Path(a.cache).write_text(json.dumps(got), encoding="utf-8")
    keys = list(got[0][1])
    X = np.array([[x[k] for k in keys] for _, x in got])
    y = np.array([EXPECT[c] for c, _ in got])
    cats = [c for c, _ in got]
    w = np.ones(len(y))
    if a.balance:
        cnt = {k: (y == k).sum() for k in set(y)}
        w = np.array([len(y) / (len(cnt) * cnt[k]) for k in y])
    w = w * np.array([a.old_weight if c in old_cats else 1.0 for c in cats])
    P = cross_val_predict(GradientBoostingClassifier(**PARAMS), X, y, method="predict_proba", params={"sample_weight": w},
                          cv=StratifiedKFold(5, shuffle=True, random_state=0).split(X, cats))
    cl = np.array(sorted(set(y)))
    pr, conf = cl[P.argmax(1)], P.max(1)
    print("교차 검증 (학습용 안에서): 확률 기준 → 맞음 / 틀림 / 모르겠음")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        ok, bad = int(((pr == y) & (conf >= th)).sum()), int(((pr != y) & (conf >= th)).sum())
        print(f"  {th}: {ok} / {bad} / {len(y) - ok - bad}")
    m = GradientBoostingClassifier(**PARAMS).fit(X, y, sample_weight=w)
    trees = []
    for stage in m.estimators_:
        trees.append([[t.tree_.feature.tolist(), t.tree_.threshold.tolist(), t.tree_.children_left.tolist(),
                       t.tree_.children_right.tolist(), t.tree_.value[:, 0, 0].tolist()] for t in stage])
    model = {"about": "eval_real/train_guess.py 로 만듦 (QuickDraw CC BY 4.0 학습용 그림, %d 카테고리 × %d장, 예전 10 카테고리 무게 %g, 나무 깊이 %d × %d개)"
             % (len(EXPECT), a.n, a.old_weight, a.depth, a.trees), "min_p": a.min_p,
             "features": keys, "classes": m.classes_.tolist(), "lr": m.learning_rate,
             "init": m._raw_predict_init(X[:1])[0].tolist(), "trees": trees}
    out = HERE.parent / "guess_model.json"
    out.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    # numpy 로 읽는 쪽(rig_types.guess_proba)이 scikit-learn 과 같은 확률을 내는지
    rt._guess_model = None
    mine = np.array([[rt.guess_proba(dict(zip(keys, row)))[c] for c in m.classes_] for row in X])
    print("scikit-learn 과 차이(최대):", float(np.abs(mine - m.predict_proba(X)).max()), "→", out)


if __name__ == "__main__":
    main()
