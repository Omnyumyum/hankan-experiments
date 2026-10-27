"""동물 관절 모델(ViTPose++ APT-36K, AP-10K 17점)을 그림에 돌려 겹쳐 그린다. CPU, onnxruntime."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import ViTPose

HERE = Path(__file__).parent
MODELS = {
    "b": (str(HERE / "models/vitpose-b-apt36k.onnx"), (192, 256)),
    "h": (str(HERE / "models/vitpose-h-apt36k.onnx"), (288, 384)),
}
NAMES = ["l_eye", "r_eye", "nose", "neck", "tail_root", "l_shoulder", "l_elbow", "l_fpaw", "r_shoulder", "r_elbow",
         "r_fpaw", "l_hip", "l_knee", "l_bpaw", "r_hip", "r_knee", "r_bpaw"]
BONES = [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10), (4, 11), (11, 12), (12, 13),
         (4, 14), (14, 15), (15, 16)]


def ink_bbox(img):
    """종이 위 선이 있는 영역. 그림 찾기 모델 대신 쓴다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    h, w = g.shape
    m = 0.03
    th[: int(h * m)] = 0; th[-int(h * m):] = 0; th[:, : int(w * m)] = 0; th[:, -int(w * m):] = 0
    ys, xs = np.nonzero(th)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 0.1 * max(x1 - x0, y1 - y0)
    return [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]


def color(i):
    n = NAMES[i]
    return (0, 0, 255) if n.startswith("r_") else (255, 0, 0) if n.startswith("l_") else (0, 160, 0)


def main(size):
    url, inp = MODELS[size]
    model = ViTPose(url, model_input_size=inp, backend="onnxruntime", device="cpu")
    out, tiles = {}, []
    for f in sorted((HERE / "drawings").glob("*.png")):
        img = cv2.imread(str(f))
        box = ink_bbox(img)
        t0 = time.time()
        kpts, scores = model(img, bboxes=[box])
        ms = (time.time() - t0) * 1000
        k, s = kpts[0], scores[0]
        out[f.stem] = {"kpts": k.tolist(), "scores": s.tolist(), "bbox": [float(v) for v in box], "ms": ms}
        vis = img.copy()
        for a, b in BONES:
            cv2.line(vis, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), color(b), 3)
        for i, p in enumerate(k):
            cv2.circle(vis, tuple(int(v) for v in p), 6, color(i), -1)
        h, w = vis.shape[:2]
        sc = 300 / max(h, w)
        t = cv2.resize(vis, (int(w * sc), int(h * sc)))
        tile = np.full((330, 300, 3), 255, np.uint8)
        tile[30:30 + t.shape[0], :t.shape[1]] = t
        cv2.putText(tile, f.stem, (4, 14), 0, 0.45, (0, 0, 0), 1)
        cv2.putText(tile, f"mean {np.mean(s):.2f} min {np.min(s):.2f} {ms:.0f}ms", (4, 27), 0, 0.4, (0, 0, 180), 1)
        tiles.append(tile)
        print(f.stem, round(float(np.mean(s)), 2), round(float(np.min(s)), 2), f"{ms:.0f}ms")
    while len(tiles) % 5:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(HERE / f"animal_{size}_sheet.png"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
    (HERE / f"animal_{size}.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b")
