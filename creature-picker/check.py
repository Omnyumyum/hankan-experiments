"""브라우저 없이 확인: 그림마다 점 미리 놓기 → 종류별로 움직여 GIF · 모아 보기를 만든다.

    venv/Scripts/python check.py                       # 기본 시험 그림
    venv/Scripts/python check.py 그림.png --kind wings  # 한 종류만
결과: out/이름_종류.gif, out/sheet.png
"""
import argparse
import base64
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import rig_types as rt

HERE = Path(__file__).parent
DEFAULT = ["drawings/animal_dog.png", "drawings_quad/q_horse_long_legs.png", "drawings/snake_s.png", "drawings/worm.png",
           "drawings/animal_bird.png", "drawings/monster_blob.png", "drawings/monster_spiky.png", "drawings/real_pig.png",
           HERE / "drawings/wings_butterfly.png", HERE / "drawings/wings_bird_front.png"]


def warp_in_order(tex, alpha, v0, v1, tris, canvas_hw, off):
    """브라우저(Pixi)처럼 삼각형을 순서대로 그린다: 나중 삼각형이 앞을 덮는다.
    (shape_rig.warp 는 겹친 곳에서 더 진한 쪽을 남겨 그리는 순서를 무시한다 → 날개가 몸 뒤로 가는지 확인할 수 없었다)"""
    Hc, Wc = canvas_hw
    out = np.full((Hc, Wc, 3), 255, np.float32)
    for t in tris:
        s_ = v0[t].astype(np.float32)
        d_ = (v1[t] + off).astype(np.float32)
        x, y, w, h = cv2.boundingRect(d_)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > Wc or y + h > Hc:
            continue
        M = cv2.getAffineTransform(s_, (d_ - [x, y]).astype(np.float32))
        patch = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tm = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(tm, np.round(d_ - [x, y]).astype(np.int32), 1.0)
        a = (cv2.warpAffine(alpha, M, (w, h)) * tm)[..., None]
        region = out[y:y + h, x:x + w]
        region[:] = patch * a + region * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def frames_of(d, kind, anim):
    mask, alpha = d.masks(kind)
    H, W = mask.shape
    pad = anim["pad"]
    n = anim["n_vertices"]
    pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
    uv = np.frombuffer(base64.b64decode(anim["uvs"]), np.float32).reshape(n, 2)
    tris = np.frombuffer(base64.b64decode(anim["indices"]), np.uint32).reshape(-1, 3).astype(np.int64)
    v0 = uv * [W, H]
    a = (cv2.GaussianBlur(alpha, (3, 3), 0) / 255.0).astype(np.float32)
    out = []
    for f in range(anim["frames"]):
        img = warp_in_order(d.crop.astype(np.float32), a, v0, pos[f], tris, (H + 2 * pad, W + 2 * pad), np.array([pad, pad], np.float32))
        for x, y in anim["pins"][f]:
            cv2.circle(img, (int(x + pad), int(y + pad)), 3, (0, 170, 0), -1)
        out.append(img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--kind", choices=rt.TYPES)
    args = ap.parse_args()
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    files = [Path(t) for t in args.targets] or [rt.AM / t for t in DEFAULT]
    rows = []
    for f in files:
        d = rt.Drawing(rt.sr.load_image(f))
        prop = rt.propose(d)
        print(f"\n{f.name}: AI 추측 = {prop['guess']}  {prop['why']}")
        tiles = []
        for kind in [args.kind] if args.kind else rt.TYPES:
            t0 = time.time()
            try:
                anim = rt.animate(d, kind, prop["points"][kind])
            except Exception as e:
                print(f"  {kind}: 실패 {e}")
                continue
            fr = frames_of(d, kind, anim)
            imageio.mimsave(out / f"{f.stem}_{kind}.gif", [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in fr], duration=anim["frame_ms"], loop=0)
            print(f"  {kind}: {anim['mesh']} 꼭짓점 {anim['n_vertices']} 삼각형 {anim['n_triangles']}  {time.time() - t0:.1f}s")
            pics = [cv2.resize(fr[i], (160, 160)) for i in (0, 8, 24)]
            tile = np.hstack(pics)
            tile = cv2.copyMakeBorder(tile, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, f"{f.stem[:18]} {kind}" + (" (AI)" if prop["guess"] == kind else ""), (4, 13), 0, 0.42, (0, 0, 0), 1)
            tiles.append(tile)
        if tiles:
            w = max(t.shape[1] for t in tiles)
            rows.append(np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]))
    if rows:
        w = max(r.shape[1] for r in rows)
        cv2.imwrite(str(out / "sheet.png"), np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]))
        print(f"\n→ {out / 'sheet.png'}")


if __name__ == "__main__":
    main()
