"""외곽선을 지키는 메시 (PythonCDT, MPL-2.0).

shape_rig.build_grid_mesh 는 칸에 그림이 조금만 있어도 삼각형을 둔다. 그래서 칸 크기(6px)보다 좁은 다리 사이 틈이 메워져
다리가 붙어 움직인다. 여기서는 마스크 외곽선을 고정 모서리로 넣고 안쪽 점만 더해 삼각형을 만든다.
외곽선 밖(다리 사이 틈)에는 삼각형이 생기지 않는다. 실패하면 격자 메시로 돌아간다.
"""
import cv2
import numpy as np
import pythoncdt as cdt

import shape_rig as sr

EDGE = 4  # 외곽선 점 간격 (px). 짧을수록 선이 덜 깎인다
STEP = 8  # 안쪽 점 간격


def components(tris):
    """모서리를 나눠 갖는 삼각형끼리 덩어리로 묶는다 (shape_rig.build_mesh 와 같은 규칙)."""
    parent = list(range(len(tris)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for ti, (a, b, c) in enumerate(tris):
        for e in ((a, b), (b, c), (c, a)):
            e = (min(e), max(e))
            if e in owner:
                ra, rb = find(ti), find(owner[e])
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[e] = ti
    roots = np.array([find(i) for i in range(len(tris))])
    _, tri_comp = np.unique(roots, return_inverse=True)
    comps = [np.unique(tris[tri_comp == c]) for c in range(tri_comp.max() + 1)]
    return comps, tri_comp


def resample(contour, step):
    """외곽선을 둘레 길이로 같은 간격 점으로."""
    c = contour.astype(np.float64)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(8, int(s[-1] / step))
    at = np.linspace(0, s[-1], n, endpoint=False)
    cc = np.vstack([c, c[:1]])
    return np.stack([np.interp(at, s, cc[:, 0]), np.interp(at, s, cc[:, 1])], 1)


def build_cdt_mesh(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    verts, edges, seen = [], [], {}

    def add(p):
        k = (round(p[0], 1), round(p[1], 1))
        if k not in seen:
            seen[k] = len(verts)
            verts.append(k)
        return seen[k]

    for c in cs:
        if cv2.contourArea(c) < 30:
            continue
        ring = [add(p) for p in resample(c[:, 0, :], EDGE)]
        ring = [v for i, v in enumerate(ring) if v != ring[i - 1]]
        edges += [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    # 안쪽 점: 외곽선에서 EDGE 이상 떨어진 격자점만 (붙으면 납작한 삼각형이 생긴다)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt > EDGE)
    for x, y in zip(xs, ys):
        if x % STEP == 0 and y % STEP == 0:
            add((float(x), float(y)))
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices([cdt.V2d(x, y) for x, y in verts])
    t.insert_edges([cdt.Edge(a, b) for a, b in edges if a != b])
    t.erase_outer_triangles_and_holes()
    va = t.vertices_array()
    v = np.stack([va["x"], va["y"]], 1).astype(np.float32)
    tris = t.triangles_array()["vertices"].astype(np.int64)
    # 겹친 외곽선(아주 가는 선)을 TRY_RESOLVE 가 풀면서 생긴 바깥 삼각형을 한 번 더 거른다
    cen = v[tris].mean(1)
    inside = mask[np.clip(np.round(cen[:, 1]).astype(int), 0, mask.shape[0] - 1), np.clip(np.round(cen[:, 0]).astype(int), 0, mask.shape[1] - 1)] > 0
    tris = tris[inside]
    used = np.unique(tris)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    v, tris = v[used], remap[tris].astype(np.int32)
    return v, tris, components(tris)


def build_mesh(mask):
    """반환: 꼭짓점, 삼각형, (덩어리, 삼각형별 덩어리), 쓴 방식."""
    try:
        v, tris, comps = build_cdt_mesh(mask)
        if len(tris) >= 8:
            return v, tris, comps, "cdt"
    except Exception:  # CDT 가 못 풀면 예전 격자 메시
        pass
    v, tris, comps = sr.build_grid_mesh(mask)
    return v, tris, comps, "grid"
