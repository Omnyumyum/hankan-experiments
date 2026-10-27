"""종류 고르기 시험판 로컬 서버 (표준 라이브러리 http.server 만 씀).

    venv/Scripts/python server.py          # → http://localhost:8765

API
- GET  /api/samples                → 시험 그림 목록
- POST /api/prepare  {image: dataURL | sample: "drawings/animal_dog.png"}
                                   → {id, crop(dataURL), w, h, guess, points{종류: 점}, why, notes}
- POST /api/animate  {id, kind, points}
                                   → 메시 · 프레임별 꼭짓점(base64 float32) · 텍스처. 그리는 것은 브라우저 Pixi
- POST /api/weights  {id, points}  → 날개가 움직일 범위 겹침 그림 (점 고치기 화면용), 칸 고르기면 자동 뿌리
- POST /api/region   {id, x, y} | {id, points} → 누른 곳의 칸 번호 | 문지른 길이 지나간 칸들 (칸 고르기)
"""
import base64
import json
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import rig_types as rt

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
PORT = 8765
SAMPLE_DIRS = {"drawings": rt.AM / "drawings", "drawings_quad": rt.AM / "drawings_quad", "wings": HERE / "drawings"}
SESSIONS, LOCK, KEEP = {}, threading.Lock(), 30


def samples():
    out = []
    for key, d in SAMPLE_DIRS.items():
        if d.is_dir():
            out += [f"{key}/{f.name}" for f in sorted(d.iterdir()) if f.suffix.lower() in rt.sr.IMG_EXT]
    return out


def read_image(body):
    if body.get("sample"):
        key, _, name = body["sample"].partition("/")
        if key not in SAMPLE_DIRS or "/" in name or "\\" in name or ".." in name:
            raise ValueError("시험 그림 이름이 이상합니다")
        return rt.sr.load_image(SAMPLE_DIRS[key] / name)
    data = body.get("image", "")
    raw = base64.b64decode(data.split(",", 1)[1] if "," in data else data)
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("그림 파일이 너무 큽니다 (15MB)")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("그림 파일로 읽을 수 없습니다")
    tmp = HERE / "out" / f"_upload_{uuid.uuid4().hex}.png"
    tmp.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(tmp), img)
    try:
        return rt.sr.load_image(tmp)  # 투명 PNG · 흑백 · 큰 사진 처리를 shape_rig 와 똑같이
    finally:
        tmp.unlink(missing_ok=True)


def prepare(body):
    t0 = time.time()
    d = rt.Drawing(read_image(body))
    prop = rt.propose(d)
    sid = uuid.uuid4().hex[:12]
    with LOCK:
        SESSIONS[sid] = d
        while len(SESSIONS) > KEEP:
            SESSIONS.pop(next(iter(SESSIONS)))
    ok, png = cv2.imencode(".png", d.crop)
    H, W = d.mask.shape
    return {"id": sid, "crop": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(), "w": W, "h": H,
            **prop, "notes": d.notes, "sec": round(time.time() - t0, 2)}


def animate(body):
    t0 = time.time()
    with LOCK:
        d = SESSIONS.get(body.get("id"))
    if d is None:
        raise ValueError("그림을 다시 올려 주세요 (서버가 다시 켜졌거나 오래됨)")
    out = rt.animate(d, body.get("kind"), body.get("points") or {})
    out["sec"] = round(time.time() - t0, 2)
    return out


def weights(body):
    with LOCK:
        d = SESSIONS.get(body.get("id"))
    if d is None:
        raise ValueError("그림을 다시 올려 주세요 (서버가 다시 켜졌거나 오래됨)")
    return rt.wing_overlay(d, body.get("points") or {})


def region(body):
    with LOCK:
        d = SESSIONS.get(body.get("id"))
    if d is None:
        raise ValueError("그림을 다시 올려 주세요 (서버가 다시 켜졌거나 오래됨)")
    if body.get("points"):  # 문지르기
        p0 = body["points"][0]
        return {"regions": rt.regions_along(d, body["points"]), "first": rt.region_at(d, float(p0[0]), float(p0[1]))}
    return {"region": rt.region_at(d, float(body["x"]), float(body["y"]))}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            print(time.strftime("%H:%M:%S"), fmt % args)

    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/samples":
            return self.send_json(200, samples())
        return super().do_GET()

    def do_POST(self):
        route = {"/api/prepare": prepare, "/api/animate": animate, "/api/weights": weights, "/api/region": region}.get(self.path)
        if route is None:
            return self.send_json(404, {"error": "없는 주소"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.send_json(200, route(body))
        except Exception as e:  # 그림 하나가 안 돼도 서버는 계속 돈다
            traceback.print_exc()
            self.send_json(400, {"error": str(e)})


if __name__ == "__main__":
    print(f"http://localhost:{PORT}  (끄려면 Ctrl+C)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
