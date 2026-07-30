"""HTTP + SSE server for the runtime demo.

Same architecture as thor_profile/server.py: stdlib-only, SSE rather than
WebSocket (data flows server → browser only; control is plain POST).

    GET  /            the page (web/index.html)
    GET  /clips       JSON list of available clips
    GET  /status      current engine status
    GET  /stream      SSE stream (status + telemetry packets)
    GET  /audio/in.wav | /audio/A.wav | /audio/B.wav
                      what has been played so far, as WAV
    POST /control     {"action": load|config|play|pause|restart|speed, ...}

Run:  ../.venv/bin/python server.py [--port 8085] [--bind 0.0.0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import sys
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# CPU-only, single-thread BLAS — the measured configuration (see PLAN.md).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from engine import Engine, list_clips                    # noqa: E402
from sweep import run_sweep                              # noqa: E402


class Hub:
    """Fan-out of JSON events to connected SSE clients."""

    def __init__(self) -> None:
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()

    def attach(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._clients.append(q)
        return q

    def detach(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def broadcast(self, msg: dict) -> None:
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                # Slow client: drop the oldest so live data keeps flowing.
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass


class _Handler(BaseHTTPRequestHandler):
    hub: Hub
    engine: Engine

    def log_message(self, fmt, *args):                    # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:                             # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            page = (HERE / "web" / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if path == "/clips":
            return self._json(list_clips())
        if path == "/status":
            return self._json(self.engine.status())
        if path == "/stream":
            return self._stream()
        if path.startswith("/audio/") and path.endswith(".wav"):
            which = path[len("/audio/"):-len(".wav")]
            try:
                wav = self.engine.audio_wav(which)
            except Exception as e:                        # noqa: BLE001
                return self._json({"ok": False, "reason": str(e)}, 400)
            return self._send(200, wav, "audio/wav")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:                            # noqa: N802
        if self.path.split("?", 1)[0] != "/control":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, OSError):
            return self._json({"ok": False, "reason": "bad json"}, 400)
        action = req.get("action")
        eng = self.engine
        try:
            if action == "load":
                st = eng.load_clip(str(req.get("clip", "")))
            elif action == "config":
                st = eng.set_config(str(req.get("side", "")),
                                    dict(req.get("cfg") or {}))
            elif action == "play":
                st = eng.play()
            elif action == "pause":
                st = eng.pause()
            elif action == "restart":
                st = eng.restart()
            elif action == "speed":
                st = eng.set_speed(float(req.get("speed", 1.0)))
            elif action == "sweep":
                st = self._start_sweep(bool(req.get("no_cache", False)))
            else:
                return self._json({"ok": False, "reason": "unknown action"}, 400)
        except Exception as e:                            # noqa: BLE001
            return self._json({"ok": False, "reason": str(e)}, 400)
        return self._json({"ok": True, "status": st})

    _sweep_lock = threading.Lock()
    _sweep_running = False

    def _start_sweep(self, no_cache: bool) -> dict:
        """Kick a background sweep of the current clip. The engine is paused
        first — a live pipeline racing the sweep would corrupt both its own
        pacing and the sweep's power measurement."""
        cls = _Handler
        with cls._sweep_lock:
            if cls._sweep_running:
                raise RuntimeError("sweep already running")
            st = self.engine.status()
            if not st.get("clip"):
                raise RuntimeError("no clip loaded")
            cls._sweep_running = True
        clip_name = st["clip"]["name"]
        self.engine.pause()
        hub = self.hub

        def work() -> None:
            try:
                def prog(i, n, label):
                    hub.broadcast({"type": "sweep_progress",
                                   "i": i, "n": n, "label": label})
                res = run_sweep(clip_name, prog, use_cache=not no_cache)
                hub.broadcast(res)
            except Exception as e:                        # noqa: BLE001
                hub.broadcast({"type": "sweep_error", "reason": str(e)})
            finally:
                with cls._sweep_lock:
                    cls._sweep_running = False

        threading.Thread(target=work, daemon=True, name="sweep").start()
        return st

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = self.hub.attach()
        try:
            first = json.dumps(self.engine.status(), separators=(",", ":"))
            self.wfile.write(f"data: {first}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(msg, separators=(",", ":"))
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.hub.detach(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()

    hub = Hub()
    engine = Engine(hub)
    _Handler.hub, _Handler.engine = hub, engine

    srv = ThreadingHTTPServer((args.bind, args.port), _Handler)
    srv.daemon_threads = True
    print(f"[runtime_demo] http://{args.bind}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
