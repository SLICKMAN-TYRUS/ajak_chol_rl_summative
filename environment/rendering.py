"""Rendering for AidPrepositioningEnv.

1. WebViewer: a live 3D simulation built with three.js (WebGL). Python runs a small
   HTTP server; the browser polls /api/state and animates convoys, airdrops,
   barges, flood risk and IPC phases. The same server exposes a JSON policy API
   (POST /api/policy) so a dashboard or mobile client can query the trained agent.
2. render_frame: a static 2D matplotlib frame (rgb_array) for reports and tests.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

WEB_DIR = Path(__file__).resolve().parent / "web"
IPC_COLORS = {1: "#cdfacd", 2: "#fae61e", 3: "#e67800", 4: "#c80000", 5: "#640000"}


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.state: dict = {}
        self.history: list[dict] = []
        self.policy_fn = None   # callable(list[float]) -> dict
        self.meta: dict = {}


class WebViewer:
    """Serve the three.js viewer and push environment states to it."""

    def __init__(self, port: int = 8765, open_browser: bool = True):
        self.shared = _State()
        shared = self.shared

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the terminal for agent logs
                pass

            def _send(self, code, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    return self._send(200, (WEB_DIR / "viewer.html").read_bytes(), "text/html")
                if path.startswith("/vendor/"):
                    f = (WEB_DIR / path.lstrip("/")).resolve()
                    if WEB_DIR in f.parents and f.exists():
                        return self._send(200, f.read_bytes(), "text/javascript")
                if path == "/api/state":
                    with shared.lock:
                        body = json.dumps(dict(seq=shared.seq, state=shared.state, meta=shared.meta))
                    return self._send(200, body.encode(), "application/json")
                if path == "/api/history":
                    with shared.lock:
                        body = json.dumps(shared.history)
                    return self._send(200, body.encode(), "application/json")
                return self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/api/policy":
                    return self._send(404, b"not found", "text/plain")
                n = int(self.headers.get("Content-Length", 0))
                try:
                    obs = json.loads(self.rfile.read(n) or b"{}")["observation"]
                    if shared.policy_fn is None:
                        raise RuntimeError("no policy loaded")
                    body = json.dumps(shared.policy_fn(obs)).encode()
                    return self._send(200, body, "application/json")
                except Exception as e:  # noqa: BLE001
                    return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        url = f"http://127.0.0.1:{port}/"
        print(f"\n[viewer] 3D simulation running at {url}  (open it in your browser)\n")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass

    def set_policy(self, fn, meta: dict | None = None):
        self.shared.policy_fn = fn
        if meta:
            self.shared.meta = meta

    def push(self, state: dict, delay: float = 0.8):
        with self.shared.lock:
            if state.get("week", 0) == 0:
                self.shared.history = []
            self.shared.seq += 1
            self.shared.state = state
            self.shared.history.append(state)
        if delay:
            time.sleep(delay)

    def close(self):
        self.server.shutdown()


def serve_forever(port: int = 8765, policy_fn=None, meta: dict | None = None):
    """Run only the API/viewer server (used by `main.py --serve`)."""
    v = WebViewer(port=port, open_browser=False)
    v.set_policy(policy_fn, meta)
    print(f"[api] POST http://127.0.0.1:{port}/api/policy  body: {{\"observation\": [41 floats]}}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        v.close()


# ----------------------------------------------------------------------------- #
# Static 2D frame (matplotlib) for rgb_array / report figures
# ----------------------------------------------------------------------------- #
def render_frame(s: dict, ax=None, title: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=110)
    else:
        fig = ax.figure
    ax.set_facecolor("#eef3e6")
    nile = np.array([[31.62, 5.6], [31.56, 6.21], [31.45, 6.9], [31.33, 7.6], [31.2, 8.3], [31.1, 8.6]])
    ax.plot(nile[:, 0], nile[:, 1], color="#3b82c4", lw=4, alpha=0.7, zorder=1, label="White Nile (barge route)")
    hx, hy = s["hub_lon"], s["hub_lat"]
    for c in s["counties"]:
        ax.plot([hx, c["lon"]], [hy, c["lat"]], ls="-" if c["road_open"] else "--",
                color="#7a7a7a" if c["road_open"] else "#d62728", lw=2 if c["road_open"] else 1.5, zorder=2)
        ax.add_patch(plt.Circle((c["lon"], c["lat"]), 0.12 + 0.25 * c["flood_risk"], color="#3b82c4",
                                alpha=0.18, zorder=1))
        ax.scatter(c["lon"], c["lat"], s=500, color=IPC_COLORS[c["ipc"]], edgecolor="k", zorder=3)
        ax.bar(c["lon"] + 0.17, c["stock"] * 0.12, width=0.06, bottom=c["lat"] - 0.1, color="#2ca02c", zorder=4)
        ax.text(c["lon"], c["lat"] - 0.27, f"{c['name']}\nIPC {c['ipc']} | stock {c['stock']:.1f}",
                ha="center", va="top", fontsize=7.5, zorder=5)
    ax.scatter(hx, hy, marker="s", s=420, color="#4b5563", edgecolor="k", zorder=3)
    ax.text(hx, hy - 0.22, f"Bor hub\nstock {s['hub']:.1f}", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.set_xlim(30.8, 33.6)
    ax.set_ylim(5.7, 8.6)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(f"Week {s['week']}/{s['horizon']} | budget {s['budget']:.1f} | "
                     f"last: {s['last']['action_name']}", fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#7a7a7a", lw=2, label="road open"),
               Line2D([], [], color="#d62728", ls="--", label="road cut by flood")] + \
              [Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=IPC_COLORS[k],
                      markeredgecolor="k", label=f"IPC {k}") for k in range(1, 6)]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    if own:
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
    return None
