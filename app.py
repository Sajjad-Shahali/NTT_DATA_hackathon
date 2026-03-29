"""
GripSense Live — Flask backend
Replays friction_data_full.csv via Server-Sent Events, runs sliding-window
Burckhardt NLS classifier, streams results to the dashboard.
"""
import json
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, jsonify
from scipy.optimize import curve_fit

app = Flask(__name__)

# ── Data ────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "raw" / "friction_data_full.csv"

def load_replay_frames():
    """
    Build a list of frames for replay.
    Each frame = {time, left:{slip,mu,surface,alpha}, right:{…}}
    Left = FL wheel,  Right = FR wheel (both from the same simulation).
    We pick one representative simulation (test_no=1, mode=Framework, scenario=Snow Low),
    then loop through all simulations on repeat.
    """
    df = pd.read_csv(DATA_PATH)
    # Dataset only has FL (front-left) and RL (rear-left)
    # Map FL → left panel, RL → right panel
    df = df[df["wheel"].isin(["FL", "RL"])].copy()
    df = df.sort_values(["test_no", "case_no", "wheel", "time"]).reset_index(drop=True)

    frames = []
    grouped = df.groupby(["test_no", "case_no"])
    for (test_no, case_no), grp in grouped:
        fl = grp[grp["wheel"] == "FL"].sort_values("time").reset_index(drop=True)
        fr = grp[grp["wheel"] == "RL"].sort_values("time").reset_index(drop=True)
        n = min(len(fl), len(fr))
        for i in range(n):
            frames.append({
                "test_no": int(test_no),
                "case_no": int(case_no),
                "left":  {
                    "slip":    float(fl.at[i, "slip"]),
                    "mu":      float(fl.at[i, "mu_noisy"]),
                    "surface": str(fl.at[i, "surface"]),
                    "alpha":   float(fl.at[i, "alpha"]),
                },
                "right": {
                    "slip":    float(fr.at[i, "slip"]),
                    "mu":      float(fr.at[i, "mu_noisy"]),
                    "surface": str(fr.at[i, "surface"]),
                    "alpha":   float(fr.at[i, "alpha"]),
                },
            })
    return frames


print("Loading replay data …")
FRAMES = load_replay_frames()
print(f"  {len(FRAMES)} frames loaded from {DATA_PATH.name}")

# ── Classifier ──────────────────────────────────────────────────────────────
PROTOTYPES = {
    "Dry asphalt": np.array([1.280, 23.99, 0.520]),
    "Wet asphalt": np.array([0.857, 33.82, 0.347]),
    "Snow":        np.array([0.195, 94.13, 0.065]),
}
NORM_MIN = np.array([0.195, 23.99, 0.065])
NORM_RNG = np.array([1.280 - 0.195, 94.13 - 23.99, 0.520 - 0.065])
SEEDS = [
    [1.28, 24.0, 0.52],
    [0.86, 34.0, 0.35],
    [0.19, 94.0, 0.065],
]
WINDOW = 50


def _burckhardt(s, c1, c2, c3):
    return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)


def classify(slips, mus):
    """
    Run Burckhardt NLS with multiple seeds; return (label, c1, c2, c3, mu_peak).
    Returns None values on failure.
    """
    slips = np.asarray(slips, dtype=float)
    mus   = np.asarray(mus,   dtype=float)
    best_params, best_resid = None, np.inf
    for seed in SEEDS:
        try:
            popt, _ = curve_fit(
                _burckhardt, slips, mus,
                p0=seed,
                bounds=([0, 0, 0], [3.0, 300.0, 3.0]),
                maxfev=800,
            )
            resid = float(np.mean((_burckhardt(slips, *popt) - mus) ** 2))
            if resid < best_resid:
                best_resid = resid
                best_params = popt
        except Exception:
            pass

    if best_params is None:
        return None, None, None, None, None

    c1, c2, c3 = best_params
    vec = np.array([c1, c2, c3])
    norm_vec = (vec - NORM_MIN) / NORM_RNG

    best_label, best_dist = None, np.inf
    for label, proto in PROTOTYPES.items():
        d = float(np.linalg.norm(norm_vec - (proto - NORM_MIN) / NORM_RNG))
        if d < best_dist:
            best_dist = d
            best_label = label

    # mu_peak = c1 * (1 - exp(-c2*s*)) - c3*s*  at optimal slip
    s_opt = np.log(c1 * c2 / max(c3, 1e-9)) / max(c2, 1e-9)
    mu_peak = float(_burckhardt(max(s_opt, 0.0), c1, c2, c3))

    return best_label, float(c1), float(c2), float(c3), mu_peak


# ── Stream state ────────────────────────────────────────────────────────────
_state = {
    "pos":    0,
    "speed":  1.0,
    "paused": False,
    "lock":   threading.Lock(),
}

# Per-side sliding window buffers (rebuilt per SSE client to avoid sharing)
class _SideBuffer:
    def __init__(self):
        self.slips = deque(maxlen=WINDOW)
        self.mus   = deque(maxlen=WINDOW)
        self.surface = "—"
        self.c1, self.c2, self.c3, self.mu_peak = 0.0, 0.0, 0.0, 0.0
        self.last_classified = -1  # frame index

    def push(self, slip, mu, frame_idx):
        self.slips.append(slip)
        self.mus.append(mu)
        if len(self.slips) == WINDOW and frame_idx != self.last_classified:
            self.last_classified = frame_idx
            label, c1, c2, c3, mu_peak = classify(list(self.slips), list(self.mus))
            if label:
                self.surface = label
                self.c1, self.c2, self.c3 = c1, c2, c3
                self.mu_peak = mu_peak

    def to_dict(self, slip, mu):
        return {
            "slip":        round(slip, 4),
            "mu":          round(mu, 4),
            "surface":     self.surface,
            "c1":          round(self.c1, 3),
            "c2":          round(self.c2, 2),
            "c3":          round(self.c3, 3),
            "mu_peak":     round(self.mu_peak, 3),
            "buffer_fill": len(self.slips),
        }


def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    """SSE endpoint — streams one frame every 100ms (adjustable by speed)."""
    def generate():
        left_buf  = _SideBuffer()
        right_buf = _SideBuffer()
        local_pos = _state["pos"]

        while True:
            with _state["lock"]:
                speed  = _state["speed"]
                paused = _state["paused"]

            if paused:
                time.sleep(0.1)
                yield _event({"paused": True})
                continue

            frame = FRAMES[local_pos % len(FRAMES)]
            idx   = local_pos % len(FRAMES)

            left_buf.push(frame["left"]["slip"],  frame["left"]["mu"],  idx)
            right_buf.push(frame["right"]["slip"], frame["right"]["mu"], idx)

            payload = {
                "left":       left_buf.to_dict(frame["left"]["slip"],  frame["left"]["mu"]),
                "right":      right_buf.to_dict(frame["right"]["slip"], frame["right"]["mu"]),
                "left_true":  frame["left"]["surface"],
                "right_true": frame["right"]["surface"],
                "row":        idx,
                "total":      len(FRAMES),
                "speed":      speed,
                "test_no":    frame["test_no"],
                "case_no":    frame["case_no"],
            }
            yield _event(payload)

            local_pos += 1
            with _state["lock"]:
                _state["pos"] = local_pos % len(FRAMES)

            delay = max(0.04, 0.1 / max(speed, 0.1))
            time.sleep(delay)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/speed", methods=["POST"])
def set_speed():
    data = request.get_json(silent=True) or {}
    with _state["lock"]:
        _state["speed"] = float(data.get("speed", 1.0))
    return jsonify({"speed": _state["speed"]})


@app.route("/api/pause", methods=["POST"])
def toggle_pause():
    with _state["lock"]:
        _state["paused"] = not _state["paused"]
    return jsonify({"paused": _state["paused"]})


@app.route("/api/reset", methods=["POST"])
def reset():
    with _state["lock"]:
        _state["pos"] = 0
    return jsonify({"pos": 0})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(silent=True) or {}
    slips = data.get("slips", [])
    mus   = data.get("mus",   [])
    if len(slips) < 10 or len(mus) < 10:
        return jsonify({"surface": None})
    label, c1, c2, c3, mu_peak = classify(slips, mus)
    if label is None:
        return jsonify({"surface": None})
    return jsonify({"surface": label, "c1": round(c1, 3),
                    "c2": round(c2, 2), "c3": round(c3, 3),
                    "mu_peak": round(mu_peak, 3)})


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)
