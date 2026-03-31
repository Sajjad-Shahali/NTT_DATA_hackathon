"""
Real-Time Tire-Road Friction Identification — Flask backend
Serves the landing page and the in-browser Burckhardt NLS classifier API.
"""
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.optimize import curve_fit

app = Flask(__name__)

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


def _burckhardt(s, c1, c2, c3):
    return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)


def classify(slips, mus):
    """Burckhardt NLS + nearest-neighbour. Returns (label, c1, c2, c3, mu_peak)."""
    slips = np.asarray(slips, dtype=float)
    mus   = np.asarray(mus,   dtype=float)
    best_params, best_resid = None, np.inf
    for seed in SEEDS:
        try:
            popt, _ = curve_fit(
                _burckhardt, slips, mus,
                p0=seed,
                bounds=([0, 0, 0], [3.0, 400.0, 3.0]),
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

    s_opt   = np.log(c1 * c2 / max(c3, 1e-9)) / max(c2, 1e-9)
    mu_peak = float(_burckhardt(max(s_opt, 0.0), c1, c2, c3))

    return best_label, float(c1), float(c2), float(c3), mu_peak


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data  = request.get_json(silent=True) or {}
    slips = data.get("slips", [])
    mus   = data.get("mus",   [])
    if len(slips) < 10 or len(mus) < 10:
        return jsonify({"surface": None})
    label, c1, c2, c3, mu_peak = classify(slips, mus)
    if label is None:
        return jsonify({"surface": None})
    return jsonify({
        "surface":  label,
        "c1":       round(c1, 3),
        "c2":       round(c2, 2),
        "c3":       round(c3, 3),
        "mu_peak":  round(mu_peak, 3),
    })


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)
