"""
ESC_TwoPoint_ID.py  —  LUT + ESC-Gradient Analytic + Brent (v3)

Python port of ESC_TwoPoint_ID.m

Burckhardt tire friction model:
    mu(s) = c1*(1 - exp(-c2*|s|)) - c3*|s|

Three identification layers:
    Layer 1 — LUT         : mu_max -> c2_lut  (coarse, fast)
    Layer 2 — ESC-Gradient: g_esc + (s, mu) -> c1, c3  (analytic, v3)
    Layer 3 — Brent       : two-point -> refined c2  (when excitation sufficient)

MATLAB persistent variables become instance attributes on ESCTwoPointID.
Call identifier.identify(...) each timestep.

debug_flag meanings:
    0  = Brent + gradient
    1  = warmup (t < t_start)
    10 = LUT + gradient
    20 = LUT/Brent + peak fallback
"""

import numpy as np
from typing import Tuple


class ESCTwoPointID:
    """
    Stateful tire-road friction identifier (Burckhardt model, v3).

    Usage
    -----
    identifier = ESCTwoPointID()
    result = identifier.identify(mu_measured, s_measured, s_probe,
                                  c1_prev, c2_prev, c3_prev,
                                  t, params, g_esc, a_esc, ...)

    Returns
    -------
    (c1_new, c2_new, c3_new, s_opt_out, mu_opt_out, valid, debug_flag,
     best_fark_out, fark_simdi_out, best_s_out, best_mu_out,
     s_at_mu_max_k, mu_opt_estimated)
    """

    # LUT: peak friction -> c2  (ice, snow, wet asphalt, dry asphalt)
    # Extended to include ice (mu~0.05) so low-friction surfaces aren't
    # incorrectly clamped to the snow entry (mu=0.19, c2=94.1).
    MU_LUT = np.array([0.05, 0.19, 0.40, 0.85, 1.15])
    C2_LUT = np.array([306.4, 94.1, 33.8, 33.8, 23.99])

    def __init__(self):
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self, buf_size: int, mu_buf_init: float,
                    s_buf_init: float, c2_prev: float) -> None:
        self._mu_buf       = np.full(buf_size, mu_buf_init)
        self._s_buf        = np.full(buf_size, s_buf_init)
        self._buf_idx      = 0
        self._best_s_probe = s_buf_init * 0.3
        self._best_mu_probe= mu_buf_init * 0.7
        self._best_fark    = 0.0
        self._mu_max_smooth= mu_buf_init
        self._c2_brent_last= c2_prev
        self._initialized  = True

    @staticmethod
    def _brent_residual(c2: float, s_k: float, mu_k: float,
                        s_p: float, mu_p: float) -> Tuple[float, bool]:
        """
        Residual for the two-point Brent root-find.
        Returns (residual, valid).
        Burckhardt peak gives: mu_k = c1*(1-exp(-c2*s_k)) with c3=c1*c2*exp(-c2*s_k)
        Probe consistency: f = mu_p - c1*(1-exp(-c2*s_p)) + c3*s_p
        """
        ek = np.exp(-c2 * s_k)
        g  = 1.0 - (1.0 + c2 * s_k) * ek
        if abs(g) < 1e-12:
            return 0.0, False
        c1t = mu_k / g
        c3t = c1t * c2 * ek
        ep  = np.exp(-c2 * s_p)
        return mu_p - c1t * (1.0 - ep) + c3t * s_p, True

    # ------------------------------------------------------------------
    # Main identification step
    # ------------------------------------------------------------------

    def identify(
        self,
        mu_measured: float,
        s_measured:  float,
        s_probe:     float,
        c1_prev:     float,
        c2_prev:     float,
        c3_prev:     float,
        t:           float,
        params:      np.ndarray,
        g_esc:       float = 0.0,
        a_esc:       float = 0.01,
        s_hat_esc:   float = 0.0,
        alpha:       float = 0.0,
        dither:      float = 0.0,
    ) -> Tuple:
        """
        One identification step. Call at every simulation timestep.

        Parameters (params array, 0-indexed, matching MATLAB params(1..15))
        ----------
        params[0]  buf_size         circular buffer length  (recommended: 8)
        params[1]  (unused slot)
        params[2]  fark_threshold   slip-excitation threshold (recommended: 0.008)
        params[3]  (unused slot)
        params[4]  t_start          start time for ID
        params[5]  c1_min
        params[6]  c1_max
        params[7]  c2_min
        params[8]  c2_max
        params[9]  c3_min
        params[10] c3_max
        params[11] mu_buf_init      initial buffer fill value
        params[12] s_buf_init       initial buffer fill value
        params[13] brent_tol        Brent convergence tolerance
        params[14] brent_iter       max Brent iterations
        """

        # ---- default outputs (no update) --------------------------------
        c1_new           = c1_prev
        c2_new           = c2_prev
        c3_new           = c3_prev
        if c3_prev > 0.0:
            s_opt_out = (np.log(max(c1_prev * c2_prev / c3_prev, 1.01))
                         / max(c2_prev, 0.1))
        else:
            s_opt_out = 0.01
        mu_opt_out       = 0.0
        valid            = 0
        debug_flag       = 0
        best_fark_out    = 0.0
        fark_simdi_out   = 0.0
        best_s_out       = 0.0
        best_mu_out      = 0.0
        mu_opt_estimated = 0.0

        # ---- unpack params (MATLAB 1-based -> Python 0-based) -----------
        buf_size        = int(round(params[0]))
        fark_threshold  = params[2]
        t_start         = params[4]
        c1_min, c1_max  = params[5], params[6]
        c2_min, c2_max  = params[7], params[8]
        c3_min, c3_max  = params[9], params[10]
        mu_buf_init     = params[11]
        s_buf_init      = params[12]
        brent_tol       = params[13]
        brent_iter      = int(round(params[14]))

        # ---- initialize persistent state --------------------------------
        if not self._initialized:
            self._init_state(buf_size, mu_buf_init, s_buf_init, c2_prev)

        # =================================================================
        # STEP 1 — Circular buffer update
        # =================================================================
        self._mu_buf[self._buf_idx] = mu_measured
        self._s_buf[self._buf_idx]  = s_measured
        self._buf_idx = (self._buf_idx + 1) % buf_size

        max_idx_k     = int(np.argmax(self._mu_buf))
        mu_max_k      = float(self._mu_buf[max_idx_k])
        s_at_mu_max_k = float(self._s_buf[max_idx_k])
        mu_opt_out    = mu_max_k

        # =================================================================
        # STEP 2 — Road-change detection (symmetric ±15 %)
        # =================================================================
        if mu_max_k < self._mu_max_smooth * 0.85:
            self._best_fark     = 0.0
            self._best_s_probe  = s_at_mu_max_k * 0.3
            self._best_mu_probe = mu_max_k * 0.7
            self._c2_brent_last = c2_prev

        if mu_max_k > self._mu_max_smooth * 1.15:
            self._best_fark     = 0.0
            self._best_s_probe  = s_at_mu_max_k * 0.3
            self._best_mu_probe = mu_max_k * 0.7
            self._c2_brent_last = c2_prev

        self._mu_max_smooth = 0.95 * self._mu_max_smooth + 0.05 * mu_max_k

        fark_simdi     = abs(s_probe - s_at_mu_max_k)
        fark_simdi_out = fark_simdi

        # Update best excitation BEFORE capturing outputs so returned values
        # reflect the state after this step (not stale from previous step).
        if (fark_simdi > fark_threshold
                and fark_simdi > self._best_fark * 1.1):
            self._best_fark     = fark_simdi
            self._best_s_probe  = s_probe
            self._best_mu_probe = mu_measured

        best_fark_out  = self._best_fark
        best_s_out     = self._best_s_probe
        best_mu_out    = self._best_mu_probe

        # =================================================================
        # STEP 3 — Warmup guard
        # =================================================================
        if t < t_start:
            debug_flag = 1
            return (c1_new, c2_new, c3_new, s_opt_out, mu_opt_out, valid,
                    debug_flag, best_fark_out, fark_simdi_out,
                    best_s_out, best_mu_out, s_at_mu_max_k, mu_opt_estimated)

        # =================================================================
        # STEP 4 — LAYER 1: LUT  -> c2_lut_val
        # =================================================================
        mu_k_cl    = float(np.clip(mu_max_k, self.MU_LUT[0], self.MU_LUT[-1]))
        c2_lut_val = float(np.interp(mu_k_cl, self.MU_LUT, self.C2_LUT))

        # =================================================================
        # STEP 5 — LAYER 3: Brent two-point root-find (when excitation OK)
        # =================================================================
        s_k = abs(s_at_mu_max_k)
        mu_k = mu_max_k
        s_p  = abs(self._best_s_probe)
        mu_p = self._best_mu_probe

        c2_solved     = self._c2_brent_last
        bracket_found = False

        if self._best_fark >= fark_threshold:
            bracket_range = c2_lut_val * 0.8

            # --- Try narrow bracket around LUT estimate first ---
            c2_lo1 = max(c2_min, c2_lut_val - bracket_range)
            c2_hi1 = min(c2_max, c2_lut_val + bracket_range)

            f_lo1, ok_lo = self._brent_residual(c2_lo1, s_k, mu_k, s_p, mu_p)
            f_hi1, ok_hi = self._brent_residual(c2_hi1, s_k, mu_k, s_p, mu_p)

            if ok_lo and ok_hi and f_lo1 * f_hi1 <= 0:
                bracket_found = True
                c2_lo, c2_hi = c2_lo1, c2_hi1
                f_low, f_high = f_lo1, f_hi1

            # --- Fall back to full range ---
            if not bracket_found:
                f_lo2, ok_lo2 = self._brent_residual(c2_min, s_k, mu_k, s_p, mu_p)
                f_hi2, ok_hi2 = self._brent_residual(c2_max, s_k, mu_k, s_p, mu_p)
                if ok_lo2 and ok_hi2 and f_lo2 * f_hi2 <= 0:
                    bracket_found = True
                    c2_lo, c2_hi = c2_min, c2_max
                    f_low, f_high = f_lo2, f_hi2

            # --- Brent iteration ---
            if bracket_found:
                a_b = c2_lo;  fa = f_low
                b_b = c2_hi;  fb = f_high
                c_br = a_b;   fc = fa
                d_br = b_b - a_b
                e_br = d_br

                for _ in range(brent_iter):
                    if fb * fc > 0:
                        c_br = a_b; fc = fa
                        d_br = b_b - a_b; e_br = d_br

                    if abs(fc) < abs(fb):
                        # Correct 3-way rotation: (a,b,c) ← (b,c,a)
                        temp_ab = a_b; a_b = b_b; b_b = c_br; c_br = temp_ab
                        temp_fa = fa;  fa  = fb;  fb  = fc;   fc  = temp_fa

                    tol1 = 2.0 * 2.2e-16 * abs(b_b) + 0.5 * brent_tol
                    xm   = 0.5 * (c_br - b_b)

                    if abs(xm) <= tol1 or abs(fb) < brent_tol:
                        break

                    if abs(e_br) >= tol1 and abs(fa) > abs(fb):
                        s_br = fb / fa
                        if a_b == c_br:
                            p_br = 2.0 * xm * s_br
                            q_br = 1.0 - s_br
                        else:
                            q_br = fa / fc
                            r_br = fb / fc
                            p_br = s_br * (2.0 * xm * q_br * (q_br - r_br)
                                           - (b_b - a_b) * (r_br - 1.0))
                            q_br = (q_br - 1.0) * (r_br - 1.0) * (s_br - 1.0)

                        if p_br > 0:
                            q_br = -q_br
                        else:
                            p_br = -p_br

                        if (abs(q_br) > 1e-14
                                and 2.0 * p_br < min(3.0 * xm * q_br - abs(tol1 * q_br),
                                                     abs(e_br * q_br))):
                            e_br = d_br
                            d_br = p_br / q_br
                        else:
                            d_br = xm
                            e_br = d_br
                    else:
                        d_br = xm
                        e_br = d_br

                    a_b = b_b; fa = fb
                    b_b = b_b + (d_br if abs(d_br) > tol1
                                 else np.sign(xm) * tol1)

                    fb, ok_b = self._brent_residual(b_b, s_k, mu_k, s_p, mu_p)
                    if not ok_b:
                        break

                c2_solved = float(np.clip(b_b, c2_min, c2_max))

                # Sanity: reject if too far from LUT
                if abs(c2_solved - c2_lut_val) > c2_lut_val * 0.6:
                    c2_solved     = c2_lut_val
                    bracket_found = False
                else:
                    self._c2_brent_last = c2_solved

        if not bracket_found:
            c2_solved = c2_lut_val

        # =================================================================
        # STEP 6 — LAYER 2: ESC-Gradient analytic c1, c3  (v3)
        # =================================================================
        # Two equations at current operating point (s_measured, mu_measured):
        #   mu  = c1*(1 - exp(-c2*|s|)) - c3*|s|               ... (1)
        #   g   = c1*c2*exp(-c2*|s|) - c3                       ... (2)
        # where g = g_true = -g_esc * 2 / a_esc
        #   (sign: ESC perturbs s_hat negatively; LPF output anti-correlates)
        #
        # Solving:
        #   c1 = (mu - g*|s|) / (1 - (1 + c2*|s|)*exp(-c2*|s|))
        #   c3 = c1*c2*exp(-c2*|s|) - g
        # =================================================================
        s_current  = abs(s_measured)
        mu_current = abs(mu_measured)

        c1_solved = c1_prev
        c3_solved = c3_prev

        use_gradient = (
            abs(a_esc) > 1e-6
            and abs(g_esc) < 100.0
            and s_current > 0.005
        )

        if use_gradient:
            g_true = -g_esc * 2.0 / a_esc
            ek_c   = np.exp(-c2_solved * s_current)
            denom  = 1.0 - (1.0 + c2_solved * s_current) * ek_c

            if abs(denom) > 1e-10:
                c1_solved = (mu_current - g_true * s_current) / denom
                c3_solved = c1_solved * c2_solved * ek_c - g_true
                debug_flag = 0 if bracket_found else 10
            else:
                use_gradient = False   # ill-conditioned -> fallback

        if not use_gradient:
            # Peak-based fallback (original v1 behaviour)
            s_k_abs = abs(s_at_mu_max_k)
            ek_f    = np.exp(-c2_solved * s_k_abs)
            g_f     = 1.0 - (1.0 + c2_solved * s_k_abs) * ek_f

            if abs(g_f) > 1e-10:
                c1_solved = mu_max_k / g_f
                c3_solved = c1_solved * c2_solved * ek_f
                debug_flag = 20
            else:
                return (c1_new, c2_new, c3_new, s_opt_out, mu_opt_out, valid,
                        debug_flag, best_fark_out, fark_simdi_out,
                        best_s_out, best_mu_out, s_at_mu_max_k, mu_opt_estimated)

        # =================================================================
        # STEP 7 — Physical bounds
        # =================================================================
        c1_solved = float(np.clip(c1_solved, c1_min, c1_max))
        c2_solved = float(np.clip(c2_solved, c2_min, c2_max))
        c3_solved = float(np.clip(c3_solved, c3_min, c3_max))

        # Require valid peak to exist: c1*c2 > c3
        if c1_solved * c2_solved <= c3_solved:
            return (c1_new, c2_new, c3_new, s_opt_out, mu_opt_out, valid,
                    debug_flag, best_fark_out, fark_simdi_out,
                    best_s_out, best_mu_out, s_at_mu_max_k, mu_opt_estimated)

        # =================================================================
        # STEP 8 — Optimal slip  s_opt = ln(c1*c2/c3) / c2
        # =================================================================
        ratio = c1_solved * c2_solved / c3_solved
        if ratio <= 1.0:
            return (c1_new, c2_new, c3_new, s_opt_out, mu_opt_out, valid,
                    debug_flag, best_fark_out, fark_simdi_out,
                    best_s_out, best_mu_out, s_at_mu_max_k, mu_opt_estimated)

        s_opt_abs = float(np.clip(np.log(ratio) / c2_solved, 0.01, 0.45))

        # =================================================================
        # Outputs
        # =================================================================
        c1_new           = c1_solved
        c2_new           = c2_solved
        c3_new           = c3_solved
        s_opt_out        = s_opt_abs
        valid            = 1
        mu_opt_estimated = (c1_solved * (1.0 - np.exp(-c2_solved * s_opt_abs))
                            - c3_solved * s_opt_abs)

        return (c1_new, c2_new, c3_new, s_opt_out, mu_opt_out, valid,
                debug_flag, best_fark_out, fark_simdi_out,
                best_s_out, best_mu_out, s_at_mu_max_k, mu_opt_estimated)


# ---------------------------------------------------------------------------
# Convenience: standalone function wrapper (mirrors MATLAB function signature)
# ---------------------------------------------------------------------------
_global_identifier = ESCTwoPointID()

def ESC_TwoPoint_ID(mu_measured, s_measured, s_probe,
                    c1_prev, c2_prev, c3_prev,
                    t, params, g_esc=0.0, a_esc=0.01,
                    s_hat_esc=0.0, alpha=0.0, dither=0.0):
    """
    Stateless-style wrapper that uses a module-level ESCTwoPointID instance.

    WARNING: Uses a single shared state — only suitable for single-wheel use.
    For multi-wheel simulations, instantiate ESCTwoPointID() per wheel.
    """
    return _global_identifier.identify(
        mu_measured, s_measured, s_probe,
        c1_prev, c2_prev, c3_prev,
        t, params, g_esc, a_esc, s_hat_esc, alpha, dither
    )
