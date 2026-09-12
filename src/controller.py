
"""Physics-informed adaptive controller.

Version 1 controls:
    - high-pulse volumetric airflow
    - duty cycle

Pulse frequency remains fixed at 4 Hz in the first study so we can isolate
whether real-time feed information has value before adding another control
degree of freedom.

The controller uses a learned surrogate of the dynamic separator. It does not
receive true material, coating fraction, density, or suspension velocity from
the plant.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from config import D_STRAIGHT


def flow_to_velocity(flow_m3_h):
    q = float(flow_m3_h) / 3600.0
    area = math.pi * D_STRAIGHT**2 / 4.0
    return q / area


class AdaptiveController:
    def __init__(
        self,
        fraction,
        surrogate,
        suspension_lookup,
        nominal_flow_m3_h,
        frequency_hz=4.0,
        low_flow_fraction=0.35,
        target_top_cu_fraction=0.12,
    ):
        self.fraction = fraction
        self.surrogate = surrogate
        self.suspension_lookup = suspension_lookup
        self.nominal_flow_m3_h = float(nominal_flow_m3_h)
        self.frequency_hz = float(frequency_hz)
        self.low_flow_fraction = float(low_flow_fraction)
        self.target_top_cu_fraction = float(target_top_cu_fraction)

        # Candidate actions are deliberately bounded around the literature
        # operating point. They are engineering search limits, not literature
        # optima.
        self.flow_candidates = np.arange(
            max(45.0, self.nominal_flow_m3_h - 20.0),
            self.nominal_flow_m3_h + 20.1,
            5.0,
        )
        self.duty_candidates = np.array([0.35, 0.45, 0.55, 0.65])

        self.previous_top_cu_fraction = None

    def _estimate_us(self, camera_material, size_mm):
        """
        Estimate suspension velocity from observable material label + size.

        Coating is latent. The lookup table was built from the calibrated feed
        model, so this is an EXPECTED suspension velocity, not the true one.
        """
        key = str(camera_material)

        table = self.suspension_lookup.get(key)
        if table is None or len(table) == 0:
            return 1.4

        sizes = table["size_mm"].to_numpy(dtype=float)
        us = table["mean_us"].to_numpy(dtype=float)
        idx = int(np.argmin(np.abs(sizes - float(size_mm))))
        return float(us[idx])

    def _controller_particles(self, camera_obs):
        if camera_obs.empty:
            return pd.DataFrame(
                columns=[
                    "material_est",
                    "size_mm",
                    "u_s_est",
                ]
            )

        p = pd.DataFrame(
            {
                "material_est": camera_obs["camera_material"].astype(str),
                "size_mm": camera_obs["camera_size_mm"].astype(float),
            }
        )
        p["u_s_est"] = [
            self._estimate_us(m, s)
            for m, s in zip(
                p["material_est"],
                p["size_mm"],
            )
        ]
        return p

    def _score_action(self, estimated_particles, flow, duty):
        if estimated_particles.empty:
            return -1e9, {}

        velocity = flow_to_velocity(flow)

        X = pd.DataFrame(
            {
                "u_s": estimated_particles["u_s_est"],
                "size_mm": estimated_particles["size_mm"],
                "flow_velocity": velocity,
                "duty": duty,
                "low_flow_fraction": self.low_flow_fraction,
            }
        )

        probs = self.surrogate.predict_proba(X)
        classes = list(self.surrogate.classes_)

        def prob(label):
            if label not in classes:
                return np.zeros(len(X))
            return probs[:, classes.index(label)]

        p_light = prob("light")
        p_heavy = prob("heavy")
        p_active = prob("active")

        mat = estimated_particles["material_est"].to_numpy()
        al = mat == "aluminum"
        cu = mat == "copper"

        def safe_mean(x):
            return float(np.mean(x)) if len(x) else 0.0

        al_recovery_light = safe_mean(p_light[al])
        cu_recovery_heavy = safe_mean(p_heavy[cu])

        expected_light_total = float(np.sum(p_light))
        expected_heavy_total = float(np.sum(p_heavy))

        al_light = float(np.sum(p_light[al]))
        cu_heavy = float(np.sum(p_heavy[cu]))

        al_purity = (
            al_light / expected_light_total
            if expected_light_total > 1e-9
            else 0.0
        )
        cu_purity = (
            cu_heavy / expected_heavy_total
            if expected_heavy_total > 1e-9
            else 0.0
        )
        unresolved = safe_mean(p_active)

        # Balanced quality/recovery objective.
        score = (
            0.30 * al_recovery_light
            + 0.30 * cu_recovery_heavy
            + 0.18 * al_purity
            + 0.18 * cu_purity
            - 0.04 * unresolved
        )

        # Small actuator penalty keeps the controller near the nominal Zhu
        # setting unless the camera information predicts a meaningful gain.
        score -= 0.015 * abs(flow - self.nominal_flow_m3_h) / 20.0
        score -= 0.010 * abs(duty - 0.50) / 0.15

        return score, {
            "pred_al_recovery": al_recovery_light,
            "pred_cu_recovery": cu_recovery_heavy,
            "pred_al_purity": al_purity,
            "pred_cu_purity": cu_purity,
            "pred_unresolved": unresolved,
        }

    def choose_action(self, camera_obs):
        particles = self._controller_particles(camera_obs)

        best = None
        for flow in self.flow_candidates:
            for duty in self.duty_candidates:
                score, metrics = self._score_action(
                    particles,
                    flow,
                    duty,
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "flow_m3_h": float(flow),
                        "duty": float(duty),
                        **metrics,
                    }

        # Outlet-camera feedback:
        # if Cu contamination in the previous TOP stream exceeded the target,
        # reduce upward impulse one small step.
        if self.previous_top_cu_fraction is not None:
            error = (
                self.previous_top_cu_fraction
                - self.target_top_cu_fraction
            )
            if error > 0.03:
                best["flow_m3_h"] = max(
                    self.flow_candidates.min(),
                    best["flow_m3_h"] - 5.0,
                )
                best["feedback_action"] = "decrease_flow"
            elif error < -0.06:
                best["flow_m3_h"] = min(
                    self.flow_candidates.max(),
                    best["flow_m3_h"] + 5.0,
                )
                best["feedback_action"] = "increase_flow"
            else:
                best["feedback_action"] = "hold"
        else:
            best["feedback_action"] = "none"

        return best

    def update_outlet_feedback(self, top_camera_summary):
        self.previous_top_cu_fraction = float(
            top_camera_summary.get("copper_fraction", 0.0)
        )
