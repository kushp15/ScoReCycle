
"""Stage-aware closed-loop controller for the hackathon proof of concept.

Control philosophy
------------------
Stage 2:
    Make a valuable Al-rich TOP product.
    Slowly increase high-pulse flow only while Al removal improves and Cu
    contamination of the top product remains acceptably low.

Stage 3:
    Clean the Stage-2 HEAVY product.
    Make a valuable Cu-rich BOTTOM product. More Cu loss to the top is allowed
    than in Stage 2 if that materially improves bottom Cu purity.

The ML surrogate is used as a local decision aid, not as the plant.
The DynamicSeparatorSimulation remains the ground-truth digital twin.

The controller deliberately changes only high-pulse volumetric flow in this
demo. Pulse frequency and duty are held fixed so the judges can see exactly
what feedback is changing.
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


class StageAwareAdaptiveController:
    """Slow, constrained, camera-informed airflow controller."""

    def __init__(
        self,
        stage,
        surrogate,
        suspension_lookup,
        initial_flow_m3_h,
        low_flow_fraction=0.35,
        duty=0.50,
        flow_step_m3_h=1.0,
        min_flow_m3_h=None,
        max_flow_m3_h=None,
        stage2_top_cu_limit=0.06,
        stage3_top_cu_soft_limit=0.30,
    ):
        if stage not in {2, 3}:
            raise ValueError("stage must be 2 or 3")

        self.stage = int(stage)
        self.surrogate = surrogate
        self.suspension_lookup = suspension_lookup

        self.current_flow_m3_h = float(initial_flow_m3_h)
        self.initial_flow_m3_h = float(initial_flow_m3_h)
        self.low_flow_fraction = float(low_flow_fraction)
        self.duty = float(duty)
        self.flow_step_m3_h = float(flow_step_m3_h)

        self.min_flow_m3_h = float(
            min_flow_m3_h
            if min_flow_m3_h is not None
            else max(35.0, initial_flow_m3_h - 15.0)
        )
        self.max_flow_m3_h = float(
            max_flow_m3_h
            if max_flow_m3_h is not None
            else initial_flow_m3_h + 20.0
        )

        # Stage 2 is intentionally conservative about losing Cu to the Al
        # product. Stage 3 can tolerate more Cu loss in exchange for a cleaner
        # bottom Cu product.
        self.stage2_top_cu_limit = float(stage2_top_cu_limit)
        self.stage3_top_cu_soft_limit = float(stage3_top_cu_soft_limit)

        self.feed_particles_est = pd.DataFrame()
        self.previous_bottom_cu_purity = None
        self.previous_top_al_purity = None

    def _estimate_us(self, camera_material, size_mm):
        table = self.suspension_lookup.get(str(camera_material))
        if table is None or len(table) == 0:
            return 1.4
        sizes = table["size_mm"].to_numpy(dtype=float)
        us = table["mean_us"].to_numpy(dtype=float)
        idx = int(np.argmin(np.abs(sizes - float(size_mm))))
        return float(us[idx])

    def set_feed_observation(self, camera_obs):
        """Store only camera-observable state for local ML predictions."""
        if camera_obs.empty:
            self.feed_particles_est = pd.DataFrame(
                columns=["material_est", "size_mm", "u_s_est"]
            )
            return

        p = pd.DataFrame(
            {
                "material_est": camera_obs["camera_material"].astype(str),
                "size_mm": camera_obs["camera_size_mm"].astype(float),
            }
        )
        p["u_s_est"] = [
            self._estimate_us(m, s)
            for m, s in zip(p["material_est"], p["size_mm"])
        ]
        self.feed_particles_est = p

    def _predicted_metrics(self, flow_m3_h):
        """Use the learned surrogate to predict outlet probabilities."""
        p = self.feed_particles_est
        if p.empty:
            return {
                "al_recovery_top": 0.0,
                "al_purity_top": 0.0,
                "cu_recovery_bottom": 0.0,
                "cu_purity_bottom": 0.0,
                "cu_fraction_top": 0.0,
                "unresolved": 1.0,
            }

        X = pd.DataFrame(
            {
                "u_s": p["u_s_est"],
                "size_mm": p["size_mm"],
                "flow_velocity": flow_to_velocity(flow_m3_h),
                "duty": self.duty,
                "low_flow_fraction": self.low_flow_fraction,
            }
        )

        probs = self.surrogate.predict_proba(X)
        classes = list(self.surrogate.classes_)

        def prob(name):
            if name not in classes:
                return np.zeros(len(X))
            return probs[:, classes.index(name)]

        p_light = prob("light")
        p_heavy = prob("heavy")
        p_active = prob("active")

        mat = p["material_est"].to_numpy()
        al = mat == "aluminum"
        cu = mat == "copper"

        def mean_or_zero(x):
            return float(np.mean(x)) if len(x) else 0.0

        al_recovery_top = mean_or_zero(p_light[al])
        cu_recovery_bottom = mean_or_zero(p_heavy[cu])

        expected_top = float(np.sum(p_light))
        expected_bottom = float(np.sum(p_heavy))
        al_top = float(np.sum(p_light[al]))
        cu_top = float(np.sum(p_light[cu]))
        cu_bottom = float(np.sum(p_heavy[cu]))

        return {
            "al_recovery_top": al_recovery_top,
            "al_purity_top": (
                al_top / expected_top if expected_top > 1e-9 else 0.0
            ),
            "cu_recovery_bottom": cu_recovery_bottom,
            "cu_purity_bottom": (
                cu_bottom / expected_bottom if expected_bottom > 1e-9 else 0.0
            ),
            "cu_fraction_top": (
                cu_top / expected_top if expected_top > 1e-9 else 0.0
            ),
            "unresolved": float(np.mean(p_active)),
        }

    def _bounded(self, q):
        return float(
            np.clip(q, self.min_flow_m3_h, self.max_flow_m3_h)
        )

    def decide(
        self,
        top_camera_summary,
        bottom_camera_summary,
    ):
        """Return a small flow-rate adjustment.

        Camera summaries are from particles that exited during the most recent
        control interval when enough particles are available; otherwise they
        can be cumulative summaries.

        The feedback rule is deliberately interpretable for the hackathon.
        """
        q = self.current_flow_m3_h
        step = self.flow_step_m3_h

        top_cu = float(top_camera_summary.get("copper_fraction", 0.0))
        top_al = float(top_camera_summary.get("aluminum_fraction", 0.0))
        bottom_cu = float(bottom_camera_summary.get("copper_fraction", 0.0))
        bottom_al = float(bottom_camera_summary.get("aluminum_fraction", 0.0))

        pred_now = self._predicted_metrics(q)
        pred_up = self._predicted_metrics(self._bounded(q + step))
        pred_down = self._predicted_metrics(self._bounded(q - step))

        reason = "hold"
        q_next = q

        if self.stage == 2:
            # Hard priority: protect Cu from reporting to the Al top product.
            if top_cu > self.stage2_top_cu_limit:
                q_next = self._bounded(q - 2.0 * step)
                reason = "Cu detected in Stage-2 top -> decrease flow"
            else:
                # Residual Al in the bottom says there is still Al available to
                # recover. The ML surrogate must also predict that the next
                # small increase does not materially worsen top Cu carryover.
                more_al_available = bottom_al > 0.08

                # ML is used as a local risk indicator, but outlet-camera
                # feedback has priority. We do not let an imperfect surrogate
                # prevent the controller from cautiously exploring one small
                # step when real cameras say Cu is still protected.
                ml_warns_cu_risk = (
                    pred_up["cu_fraction_top"]
                    > max(
                        0.18,
                        pred_now["cu_fraction_top"] + 0.08,
                    )
                )

                if (
                    more_al_available
                    and top_cu < self.stage2_top_cu_limit
                    and not ml_warns_cu_risk
                ):
                    q_next = self._bounded(q + step)
                    reason = "Al remains retained and Cu top is clean -> increase"
                elif top_al > 0.80 and top_cu < 0.5 * self.stage2_top_cu_limit:
                    q_next = self._bounded(q + step)
                    reason = "Clean Al top product -> cautiously increase"
                else:
                    reason = "Stage-2 Cu-protection boundary -> hold"

            self.previous_top_al_purity = top_al

        else:
            # Stage 3 objective: clean the bottom Cu product. Some Cu is allowed
            # to leave with the top stream, but not without limit.
            if top_cu > self.stage3_top_cu_soft_limit:
                q_next = self._bounded(q - step)
                reason = "Stage-3 Cu loss too high -> decrease"
            else:
                bottom_needs_cleaning = bottom_al > 0.06 or bottom_cu < 0.94

                # Stage 3 is intentionally more aggressive. The bottom camera
                # is the primary optimization signal: if Al remains in the
                # retained Cu-rich product and Cu loss at the top is still
                # tolerable, keep walking the flow upward one small step.
                #
                # The ML surrogate still supplies predicted metrics for
                # monitoring and future model-predictive control, but it is a
                # soft advisory signal here because Stage-3 feed is a selected
                # Stage-2 heavy population and can differ from the surrogate's
                # original training distribution.
                if (
                    bottom_needs_cleaning
                    and top_cu < self.stage3_top_cu_soft_limit
                ):
                    q_next = self._bounded(q + step)
                    reason = "Residual Al in Cu-rich bottom -> increase one step"
                elif (
                    self.previous_bottom_cu_purity is not None
                    and bottom_cu > self.previous_bottom_cu_purity + 0.005
                    and top_cu < self.stage3_top_cu_soft_limit
                ):
                    q_next = self._bounded(q + step)
                    reason = "Bottom Cu purity improving -> continue one step"
                else:
                    reason = "Stage-3 purity/loss boundary -> hold"

            self.previous_bottom_cu_purity = bottom_cu

        self.current_flow_m3_h = q_next

        chosen_pred = self._predicted_metrics(q_next)

        return {
            "flow_m3_h": q_next,
            "delta_flow_m3_h": q_next - q,
            "reason": reason,
            "camera_top_cu_fraction": top_cu,
            "camera_top_al_fraction": top_al,
            "camera_bottom_cu_fraction": bottom_cu,
            "camera_bottom_al_fraction": bottom_al,
            **{f"pred_{k}": v for k, v in chosen_pred.items()},
        }
