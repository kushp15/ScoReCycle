
"""
SIDE-BY-SIDE TIME-RESOLVED COMPARISON

Rows:
    0.25-1.00 mm
    1.00-1.50 mm
    1.50-2.00 mm

Columns:
    LEFT  = fixed separator control
    RIGHT = camera-informed closed-loop ML control

The exact same changing true feed batch is supplied to both columns in every
control window. The adaptive side receives only a noisy synthetic camera
observation. It chooses high-pulse airflow and duty cycle using the ML
surrogate, then receives top-stream camera feedback for the next window.

This animation is designed to answer visually:
    Does real-time knowledge of changing feed composition improve separation?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from config import (
    PNEUMATIC_FRACTION_ORDER,
    RANDOM_SEED,
    ZHU_PRIMARY_METAL_FLOW_M3_H,
)
from dynamic_config import (
    TOTAL_HEIGHT_M,
    MATERIAL_MARKERS,
    PARTICLE_VISUAL_MIN_PT2,
    PARTICLE_VISUAL_MAX_PT2,
    PARTICLE_VISUAL_POWER,
)
from dynamic_separator import (
    DynamicSettings,
    DynamicSeparatorSimulation,
    radius_at_height,
)
from feed_model import generate_feed
from camera_model import SyntheticCamera
from controller import AdaptiveController, flow_to_velocity
from run_closed_loop_experiment import (
    build_suspension_lookup,
    calibrate_surrogate,
    make_changing_windows,
    metal_pool_for_fraction,
    stream_metrics,
)


OUTPUT_DIR = Path("outputs_side_by_side")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Animate all three size classes side-by-side: fixed control "
            "versus camera-informed closed-loop ML control."
        )
    )
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--particles-per-window", type=int, default=120)
    p.add_argument("--camera-sample", type=int, default=90)
    p.add_argument("--frequency", type=float, default=4.0)
    p.add_argument("--low-flow-fraction", type=float, default=0.35)
    p.add_argument("--window-duration", type=float, default=4.5)
    p.add_argument("--dt", type=float, default=0.008)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use the faster surrogate-calibration grid.",
    )
    p.add_argument(
        "--save-gif",
        action="store_true",
        help="Save the synchronized comparison as a GIF.",
    )
    p.add_argument(
        "--gif-name",
        default="fixed_vs_closed_loop_ml.gif",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive matplotlib window.",
    )
    return p.parse_args()


class ArgsProxy:
    """Arguments needed by calibrate_surrogate()."""
    def __init__(self, args):
        self.quick = args.quick
        self.frequency = args.frequency
        self.low_flow_fraction = args.low_flow_fraction
        self.duration = args.window_duration
        self.dt = args.dt


def draw_separator(ax):
    z = np.linspace(0.0, TOTAL_HEIGHT_M, 300)
    r = radius_at_height(z)
    ax.plot(-r, z, linewidth=1.2)
    ax.plot(r, z, linewidth=1.2)
    max_r = float(np.max(r))
    ax.set_xlim(-max_r * 1.40, max_r * 1.40)
    ax.set_ylim(-0.03, TOTAL_HEIGHT_M + 0.03)
    ax.set_xticks([])
    ax.set_ylabel("Height (m)")


def marker_areas(particles):
    size = particles["size_mm"].to_numpy(dtype=float)
    lo = float(size.min())
    hi = float(size.max())
    if hi > lo:
        frac = (size - lo) / (hi - lo)
    else:
        frac = np.ones_like(size)
    return (
        PARTICLE_VISUAL_MIN_PT2
        + (PARTICLE_VISUAL_MAX_PT2 - PARTICLE_VISUAL_MIN_PT2)
        * frac**PARTICLE_VISUAL_POWER
    )


def make_sim(feed, flow_m3_h, duty, args, seed):
    settings = DynamicSettings(
        high_velocity_m_s=flow_to_velocity(flow_m3_h),
        low_flow_fraction=args.low_flow_fraction,
        pulse_frequency_hz=args.frequency,
        duty_cycle=duty,
        dt_s=args.dt,
        duration_s=args.window_duration,
    )
    return DynamicSeparatorSimulation(
        feed,
        settings=settings,
        n_particles=len(feed),
        seed=seed,
    )


class FractionComparison:
    def __init__(
        self,
        fraction,
        pool,
        surrogate,
        lookup,
        windows,
        args,
        seed,
    ):
        self.fraction = fraction
        self.pool = pool
        self.surrogate = surrogate
        self.lookup = lookup
        self.windows = windows
        self.args = args
        self.seed = seed
        self.window_index = -1

        self.nominal_flow = ZHU_PRIMARY_METAL_FLOW_M3_H[fraction]

        self.feed_camera = SyntheticCamera(
            relative_size_sd=0.04,
            seed=seed + 101,
        )
        self.top_camera = SyntheticCamera(
            relative_size_sd=0.04,
            seed=seed + 202,
        )
        self.controller = AdaptiveController(
            fraction=fraction,
            surrogate=surrogate,
            suspension_lookup=lookup,
            nominal_flow_m3_h=self.nominal_flow,
            frequency_hz=args.frequency,
            low_flow_fraction=args.low_flow_fraction,
        )

        self.fixed_sim = None
        self.ml_sim = None
        self.ml_action = None
        self.camera_summary = None

        self.fixed_metrics = []
        self.ml_metrics = []
        self.finished = False

        # Cumulative outlet totals from COMPLETED windows.
        self.fixed_completed_light = 0
        self.fixed_completed_heavy = 0
        self.ml_completed_light = 0
        self.ml_completed_heavy = 0

        # Continuous histories used by the new comparison graphs.
        self.history_time = []
        self.history_fixed_air_velocity = []
        self.history_ml_air_velocity = []
        self.history_fixed_light = []
        self.history_fixed_heavy = []
        self.history_ml_light = []
        self.history_ml_heavy = []

        self.start_next_window()
        self.record_history()

    def start_next_window(self):
        self.window_index += 1
        if self.window_index >= len(self.windows):
            self.finished = True
            return

        true_feed = self.windows[self.window_index].copy()

        # Noisy feed camera is available only to the adaptive controller.
        obs = self.feed_camera.observe_particles(
            true_feed,
            max_particles=self.args.camera_sample,
        )
        self.camera_summary = self.feed_camera.summarize(obs)
        self.ml_action = self.controller.choose_action(obs)

        # Same true particles and same random initialization seed on both sides.
        # Only the control action differs.
        sim_seed = self.seed + 1000 + self.window_index

        self.fixed_sim = make_sim(
            true_feed,
            flow_m3_h=self.nominal_flow,
            duty=0.50,
            args=self.args,
            seed=sim_seed,
        )
        self.ml_sim = make_sim(
            true_feed,
            flow_m3_h=self.ml_action["flow_m3_h"],
            duty=self.ml_action["duty"],
            args=self.args,
            seed=sim_seed,
        )

    def local_finished(self, sim):
        return (
            sim.time_s >= self.args.window_duration
            or not np.any(sim.active_mask)
        )

    def record_history(self):
        """Record actual filtered pulse velocity and cumulative outlet counts."""
        if self.fixed_sim is None or self.ml_sim is None:
            return

        global_time = (
            self.window_index * self.args.window_duration
            + max(self.fixed_sim.time_s, self.ml_sim.time_s)
        )

        area_ref = math.pi * 0.100**2 / 4.0
        fixed_air = self.fixed_sim.flow_rate_m3_s / area_ref
        ml_air = self.ml_sim.flow_rate_m3_s / area_ref

        fixed_light = (
            self.fixed_completed_light
            + int(np.sum(self.fixed_sim.light_mask))
        )
        fixed_heavy = (
            self.fixed_completed_heavy
            + int(np.sum(self.fixed_sim.heavy_mask))
        )
        ml_light = (
            self.ml_completed_light
            + int(np.sum(self.ml_sim.light_mask))
        )
        ml_heavy = (
            self.ml_completed_heavy
            + int(np.sum(self.ml_sim.heavy_mask))
        )

        # Avoid duplicate timestamps during a frame with multiple numerical
        # substeps; the latest state at a time point is the useful graph point.
        if self.history_time and global_time <= self.history_time[-1]:
            self.history_time[-1] = global_time
            self.history_fixed_air_velocity[-1] = fixed_air
            self.history_ml_air_velocity[-1] = ml_air
            self.history_fixed_light[-1] = fixed_light
            self.history_fixed_heavy[-1] = fixed_heavy
            self.history_ml_light[-1] = ml_light
            self.history_ml_heavy[-1] = ml_heavy
        else:
            self.history_time.append(global_time)
            self.history_fixed_air_velocity.append(fixed_air)
            self.history_ml_air_velocity.append(ml_air)
            self.history_fixed_light.append(fixed_light)
            self.history_fixed_heavy.append(fixed_heavy)
            self.history_ml_light.append(ml_light)
            self.history_ml_heavy.append(ml_heavy)

    def finalize_window(self):
        fixed_result = self.fixed_sim.results()
        ml_result = self.ml_sim.results()

        fixed_m = stream_metrics(fixed_result)
        ml_m = stream_metrics(ml_result)

        fixed_m["window"] = self.window_index + 1
        fixed_m["flow_m3_h"] = self.nominal_flow
        fixed_m["duty"] = 0.50

        ml_m["window"] = self.window_index + 1
        ml_m["flow_m3_h"] = self.ml_action["flow_m3_h"]
        ml_m["duty"] = self.ml_action["duty"]

        self.fixed_metrics.append(fixed_m)
        self.ml_metrics.append(ml_m)

        # Freeze this window's outlet counts into the cumulative totals BEFORE
        # a fresh batch is loaded into the next control window.
        self.fixed_completed_light += int(np.sum(self.fixed_sim.light_mask))
        self.fixed_completed_heavy += int(np.sum(self.fixed_sim.heavy_mask))
        self.ml_completed_light += int(np.sum(self.ml_sim.light_mask))
        self.ml_completed_heavy += int(np.sum(self.ml_sim.heavy_mask))

        # Top-stream camera closes the feedback loop for the NEXT window.
        top = ml_result[ml_result["dynamic_stream"].eq("light")]
        top_obs = self.top_camera.observe_particles(
            top,
            max_particles=120,
        )
        top_summary = self.top_camera.summarize(top_obs)
        self.controller.update_outlet_feedback(top_summary)

    def step(self, substeps):
        if self.finished:
            return

        for _ in range(substeps):
            if not self.local_finished(self.fixed_sim):
                self.fixed_sim.step()
            if not self.local_finished(self.ml_sim):
                self.ml_sim.step()
            self.record_history()

        if (
            self.local_finished(self.fixed_sim)
            and self.local_finished(self.ml_sim)
        ):
            # Record the terminal state of this window before resetting.
            self.record_history()
            self.finalize_window()
            self.start_next_window()
            if not self.finished:
                self.record_history()

    @property
    def elapsed_global_s(self):
        if self.finished:
            return len(self.windows) * self.args.window_duration
        return (
            self.window_index * self.args.window_duration
            + max(self.fixed_sim.time_s, self.ml_sim.time_s)
        )


def configure_scatter(ax, sim):
    areas = marker_areas(sim.particles)
    out = {}
    for material in ["aluminum", "copper"]:
        mask = sim.particles["material"].eq(material).to_numpy()
        out[material] = ax.scatter(
            sim.x[mask],
            sim.z[mask],
            s=areas[mask],
            marker=MATERIAL_MARKERS.get(material, "s"),
            alpha=0.72,
            label=material,
        )
    return out


def update_scatter(scatter_map, sim):
    areas = marker_areas(sim.particles)
    for material, artist in scatter_map.items():
        mask = sim.particles["material"].eq(material).to_numpy()
        artist.set_offsets(
            np.column_stack(
                [
                    sim.x[mask],
                    sim.z[mask],
                ]
            )
        )
        # Composition changes between control windows, so the number of
        # Al/Cu points can change. Update marker sizes at the same time.
        artist.set_sizes(areas[mask])


def cumulative_mean(metrics, key):
    if not metrics:
        return np.nan
    return float(np.mean([m[key] for m in metrics]))


def build_experiment(args):
    full_feed = generate_feed()
    rng = np.random.default_rng(RANDOM_SEED)
    proxy = ArgsProxy(args)

    comparisons = {}

    for i, fraction in enumerate(PNEUMATIC_FRACTION_ORDER):
        print()
        print("=" * 82)
        print(f"Preparing {fraction} mm comparison")
        print("=" * 82)

        pool = metal_pool_for_fraction(full_feed, fraction)
        lookup = build_suspension_lookup(pool)

        surrogate, _, acc = calibrate_surrogate(
            pool,
            fraction,
            proxy,
            rng,
        )
        print(
            f"ML surrogate held-out trajectory accuracy: {100*acc:.1f}%"
        )

        windows = make_changing_windows(
            pool,
            args.windows,
            args.particles_per_window,
            rng,
        )

        comparisons[fraction] = FractionComparison(
            fraction=fraction,
            pool=pool,
            surrogate=surrogate,
            lookup=lookup,
            windows=windows,
            args=args,
            seed=RANDOM_SEED + 10000 * i,
        )

    return comparisons


def save_summary(comparisons):
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = []
    for fraction, comp in comparisons.items():
        fixed = pd.DataFrame(comp.fixed_metrics)
        ml = pd.DataFrame(comp.ml_metrics)

        fixed.to_csv(
            OUTPUT_DIR / f"{fraction}_fixed_windows.csv",
            index=False,
        )
        ml.to_csv(
            OUTPUT_DIR / f"{fraction}_closed_loop_ml_windows.csv",
            index=False,
        )

        fixed_obj = (
            float(fixed["objective"].mean())
            if len(fixed)
            else np.nan
        )
        ml_obj = (
            float(ml["objective"].mean())
            if len(ml)
            else np.nan
        )

        rows.append(
            {
                "fraction_mm": fraction,
                "fixed_mean_objective": fixed_obj,
                "ml_mean_objective": ml_obj,
                "relative_change_percent": (
                    100.0 * (ml_obj - fixed_obj) / abs(fixed_obj)
                    if np.isfinite(fixed_obj) and abs(fixed_obj) > 1e-12
                    else np.nan
                ),
                "fixed_mean_al_recovery": (
                    fixed["al_recovery"].mean() if len(fixed) else np.nan
                ),
                "ml_mean_al_recovery": (
                    ml["al_recovery"].mean() if len(ml) else np.nan
                ),
                "fixed_mean_cu_recovery": (
                    fixed["cu_recovery"].mean() if len(fixed) else np.nan
                ),
                "ml_mean_cu_recovery": (
                    ml["cu_recovery"].mean() if len(ml) else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "side_by_side_summary.csv",
        index=False,
    )

    print()
    print("=" * 82)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 82)
    print(summary.to_string(index=False))
    return summary


def animate(comparisons, args):
    """
    Four-column comparison for every size class:

        1. fixed-control particle trajectories
        2. closed-loop ML particle trajectories
        3. actual pulsated reference airflow velocity, fixed vs ML
        4. cumulative outlet accumulation, fixed vs ML
    """
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(19.0, 12.5),
        sharex=False,
        sharey=False,
        gridspec_kw={
            "width_ratios": [0.85, 0.85, 1.35, 1.45],
        },
    )

    fixed_scatter = {}
    ml_scatter = {}
    info_text = {}

    airflow_lines = {}
    outlet_lines = {}

    total_duration = args.windows * args.window_duration
    max_outlets = args.windows * args.particles_per_window

    for row, fraction in enumerate(PNEUMATIC_FRACTION_ORDER):
        comp = comparisons[fraction]

        ax_fixed = axes[row, 0]
        ax_ml = axes[row, 1]
        ax_air = axes[row, 2]
        ax_out = axes[row, 3]

        # ----------------------------
        # Particle trajectory panels
        # ----------------------------
        draw_separator(ax_fixed)
        draw_separator(ax_ml)

        ax_fixed.set_title(
            f"{fraction} mm | FIXED"
        )
        ax_ml.set_title(
            f"{fraction} mm | CLOSED-LOOP ML"
        )

        fixed_scatter[fraction] = configure_scatter(
            ax_fixed,
            comp.fixed_sim,
        )
        ml_scatter[fraction] = configure_scatter(
            ax_ml,
            comp.ml_sim,
        )

        info_text[(fraction, "fixed")] = ax_fixed.text(
            0.02,
            0.985,
            "",
            transform=ax_fixed.transAxes,
            va="top",
            fontsize=8.0,
        )
        info_text[(fraction, "ml")] = ax_ml.text(
            0.02,
            0.985,
            "",
            transform=ax_ml.transAxes,
            va="top",
            fontsize=8.0,
        )

        if row == 0:
            ax_fixed.legend(
                loc="upper right",
                fontsize=7.5,
            )
            ax_ml.legend(
                loc="upper right",
                fontsize=7.5,
            )

        # ----------------------------
        # Pulsated airflow comparison
        # ----------------------------
        ax_air.set_title(
            f"{fraction} mm | Pulsated airflow"
        )
        ax_air.set_xlim(0.0, total_duration)
        ax_air.set_ylim(
            0.0,
            max(
                flow_to_velocity(
                    ZHU_PRIMARY_METAL_FLOW_M3_H[fraction] + 25.0
                ),
                1.0,
            ),
        )
        ax_air.set_xlabel("Global time (s)")
        ax_air.set_ylabel(
            "Reference air velocity (m/s)"
        )
        ax_air.grid(alpha=0.25)

        fixed_air_line, = ax_air.plot(
            [],
            [],
            linewidth=1.4,
            label="Fixed",
        )
        ml_air_line, = ax_air.plot(
            [],
            [],
            linewidth=1.4,
            label="Closed-loop ML",
        )
        airflow_lines[fraction] = (
            fixed_air_line,
            ml_air_line,
        )
        ax_air.legend(
            loc="upper right",
            fontsize=7.5,
        )

        # Mark control-window boundaries.
        for boundary in range(1, args.windows):
            ax_air.axvline(
                boundary * args.window_duration,
                linestyle=":",
                linewidth=0.8,
                alpha=0.35,
            )

        # ----------------------------
        # Outlet accumulation comparison
        # ----------------------------
        ax_out.set_title(
            f"{fraction} mm | Cumulative outlet accumulation"
        )
        ax_out.set_xlim(0.0, total_duration)
        ax_out.set_ylim(0, max_outlets)
        ax_out.set_xlabel("Global time (s)")
        ax_out.set_ylabel("Cumulative parcels")
        ax_out.grid(alpha=0.25)

        fixed_top, = ax_out.plot(
            [],
            [],
            linewidth=1.5,
            label="Fixed top/light",
        )
        fixed_bottom, = ax_out.plot(
            [],
            [],
            linewidth=1.5,
            linestyle="--",
            label="Fixed bottom/heavy",
        )
        ml_top, = ax_out.plot(
            [],
            [],
            linewidth=1.5,
            label="ML top/light",
        )
        ml_bottom, = ax_out.plot(
            [],
            [],
            linewidth=1.5,
            linestyle="--",
            label="ML bottom/heavy",
        )

        outlet_lines[fraction] = (
            fixed_top,
            fixed_bottom,
            ml_top,
            ml_bottom,
        )
        ax_out.legend(
            loc="upper left",
            fontsize=7.0,
        )

        for boundary in range(1, args.windows):
            ax_out.axvline(
                boundary * args.window_duration,
                linestyle=":",
                linewidth=0.8,
                alpha=0.35,
            )

    fig.suptitle(
        (
            "Fixed vs camera-informed closed-loop ML: "
            "particle motion, pulsated airflow, and outlet accumulation"
        ),
        fontsize=14,
    )
    fig.text(
        0.5,
        0.010,
        (
            "Within each row both separators receive the same changing true feed. "
            "Only the ML side uses camera estimates and outlet feedback to alter "
            "high-pulse airflow and duty cycle."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.015, 0.032, 1, 0.965])

    visual_dt = 1.0 / args.fps
    substeps = max(1, int(round(visual_dt / args.dt)))

    max_frames = int(
        math.ceil(total_duration / (substeps * args.dt))
        + 4 * args.windows
    )

    def update(_frame):
        artists = []

        for fraction in PNEUMATIC_FRACTION_ORDER:
            comp = comparisons[fraction]

            if not comp.finished:
                comp.step(substeps)

            if not comp.finished:
                update_scatter(
                    fixed_scatter[fraction],
                    comp.fixed_sim,
                )
                update_scatter(
                    ml_scatter[fraction],
                    comp.ml_sim,
                )

                fixed_obj = cumulative_mean(
                    comp.fixed_metrics,
                    "objective",
                )
                ml_obj = cumulative_mean(
                    comp.ml_metrics,
                    "objective",
                )

                ftxt = (
                    f"t = {comp.elapsed_global_s:5.1f} s\n"
                    f"window {comp.window_index + 1}/{args.windows}\n"
                    f"Q_high = {comp.nominal_flow:.0f} m3/h\n"
                    f"duty = 0.50\n"
                    f"top = {np.sum(comp.fixed_sim.light_mask):d}\n"
                    f"bottom = {np.sum(comp.fixed_sim.heavy_mask):d}"
                )
                if np.isfinite(fixed_obj):
                    ftxt += f"\nmean J = {fixed_obj:.3f}"

                mltxt = (
                    f"t = {comp.elapsed_global_s:5.1f} s\n"
                    f"window {comp.window_index + 1}/{args.windows}\n"
                    f"camera Al = {100*comp.camera_summary['aluminum_fraction']:.0f}%\n"
                    f"camera Cu = {100*comp.camera_summary['copper_fraction']:.0f}%\n"
                    f"Q_high = {comp.ml_action['flow_m3_h']:.0f} m3/h\n"
                    f"duty = {comp.ml_action['duty']:.2f}\n"
                    f"top = {np.sum(comp.ml_sim.light_mask):d}\n"
                    f"bottom = {np.sum(comp.ml_sim.heavy_mask):d}"
                )
                if np.isfinite(ml_obj):
                    mltxt += f"\nmean J = {ml_obj:.3f}"

                info_text[(fraction, "fixed")].set_text(ftxt)
                info_text[(fraction, "ml")].set_text(mltxt)

            else:
                fixed_obj = cumulative_mean(
                    comp.fixed_metrics,
                    "objective",
                )
                ml_obj = cumulative_mean(
                    comp.ml_metrics,
                    "objective",
                )
                delta = (
                    100.0 * (ml_obj - fixed_obj) / abs(fixed_obj)
                    if np.isfinite(fixed_obj) and abs(fixed_obj) > 1e-12
                    else np.nan
                )

                info_text[(fraction, "fixed")].set_text(
                    f"COMPLETE\nmean J = {fixed_obj:.3f}"
                )
                info_text[(fraction, "ml")].set_text(
                    f"COMPLETE\nmean J = {ml_obj:.3f}\n"
                    f"change = {delta:+.1f}%"
                )

            # Update pulsated-airflow histories.
            fixed_air_line, ml_air_line = airflow_lines[fraction]
            fixed_air_line.set_data(
                comp.history_time,
                comp.history_fixed_air_velocity,
            )
            ml_air_line.set_data(
                comp.history_time,
                comp.history_ml_air_velocity,
            )

            # Update cumulative top/bottom outlet histories.
            fixed_top, fixed_bottom, ml_top, ml_bottom = outlet_lines[fraction]
            fixed_top.set_data(
                comp.history_time,
                comp.history_fixed_light,
            )
            fixed_bottom.set_data(
                comp.history_time,
                comp.history_fixed_heavy,
            )
            ml_top.set_data(
                comp.history_time,
                comp.history_ml_light,
            )
            ml_bottom.set_data(
                comp.history_time,
                comp.history_ml_heavy,
            )

            artists += list(fixed_scatter[fraction].values())
            artists += list(ml_scatter[fraction].values())
            artists += [
                info_text[(fraction, "fixed")],
                info_text[(fraction, "ml")],
                fixed_air_line,
                ml_air_line,
                fixed_top,
                fixed_bottom,
                ml_top,
                ml_bottom,
            ]

        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )

    fig._comparison_animation = animation

    if args.save_gif:
        OUTPUT_DIR.mkdir(exist_ok=True)
        gif_path = OUTPUT_DIR / args.gif_name
        print()
        print(f"Saving GIF to {gif_path} ...")
        animation.save(
            gif_path,
            writer=PillowWriter(fps=args.fps),
            dpi=92,
        )
        print(f"Saved {gif_path}")

    if not args.no_show:
        plt.show()

    # Complete any remaining numerical windows so CSV summaries are complete.
    for comp in comparisons.values():
        while not comp.finished:
            comp.step(substeps)

    plt.close(fig)


def main():
    args = parse_args()
    comparisons = build_experiment(args)
    animate(comparisons, args)
    save_summary(comparisons)

    print()
    print(
        "Run complete. The left and right columns used the same changing "
        "true feed windows. The right column alone used noisy camera "
        "composition estimates and outlet feedback to change the control."
    )


if __name__ == "__main__":
    main()
