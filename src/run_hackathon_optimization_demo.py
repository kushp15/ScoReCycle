
"""Hackathon proof-of-concept: stage-aware pneumatic-separator optimization.

What this demonstrates
----------------------
For every Zhu particle-size class, compare an open-loop baseline against a
camera-informed, ML-assisted feedback controller through two sequential stages.

Stage 2 objective
    Produce a valuable Al-rich TOP product.
    Slowly adjust high-pulse volumetric flow.
    Cu appearing in the TOP product is treated as a strong constraint.

Stage 3 objective
    Feed the actual Stage-2 HEAVY product into another pneumatic pass.
    Produce a valuable Cu-rich BOTTOM product.
    Some Cu loss to the top is acceptable if bottom Cu purity improves.

Cameras
-------
Adaptive branch:
    - feed camera
    - top outlet camera
    - bottom outlet camera

The controller sees noisy synthetic camera measurements. True particle labels
are used only to evaluate/plot the digital-twin result, never to make the
control decision.

Machine learning
----------------
A HistGradientBoosting surrogate is trained from the existing dynamic
separator. The surrogate predicts particle outlet probabilities for small
candidate changes around the current operating point. The plant itself remains
DynamicSeparatorSimulation.

This is intentionally a proof of concept, not an experimentally validated
industrial controller.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from camera_model import SyntheticCamera
from config import (
    PNEUMATIC_FRACTION_ORDER,
    RANDOM_SEED,
    ZHU_PRIMARY_METAL_FLOW_M3_H,
    ZHU_STAGE3_CLEANING_FLOW_M3_H,
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
from run_closed_loop_experiment import (
    build_suspension_lookup,
    calibrate_surrogate,
    make_changing_windows,
    metal_pool_for_fraction,
)
from stage_aware_controller import (
    StageAwareAdaptiveController,
    flow_to_velocity,
)


OUTPUT_DIR = Path("outputs_hackathon_demo")

TRAJECTORY_COLUMNS = [
    "x_m",
    "z_m",
    "vx_m_s",
    "vz_m_s",
    "dynamic_stream",
    "exit_time_s",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--particles", type=int, default=110)
    p.add_argument("--camera-sample", type=int, default=90)
    p.add_argument("--frequency", type=float, default=4.0)
    p.add_argument("--duty", type=float, default=0.50)
    p.add_argument("--low-flow-fraction", type=float, default=0.35)
    p.add_argument("--dt", type=float, default=0.008)
    p.add_argument("--stage2-duration", type=float, default=8.0)
    p.add_argument("--stage3-duration", type=float, default=8.0)
    p.add_argument("--control-interval", type=float, default=0.75)
    p.add_argument("--flow-step", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--gif-name",
        default="hackathon_stage2_stage3_optimization.gif",
    )
    return p.parse_args()


class CalibrationArgs:
    def __init__(self, args):
        self.quick = args.quick
        self.frequency = args.frequency
        self.low_flow_fraction = args.low_flow_fraction
        self.duration = max(args.stage2_duration, args.stage3_duration)
        self.dt = args.dt


def clean_for_next_stage(df):
    return df.drop(
        columns=[c for c in TRAJECTORY_COLUMNS if c in df.columns],
        errors="ignore",
    ).reset_index(drop=True)


def sample_feed(pool, n, seed):
    if len(pool) <= n:
        return pool.copy().reset_index(drop=True)
    return pool.sample(
        n=n,
        replace=False,
        random_state=seed,
    ).reset_index(drop=True)


def make_settings(flow_m3_h, duration, args):
    return DynamicSettings(
        high_velocity_m_s=flow_to_velocity(flow_m3_h),
        low_flow_fraction=args.low_flow_fraction,
        pulse_frequency_hz=args.frequency,
        duty_cycle=args.duty,
        dt_s=args.dt,
        duration_s=duration,
    )


def stream_truth_metrics(result, stage):
    top = result[result["dynamic_stream"].eq("light")]

    # In this variable-diameter model, many dense particles remain suspended
    # or trapped around the expanded section rather than reaching z=0 during
    # the pulse window. Physically, when the air is shut off those retained
    # particles are discharged/collected as the heavy product.
    #
    # Therefore "bottom product" for Stage 2/3 means EVERYTHING NOT LOST
    # THROUGH THE TOP: active/retained + any particles already at the bottom.
    bottom = result[~result["dynamic_stream"].eq("light")]

    n_al = int(result["material"].eq("aluminum").sum())
    n_cu = int(result["material"].eq("copper").sum())

    top_al = int(top["material"].eq("aluminum").sum())
    top_cu = int(top["material"].eq("copper").sum())
    bottom_al = int(bottom["material"].eq("aluminum").sum())
    bottom_cu = int(bottom["material"].eq("copper").sum())

    al_top_purity = top_al / len(top) if len(top) else np.nan
    al_top_recovery = top_al / n_al if n_al else np.nan
    cu_bottom_purity = bottom_cu / len(bottom) if len(bottom) else np.nan
    cu_bottom_recovery = bottom_cu / n_cu if n_cu else np.nan
    cu_loss_top = top_cu / n_cu if n_cu else np.nan

    return {
        "top_count": len(top),
        "bottom_count": len(bottom),
        "al_top_purity": al_top_purity,
        "al_top_recovery": al_top_recovery,
        "cu_bottom_purity": cu_bottom_purity,
        "cu_bottom_recovery": cu_bottom_recovery,
        "cu_loss_top": cu_loss_top,
        "stage_metric_purity": (
            al_top_purity if stage == 2 else cu_bottom_purity
        ),
        "stage_metric_recovery": (
            al_top_recovery if stage == 2 else cu_bottom_recovery
        ),
    }


def outlet_camera_summary(
    camera,
    result,
    stream,
    since_time,
    max_particles,
):
    """Camera observation for a physical product region.

    TOP camera:
        preferentially observes particles that newly exited the top.

    BOTTOM camera:
        observes the retained heavy-side inventory (active + heavy). This is
        the material that will be discharged as the heavy product when airflow
        stops. That makes the bottom camera useful even while Cu is still held
        in the variable-diameter region.
    """
    if stream == "retained":
        retained = result[~result["dynamic_stream"].eq("light")]
        obs = camera.observe_particles(
            retained,
            max_particles=max_particles,
        )
        return camera.summarize(obs)

    outlet = result[result["dynamic_stream"].eq(stream)]

    recent = outlet[
        outlet["exit_time_s"].notna()
        & (outlet["exit_time_s"] > since_time)
    ]

    use = recent if len(recent) >= 4 else outlet
    obs = camera.observe_particles(
        use,
        max_particles=max_particles,
    )
    return camera.summarize(obs)


def snapshot(sim, stage, flow_setpoint_m3_h):
    result = sim.results()
    truth = stream_truth_metrics(result, stage)
    return {
        "time_s": float(sim.time_s),
        "x": sim.x.copy(),
        "z": sim.z.copy(),
        "status": sim.status.copy(),
        "flow_setpoint_m3_h": float(flow_setpoint_m3_h),
        "air_velocity_m_s": float(
            sim.history_reference_velocity[-1]
            if sim.history_reference_velocity
            else sim.settings.high_velocity_m_s
            * sim.settings.low_flow_fraction
        ),
        **truth,
    }


def run_fixed_stage(
    feed,
    stage,
    flow_m3_h,
    duration,
    args,
    seed,
):
    if feed.empty:
        empty = feed.copy()
        if "dynamic_stream" not in empty.columns:
            empty["dynamic_stream"] = pd.Series(dtype=object)
        if "exit_time_s" not in empty.columns:
            empty["exit_time_s"] = pd.Series(dtype=float)
        return {
            "result": empty,
            "frames": [],
            "history": pd.DataFrame(
                columns=[
                    "time_s",
                    "air_velocity_m_s",
                    "top_count",
                    "bottom_count",
                    "flow_setpoint_m3_h",
                ]
            ),
        }

    sim = DynamicSeparatorSimulation(
        feed,
        settings=make_settings(flow_m3_h, duration, args),
        n_particles=len(feed),
        seed=seed,
    )

    frame_dt = 1.0 / args.fps
    next_frame = 0.0
    frames = []

    while sim.time_s < duration and np.any(sim.active_mask):
        sim.step()
        if sim.time_s + 1e-12 >= next_frame:
            frames.append(snapshot(sim, stage, flow_m3_h))
            next_frame += frame_dt

    if not frames or frames[-1]["time_s"] < sim.time_s:
        frames.append(snapshot(sim, stage, flow_m3_h))

    hist = pd.DataFrame(
        {
            "time_s": sim.history_time,
            "air_velocity_m_s": sim.history_reference_velocity,
            "top_count": sim.history_light,
            "bottom_count": [
                len(feed) - n_light
                for n_light in sim.history_light
            ],
            "flow_setpoint_m3_h": flow_m3_h,
        }
    )

    return {
        "result": sim.results(),
        "frames": frames,
        "history": hist,
    }


def run_adaptive_stage(
    feed,
    stage,
    surrogate,
    lookup,
    initial_flow_m3_h,
    duration,
    args,
    seed,
):
    if feed.empty:
        empty = feed.copy()
        if "dynamic_stream" not in empty.columns:
            empty["dynamic_stream"] = pd.Series(dtype=object)
        if "exit_time_s" not in empty.columns:
            empty["exit_time_s"] = pd.Series(dtype=float)
        return {
            "result": empty,
            "frames": [],
            "history": pd.DataFrame(
                columns=[
                    "time_s",
                    "air_velocity_m_s",
                    "top_count",
                    "bottom_count",
                    "flow_setpoint_m3_h",
                ]
            ),
            "decisions": pd.DataFrame(
                columns=[
                    "time_s",
                    "stage",
                    "flow_m3_h",
                    "delta_flow_m3_h",
                    "reason",
                ]
            ),
        }

    feed_camera = SyntheticCamera(seed=seed + 10)
    top_camera = SyntheticCamera(seed=seed + 20)
    bottom_camera = SyntheticCamera(seed=seed + 30)

    feed_obs = feed_camera.observe_particles(
        feed,
        max_particles=args.camera_sample,
    )

    controller = StageAwareAdaptiveController(
        stage=stage,
        surrogate=surrogate,
        suspension_lookup=lookup,
        initial_flow_m3_h=initial_flow_m3_h,
        low_flow_fraction=args.low_flow_fraction,
        duty=args.duty,
        flow_step_m3_h=args.flow_step,
    )
    controller.set_feed_observation(feed_obs)

    sim = DynamicSeparatorSimulation(
        feed,
        settings=make_settings(initial_flow_m3_h, duration, args),
        n_particles=len(feed),
        seed=seed,
    )

    current_flow = float(initial_flow_m3_h)
    frame_dt = 1.0 / args.fps
    next_frame = 0.0
    next_control = args.control_interval
    last_control_time = 0.0

    frames = []
    decisions = []

    # Initial decision record: the controller has seen the feed camera but has
    # not yet received outlet feedback.
    decisions.append(
        {
            "time_s": 0.0,
            "stage": stage,
            "flow_m3_h": current_flow,
            "delta_flow_m3_h": 0.0,
            "reason": "initial operating point",
        }
    )

    while sim.time_s < duration and np.any(sim.active_mask):
        sim.step()

        if sim.time_s + 1e-12 >= next_control:
            result_now = sim.results()

            top_summary = outlet_camera_summary(
                top_camera,
                result_now,
                "light",
                last_control_time,
                args.camera_sample,
            )
            bottom_summary = outlet_camera_summary(
                bottom_camera,
                result_now,
                "retained",
                last_control_time,
                args.camera_sample,
            )

            action = controller.decide(
                top_summary,
                bottom_summary,
            )
            current_flow = action["flow_m3_h"]

            # This is the only actuator changed in the proof of concept.
            sim.settings.high_velocity_m_s = flow_to_velocity(current_flow)

            decisions.append(
                {
                    "time_s": float(sim.time_s),
                    "stage": stage,
                    **action,
                }
            )

            last_control_time = float(sim.time_s)
            next_control += args.control_interval

        if sim.time_s + 1e-12 >= next_frame:
            frames.append(snapshot(sim, stage, current_flow))
            next_frame += frame_dt

    if not frames or frames[-1]["time_s"] < sim.time_s:
        frames.append(snapshot(sim, stage, current_flow))

    hist = pd.DataFrame(
        {
            "time_s": sim.history_time,
            "air_velocity_m_s": sim.history_reference_velocity,
            "top_count": sim.history_light,
            "bottom_count": [
                len(feed) - n_light
                for n_light in sim.history_light
            ],
        }
    )

    # Convert sparse control decisions into a stepwise setpoint history.
    decisions_df = pd.DataFrame(decisions)
    hist["flow_setpoint_m3_h"] = initial_flow_m3_h
    if len(decisions_df):
        j = 0
        current = initial_flow_m3_h
        vals = []
        for t in hist["time_s"]:
            while (
                j + 1 < len(decisions_df)
                and decisions_df.iloc[j + 1]["time_s"] <= t
            ):
                j += 1
                current = decisions_df.iloc[j]["flow_m3_h"]
            vals.append(current)
        hist["flow_setpoint_m3_h"] = vals

    return {
        "result": sim.results(),
        "frames": frames,
        "history": hist,
        "decisions": decisions_df,
    }


def final_stage_row(result, stage):
    m = stream_truth_metrics(result, stage)
    return {
        "feed_particles": len(result),
        "top_particles": m["top_count"],
        "bottom_particles": m["bottom_count"],
        "al_top_purity": m["al_top_purity"],
        "al_top_recovery": m["al_top_recovery"],
        "cu_bottom_purity": m["cu_bottom_purity"],
        "cu_bottom_recovery": m["cu_bottom_recovery"],
        "cu_loss_top": m["cu_loss_top"],
    }


def run_fraction(fraction, full_feed, surrogate, lookup, args, seed):
    pool = metal_pool_for_fraction(full_feed, fraction)
    feed = sample_feed(pool, args.particles, seed)

    # ---------------- STAGE 2 ----------------
    stage2_flow = ZHU_PRIMARY_METAL_FLOW_M3_H[fraction]

    fixed2 = run_fixed_stage(
        feed=feed,
        stage=2,
        flow_m3_h=stage2_flow,
        duration=args.stage2_duration,
        args=args,
        seed=seed + 100,
    )

    adaptive2 = run_adaptive_stage(
        feed=feed,
        stage=2,
        surrogate=surrogate,
        lookup=lookup,
        initial_flow_m3_h=stage2_flow,
        duration=args.stage2_duration,
        args=args,
        seed=seed + 100,  # same initial random state as fixed
    )

    # Actual Stage-2 heavy products become the corresponding Stage-3 feeds.
    fixed3_feed = clean_for_next_stage(
        fixed2["result"][
            ~fixed2["result"]["dynamic_stream"].eq("light")
        ]
    )
    adaptive3_feed = clean_for_next_stage(
        adaptive2["result"][
            ~adaptive2["result"]["dynamic_stream"].eq("light")
        ]
    )

    # ---------------- STAGE 3 ----------------
    # For the hackathon controller, Stage 3 begins at the same measured Zhu
    # reference flow used for the primary metal cut. The older 60/65/70 m3/h
    # Stage-3 values in config were only project design assumptions and proved
    # too weak in the dynamic model to strip material from the retained
    # variable-diameter population. A future version can predict this initial
    # Stage-3 setpoint directly from the Stage-2 bottom-camera state.
    stage3_flow = ZHU_PRIMARY_METAL_FLOW_M3_H[fraction]

    fixed3 = run_fixed_stage(
        feed=fixed3_feed,
        stage=3,
        flow_m3_h=stage3_flow,
        duration=args.stage3_duration,
        args=args,
        seed=seed + 300,
    )

    adaptive3 = run_adaptive_stage(
        feed=adaptive3_feed,
        stage=3,
        surrogate=surrogate,
        lookup=lookup,
        initial_flow_m3_h=stage3_flow,
        duration=args.stage3_duration,
        args=args,
        seed=seed + 300,
    )

    return {
        "fraction": fraction,
        "feed": feed,
        "fixed2": fixed2,
        "adaptive2": adaptive2,
        "fixed3": fixed3,
        "adaptive3": adaptive3,
    }


def marker_sizes(feed):
    if feed.empty:
        return np.array([])
    size = feed["size_mm"].to_numpy(dtype=float)
    lo = float(size.min())
    hi = float(size.max())
    if hi <= lo:
        f = np.ones(len(size))
    else:
        f = (size - lo) / (hi - lo)
    return (
        PARTICLE_VISUAL_MIN_PT2
        + (PARTICLE_VISUAL_MAX_PT2 - PARTICLE_VISUAL_MIN_PT2)
        * f**PARTICLE_VISUAL_POWER
    )


def draw_separator(ax):
    z = np.linspace(0.0, TOTAL_HEIGHT_M, 300)
    r = radius_at_height(z)
    ax.plot(-r, z, linewidth=1.0)
    ax.plot(r, z, linewidth=1.0)
    max_r = float(np.max(r))
    ax.set_xlim(-1.35 * max_r, 1.35 * max_r)
    ax.set_ylim(-0.03, TOTAL_HEIGHT_M + 0.03)
    ax.set_xticks([])
    ax.set_ylabel("Height (m)")


def line_history_for_stage(run, time_offset=0.0):
    h = run["history"].copy()
    if h.empty:
        return h
    h["global_time_s"] = h["time_s"] + time_offset
    return h


def quality_history_from_frames(frames, time_offset=0.0):
    if not frames:
        return pd.DataFrame(
            columns=[
                "global_time_s",
                "purity",
                "recovery",
            ]
        )
    return pd.DataFrame(
        {
            "global_time_s": [
                f["time_s"] + time_offset for f in frames
            ],
            "purity": [f["stage_metric_purity"] for f in frames],
            "recovery": [f["stage_metric_recovery"] for f in frames],
        }
    )


def get_frame(run_data, local_time):
    frames = run_data["frames"]
    if not frames:
        return None
    times = np.array([f["time_s"] for f in frames])
    idx = int(np.searchsorted(times, local_time, side="right") - 1)
    idx = int(np.clip(idx, 0, len(frames) - 1))
    return frames[idx]


def animate(all_results, args):
    stage_boundary = args.stage2_duration
    total_time = args.stage2_duration + args.stage3_duration

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(22, 12),
        gridspec_kw={
            "width_ratios": [0.80, 0.80, 1.25, 1.30, 1.25],
        },
    )

    particle_artists = {}
    info_text = {}
    air_lines = {}
    outlet_lines = {}
    quality_lines = {}

    for row, fraction in enumerate(PNEUMATIC_FRACTION_ORDER):
        result = all_results[fraction]
        ax_fixed, ax_ml, ax_air, ax_out, ax_quality = axes[row]

        draw_separator(ax_fixed)
        draw_separator(ax_ml)
        ax_fixed.set_title(f"{fraction} mm | Fixed")
        ax_ml.set_title(f"{fraction} mm | Closed-loop ML")

        # Stage 2 initial feed is shared, so initialize both with that feed.
        feed = result["feed"]
        sizes = marker_sizes(feed)

        particle_artists[(fraction, "fixed")] = {}
        particle_artists[(fraction, "ml")] = {}

        for branch, ax in [("fixed", ax_fixed), ("ml", ax_ml)]:
            for material in ["aluminum", "copper"]:
                mask = feed["material"].eq(material).to_numpy()
                art = ax.scatter(
                    np.zeros(np.sum(mask)),
                    np.zeros(np.sum(mask)),
                    s=sizes[mask],
                    marker=MATERIAL_MARKERS.get(material, "s"),
                    alpha=0.75,
                    label=material,
                )
                particle_artists[(fraction, branch)][material] = art

            info_text[(fraction, branch)] = ax.text(
                0.02,
                0.985,
                "",
                transform=ax.transAxes,
                va="top",
                fontsize=7.8,
            )

        if row == 0:
            ax_fixed.legend(loc="upper right", fontsize=7)
            ax_ml.legend(loc="upper right", fontsize=7)

        # Airflow comparison.
        ax_air.set_title("Pulsated air velocity")
        ax_air.set_xlim(0, total_time)
        ax_air.set_xlabel("Time (s)")
        ax_air.set_ylabel("Reference velocity (m/s)")
        ax_air.grid(alpha=0.25)
        ax_air.axvline(stage_boundary, linestyle=":", alpha=0.5)
        lf, = ax_air.plot([], [], linewidth=1.3, label="Fixed")
        lm, = ax_air.plot([], [], linewidth=1.3, label="Closed-loop ML")
        air_lines[fraction] = (lf, lm)
        ax_air.legend(fontsize=7)

        # Outlet accumulation comparison.
        ax_out.set_title("Top removal / retained heavy product")
        ax_out.set_xlim(0, total_time)
        ax_out.set_xlabel("Time (s)")
        ax_out.set_ylabel("Particles (resets at Stage 3)")
        ax_out.grid(alpha=0.25)
        ax_out.axvline(stage_boundary, linestyle=":", alpha=0.5)
        ftop, = ax_out.plot([], [], linewidth=1.2, label="Fixed top")
        fbottom, = ax_out.plot([], [], linewidth=1.2, linestyle="--", label="Fixed retained")
        mtop, = ax_out.plot([], [], linewidth=1.2, label="ML top")
        mbottom, = ax_out.plot([], [], linewidth=1.2, linestyle="--", label="ML retained")
        outlet_lines[fraction] = (ftop, fbottom, mtop, mbottom)
        ax_out.legend(fontsize=6.5)

        # Product quality comparison.
        ax_quality.set_title("Product quality objective")
        ax_quality.set_xlim(0, total_time)
        ax_quality.set_ylim(0, 1.02)
        ax_quality.set_xlabel("Time (s)")
        ax_quality.set_ylabel("Purity")
        ax_quality.grid(alpha=0.25)
        ax_quality.axvline(stage_boundary, linestyle=":", alpha=0.5)
        qf, = ax_quality.plot([], [], linewidth=1.4, label="Fixed purity")
        qm, = ax_quality.plot([], [], linewidth=1.4, label="ML purity")
        quality_lines[fraction] = (qf, qm)
        ax_quality.legend(fontsize=7)

        ax_quality.text(
            0.20,
            0.08,
            "Stage 2:\nAl purity in TOP",
            transform=ax_quality.transAxes,
            fontsize=7.5,
            ha="center",
        )
        ax_quality.text(
            0.77,
            0.08,
            "Stage 3:\nCu purity in BOTTOM",
            transform=ax_quality.transAxes,
            fontsize=7.5,
            ha="center",
        )

    fig.suptitle(
        "Camera-informed pneumatic separation optimization: fixed vs closed-loop ML",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.010,
        (
            "Stage 2 protects Cu while building an Al-rich top product. "
            "Stage 3 reprocesses the Stage-2 heavy stream to build a cleaner "
            "Cu bottom product. The dotted line is the Stage-2/Stage-3 handoff."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.01, 0.035, 1, 0.965])

    # Pre-build graph histories for efficient animation.
    graph_data = {}
    for fraction in PNEUMATIC_FRACTION_ORDER:
        r = all_results[fraction]

        fixed_air = pd.concat(
            [
                line_history_for_stage(r["fixed2"], 0.0),
                line_history_for_stage(r["fixed3"], stage_boundary),
            ],
            ignore_index=True,
        )
        ml_air = pd.concat(
            [
                line_history_for_stage(r["adaptive2"], 0.0),
                line_history_for_stage(r["adaptive3"], stage_boundary),
            ],
            ignore_index=True,
        )

        fixed_q = pd.concat(
            [
                quality_history_from_frames(r["fixed2"]["frames"], 0.0),
                quality_history_from_frames(
                    r["fixed3"]["frames"], stage_boundary
                ),
            ],
            ignore_index=True,
        )
        ml_q = pd.concat(
            [
                quality_history_from_frames(r["adaptive2"]["frames"], 0.0),
                quality_history_from_frames(
                    r["adaptive3"]["frames"], stage_boundary
                ),
            ],
            ignore_index=True,
        )

        graph_data[fraction] = {
            "fixed_air": fixed_air,
            "ml_air": ml_air,
            "fixed_quality": fixed_q,
            "ml_quality": ml_q,
        }

    n_frames = int(math.ceil(total_time * args.fps)) + 1

    def set_particles(fraction, branch, frame, feed_for_frame):
        arts = particle_artists[(fraction, branch)]
        if frame is None or feed_for_frame.empty:
            for a in arts.values():
                a.set_offsets(np.empty((0, 2)))
                a.set_sizes(np.array([]))
            return

        sizes = marker_sizes(feed_for_frame)
        for material, art in arts.items():
            mask = feed_for_frame["material"].eq(material).to_numpy()
            xy = np.column_stack(
                [frame["x"][mask], frame["z"][mask]]
            )
            art.set_offsets(xy)
            art.set_sizes(sizes[mask])

    def update(frame_i):
        t_global = min(frame_i / args.fps, total_time)
        stage = 2 if t_global < stage_boundary else 3
        local_t = (
            t_global if stage == 2 else t_global - stage_boundary
        )

        artists = []

        for fraction in PNEUMATIC_FRACTION_ORDER:
            r = all_results[fraction]

            if stage == 2:
                fixed_run = r["fixed2"]
                ml_run = r["adaptive2"]
                fixed_feed = r["feed"]
                ml_feed = r["feed"]
                stage_name = "STAGE 2: optimize Al top product"
            else:
                fixed_run = r["fixed3"]
                ml_run = r["adaptive3"]
                fixed_feed = clean_for_next_stage(
                    r["fixed2"]["result"][
                        ~r["fixed2"]["result"]["dynamic_stream"].eq("light")
                    ]
                )
                ml_feed = clean_for_next_stage(
                    r["adaptive2"]["result"][
                        ~r["adaptive2"]["result"]["dynamic_stream"].eq("light")
                    ]
                )
                stage_name = "STAGE 3: optimize Cu bottom product"

            ff = get_frame(fixed_run, local_t)
            mf = get_frame(ml_run, local_t)

            set_particles(fraction, "fixed", ff, fixed_feed)
            set_particles(fraction, "ml", mf, ml_feed)

            if ff is not None:
                info_text[(fraction, "fixed")].set_text(
                    f"{stage_name}\n"
                    f"t = {t_global:4.1f} s\n"
                    f"Q set = {ff['flow_setpoint_m3_h']:.1f} m3/h\n"
                    f"top = {ff['top_count']}\n"
                    f"bottom = {ff['bottom_count']}\n"
                    + (
                        f"Al top purity = {ff['al_top_purity']:.2f}\n"
                        f"Al recovery = {ff['al_top_recovery']:.2f}"
                        if stage == 2
                        else
                        f"Cu bottom purity = {ff['cu_bottom_purity']:.2f}\n"
                        f"Cu recovery = {ff['cu_bottom_recovery']:.2f}"
                    )
                )

            if mf is not None:
                info_text[(fraction, "ml")].set_text(
                    f"{stage_name}\n"
                    f"t = {t_global:4.1f} s\n"
                    f"Q set = {mf['flow_setpoint_m3_h']:.1f} m3/h\n"
                    f"top = {mf['top_count']}\n"
                    f"bottom = {mf['bottom_count']}\n"
                    + (
                        f"Al top purity = {mf['al_top_purity']:.2f}\n"
                        f"Al recovery = {mf['al_top_recovery']:.2f}"
                        if stage == 2
                        else
                        f"Cu bottom purity = {mf['cu_bottom_purity']:.2f}\n"
                        f"Cu recovery = {mf['cu_bottom_recovery']:.2f}"
                    )
                )

            gd = graph_data[fraction]

            # Air velocity.
            lf, lm = air_lines[fraction]
            fmask = gd["fixed_air"]["global_time_s"] <= t_global
            mmask = gd["ml_air"]["global_time_s"] <= t_global
            lf.set_data(
                gd["fixed_air"].loc[fmask, "global_time_s"],
                gd["fixed_air"].loc[fmask, "air_velocity_m_s"],
            )
            lm.set_data(
                gd["ml_air"].loc[mmask, "global_time_s"],
                gd["ml_air"].loc[mmask, "air_velocity_m_s"],
            )
            axes[PNEUMATIC_FRACTION_ORDER.index(fraction), 2].relim()
            axes[PNEUMATIC_FRACTION_ORDER.index(fraction), 2].autoscale_view(
                scalex=False, scaley=True
            )

            # Outlet counts. Stage 3 deliberately resets counts.
            fstage_hist = fixed_run["history"]
            mstage_hist = ml_run["history"]
            fshow = fstage_hist[fstage_hist["time_s"] <= local_t]
            mshow = mstage_hist[mstage_hist["time_s"] <= local_t]
            offset = 0.0 if stage == 2 else stage_boundary

            ftop, fbottom, mtop, mbottom = outlet_lines[fraction]
            ftop.set_data(
                fshow["time_s"] + offset,
                fshow["top_count"],
            )
            fbottom.set_data(
                fshow["time_s"] + offset,
                fshow["bottom_count"],
            )
            mtop.set_data(
                mshow["time_s"] + offset,
                mshow["top_count"],
            )
            mbottom.set_data(
                mshow["time_s"] + offset,
                mshow["bottom_count"],
            )
            axes[PNEUMATIC_FRACTION_ORDER.index(fraction), 3].relim()
            axes[PNEUMATIC_FRACTION_ORDER.index(fraction), 3].autoscale_view(
                scalex=False, scaley=True
            )

            # Product purity.
            qf, qm = quality_lines[fraction]
            fqmask = gd["fixed_quality"]["global_time_s"] <= t_global
            mqmask = gd["ml_quality"]["global_time_s"] <= t_global
            qf.set_data(
                gd["fixed_quality"].loc[fqmask, "global_time_s"],
                gd["fixed_quality"].loc[fqmask, "purity"],
            )
            qm.set_data(
                gd["ml_quality"].loc[mqmask, "global_time_s"],
                gd["ml_quality"].loc[mqmask, "purity"],
            )

            artists.extend(
                list(particle_artists[(fraction, "fixed")].values())
                + list(particle_artists[(fraction, "ml")].values())
                + [
                    info_text[(fraction, "fixed")],
                    info_text[(fraction, "ml")],
                    lf, lm,
                    ftop, fbottom, mtop, mbottom,
                    qf, qm,
                ]
            )

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )
    fig._hackathon_animation = anim

    if args.save_gif:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / args.gif_name
        print(f"Saving animation: {path}")
        anim.save(
            path,
            writer=PillowWriter(fps=args.fps),
            dpi=90,
        )
        print(f"Saved: {path}")

    if not args.no_show:
        plt.show()

    plt.close(fig)


def save_outputs(all_results):
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []

    for fraction in PNEUMATIC_FRACTION_ORDER:
        r = all_results[fraction]

        for strategy, stage, key in [
            ("fixed", 2, "fixed2"),
            ("closed_loop_ml", 2, "adaptive2"),
            ("fixed", 3, "fixed3"),
            ("closed_loop_ml", 3, "adaptive3"),
        ]:
            run = r[key]
            row = {
                "fraction_mm": fraction,
                "strategy": strategy,
                "stage": stage,
                **final_stage_row(run["result"], stage),
            }
            rows.append(row)

            run["history"].to_csv(
                OUTPUT_DIR
                / f"{fraction}_{strategy}_stage{stage}_history.csv",
                index=False,
            )

            if "decisions" in run:
                run["decisions"].to_csv(
                    OUTPUT_DIR
                    / f"{fraction}_{strategy}_stage{stage}_decisions.csv",
                    index=False,
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "hackathon_optimization_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("HACKATHON OPTIMIZATION SUMMARY")
    print("=" * 100)
    display_cols = [
        "fraction_mm",
        "strategy",
        "stage",
        "feed_particles",
        "al_top_purity",
        "al_top_recovery",
        "cu_bottom_purity",
        "cu_bottom_recovery",
        "cu_loss_top",
    ]
    print(summary[display_cols].to_string(index=False))
    return summary


def main():
    args = parse_args()
    full_feed = generate_feed()
    cal_args = CalibrationArgs(args)
    rng = np.random.default_rng(RANDOM_SEED)

    all_results = {}

    for i, fraction in enumerate(PNEUMATIC_FRACTION_ORDER):
        print()
        print("=" * 100)
        print(f"Preparing {fraction} mm")
        print("=" * 100)

        pool = metal_pool_for_fraction(full_feed, fraction)
        lookup = build_suspension_lookup(pool)

        surrogate, _, acc = calibrate_surrogate(
            pool,
            fraction,
            cal_args,
            rng,
        )
        print(
            f"Digital-twin surrogate held-out accuracy: {100*acc:.1f}% "
            "(not real-world camera accuracy)"
        )

        all_results[fraction] = run_fraction(
            fraction=fraction,
            full_feed=full_feed,
            surrogate=surrogate,
            lookup=lookup,
            args=args,
            seed=RANDOM_SEED + i * 10000,
        )

    save_outputs(all_results)
    animate(all_results, args)

    print()
    print("Demo complete.")
    print(
        "Stage 2 objective: Al-rich top product with strong Cu-loss protection."
    )
    print(
        "Stage 3 objective: higher-purity Cu bottom product with a looser "
        "allowance for Cu loss to the top."
    )


if __name__ == "__main__":
    main()
