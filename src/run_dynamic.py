"""
Run the time-resolved pulsated pneumatic separator.

Examples
--------
Animated simulation:
    python3 run_dynamic.py --fraction 0.25-1.00

Different pulse:
    python3 run_dynamic.py --fraction 1.00-1.50 \
        --velocity 2.8 --frequency 5 --duty 0.45

Fast headless test:
    python3 run_dynamic.py --fraction 0.25-1.00 \
        --duration 2 --particles 200 --no-animation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from config import PNEUMATIC_FRACTION_ORDER
from feed_model import generate_feed
from dynamic_config import (
    TOTAL_HEIGHT_M,
    DEFAULT_N_PARTICLES,
    DEFAULT_DURATION_S,
    DEFAULT_DT_S,
    DEFAULT_HIGH_VELOCITY_M_S,
    DEFAULT_LOW_FLOW_FRACTION,
    DEFAULT_PULSE_FREQUENCY_HZ,
    DEFAULT_DUTY_CYCLE,
    ANIMATION_FPS,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Time-resolved pulsated pneumatic separator "
            "for one comminution fraction."
        )
    )

    parser.add_argument(
        "--fraction",
        choices=PNEUMATIC_FRACTION_ORDER,
        default="0.25-1.00",
        help="Comminution fraction to simulate (mm).",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_N_PARTICLES,
        help="Number of Monte Carlo parcels to animate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Maximum simulated time (s).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT_S,
        help="Numerical integration time step (s).",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=DEFAULT_HIGH_VELOCITY_M_S,
        help=(
            "High-pulse air velocity in the "
            "100-mm reference section (m/s)."
        ),
    )
    parser.add_argument(
        "--low-flow-fraction",
        type=float,
        default=DEFAULT_LOW_FLOW_FRACTION,
        help=(
            "Off-pulse flow divided by high-pulse flow "
            "(0 to 1)."
        ),
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_PULSE_FREQUENCY_HZ,
        help="Pulse frequency (Hz).",
    )
    parser.add_argument(
        "--duty",
        type=float,
        default=DEFAULT_DUTY_CYCLE,
        help="Pulse duty cycle (0 to 1].",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Run to completion without opening an animation.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save the final particle table and summary CSV.",
    )

    return parser.parse_args()


def print_basis(args, selected):
    print()
    print("=" * 78)
    print("DYNAMIC PULSATED PNEUMATIC SEPARATOR")
    print("=" * 78)
    print(f"Comminution fraction: {args.fraction} mm")
    print(f"Available feed parcels: {len(selected):,}")
    print(f"Dynamic particles: {min(args.particles, len(selected)):,}")
    print(f"High reference velocity: {args.velocity:.3f} m/s")
    print(
        "Low/high flow ratio: "
        f"{args.low_flow_fraction:.3f}"
    )
    print(f"Pulse frequency: {args.frequency:.3f} Hz")
    print(f"Duty cycle: {args.duty:.3f}")
    print(f"Integration dt: {args.dt:.4f} s")
    print(f"Maximum duration: {args.duration:.2f} s")
    print()
    print(
        "Dynamic drag is calibrated so each parcel reproduces "
        "its literature-based suspension velocity."
    )


def print_results(sim):
    summary = sim.performance_summary()
    table = sim.material_outlet_table()

    print()
    print("=" * 78)
    print("DYNAMIC RESULTS")
    print("=" * 78)

    for key, value in summary.items():
        if isinstance(value, float):
            if np.isnan(value):
                text = "n/a"
            else:
                text = f"{value:.4f}"
        else:
            text = str(value)

        print(f"{key}: {text}")

    print()
    print("Outlet counts by material:")
    print(table)


def draw_separator(ax):
    z = np.linspace(
        0.0,
        TOTAL_HEIGHT_M,
        300,
    )
    r = radius_at_height(z)

    ax.plot(
        -r,
        z,
        linewidth=1.5,
    )
    ax.plot(
        r,
        z,
        linewidth=1.5,
    )

    max_r = np.max(r)
    ax.set_xlim(
        -max_r * 1.45,
        max_r * 1.45,
    )
    ax.set_ylim(
        -0.03,
        TOTAL_HEIGHT_M + 0.03,
    )
    ax.set_xlabel(
        "Horizontal position x (m)"
    )
    ax.set_ylabel(
        "Height z (m)"
    )
    ax.set_title(
        "Particle motion in variable-diameter separator"
    )


def animate(sim, args):
    fig = plt.figure(
        figsize=(11, 7)
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.15],
    )

    ax_sep = fig.add_subplot(
        gs[:, 0]
    )
    ax_air = fig.add_subplot(
        gs[0, 1]
    )
    ax_count = fig.add_subplot(
        gs[1, 1]
    )

    draw_separator(ax_sep)

    materials = list(
        sim.particles["material"].unique()
    )

    scatter_by_material = {}

    # Visual particle area follows the predicted measured size distribution.
    measured_size_mm = (
        sim.particles["size_mm"]
        .to_numpy(dtype=float)
    )

    size_min = float(np.min(measured_size_mm))
    size_max = float(np.max(measured_size_mm))

    if size_max > size_min:
        size_fraction = (
            (measured_size_mm - size_min)
            / (size_max - size_min)
        )
    else:
        size_fraction = np.ones_like(
            measured_size_mm
        )

    marker_area = (
        PARTICLE_VISUAL_MIN_PT2
        + (
            PARTICLE_VISUAL_MAX_PT2
            - PARTICLE_VISUAL_MIN_PT2
        )
        * size_fraction**PARTICLE_VISUAL_POWER
    )

    for material in materials:
        mask = (
            sim.particles["material"]
            .eq(material)
            .to_numpy()
        )

        scatter_by_material[material] = (
            ax_sep.scatter(
                sim.x[mask],
                sim.z[mask],
                s=marker_area[mask],
                marker=MATERIAL_MARKERS.get(
                    material,
                    "o",
                ),
                alpha=0.72,
                label=material,
            )
        )

    ax_sep.legend(
        title=(
            "Material\n"
            "square = flake/film\n"
            "circle = powder/nonmetal"
        ),
        loc="upper right",
    )

    ax_sep.text(
        0.02,
        0.98,
        "Symbol area follows predicted measured particle size",
        transform=ax_sep.transAxes,
        va="top",
        fontsize=9,
    )

    status_text = ax_sep.text(
        0.02,
        0.02,
        "",
        transform=ax_sep.transAxes,
        va="bottom",
    )

    air_line, = ax_air.plot(
        [],
        [],
        label="100-mm reference velocity",
    )
    ax_air.set_xlabel("Time (s)")
    ax_air.set_ylabel("Air velocity (m/s)")
    ax_air.set_xlim(0, args.duration)
    ax_air.set_ylim(
        0,
        max(args.velocity * 1.15, 0.5),
    )
    ax_air.set_title("Pulsated airflow")
    ax_air.legend()

    light_line, = ax_count.plot(
        [],
        [],
        label="Light outlet",
    )
    heavy_line, = ax_count.plot(
        [],
        [],
        label="Heavy outlet",
    )
    active_line, = ax_count.plot(
        [],
        [],
        label="Still in separator",
    )

    ax_count.set_xlabel("Time (s)")
    ax_count.set_ylabel("Particle parcels")
    ax_count.set_xlim(0, args.duration)
    ax_count.set_ylim(0, sim.n)
    ax_count.set_title("Outlet accumulation")
    ax_count.legend()

    # Number of numerical substeps per visual frame.
    visual_dt = 1.0 / ANIMATION_FPS
    substeps = max(
        1,
        int(round(visual_dt / sim.settings.dt_s)),
    )

    max_frames = int(
        np.ceil(
            args.duration
            / (substeps * sim.settings.dt_s)
        )
    )

    def update(_frame):
        for _ in range(substeps):
            if (
                sim.time_s >= args.duration
                or not np.any(sim.active_mask)
            ):
                break
            sim.step()

        for material in materials:
            material_mask = (
                sim.particles["material"]
                .eq(material)
                .to_numpy()
            )

            # Show active particles at their location and exited
            # particles fixed at the corresponding outlet boundary.
            x = sim.x[material_mask]
            z = sim.z[material_mask]

            scatter_by_material[
                material
            ].set_offsets(
                np.column_stack([x, z])
            )

        status_text.set_text(
            f"t = {sim.time_s:5.2f} s\n"
            f"active = {np.sum(sim.active_mask):d}\n"
            f"light = {np.sum(sim.light_mask):d}\n"
            f"heavy = {np.sum(sim.heavy_mask):d}"
        )

        t = np.asarray(
            sim.history_time
        )
        u = np.asarray(
            sim.history_reference_velocity
        )

        air_line.set_data(
            t,
            u,
        )

        light_line.set_data(
            t,
            sim.history_light,
        )
        heavy_line.set_data(
            t,
            sim.history_heavy,
        )
        active_line.set_data(
            t,
            sim.history_active,
        )

        return (
            list(scatter_by_material.values())
            + [
                status_text,
                air_line,
                light_line,
                heavy_line,
                active_line,
            ]
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=1000 / ANIMATION_FPS,
        blit=False,
        repeat=False,
    )

    fig.suptitle(
        f"{args.fraction} mm | "
        f"{args.frequency:.1f} Hz | "
        f"duty {args.duty:.2f} | "
        f"high {args.velocity:.2f} m/s"
    )
    fig.tight_layout()

    # Keep a reference alive until the window closes.
    fig._separator_animation = animation
    plt.show()

    # If the user closes the figure before the full simulation has
    # finished, complete the remaining requested duration numerically.
    if (
        sim.time_s < args.duration
        and np.any(sim.active_mask)
    ):
        sim.run(
            duration_s=(
                args.duration - sim.time_s
            )
        )


def save_results(sim, args):
    out_dir = (
        Path(__file__).resolve().parent
        / "outputs"
    )
    out_dir.mkdir(exist_ok=True)

    safe_fraction = (
        args.fraction.replace(".", "p")
    )

    particle_file = (
        out_dir
        / f"dynamic_particles_{safe_fraction}.csv"
    )
    summary_file = (
        out_dir
        / f"dynamic_summary_{safe_fraction}.csv"
    )

    sim.results().to_csv(
        particle_file,
        index=False,
    )

    pd.DataFrame(
        [sim.performance_summary()]
    ).to_csv(
        summary_file,
        index=False,
    )

    print()
    print("Saved:")
    print(particle_file)
    print(summary_file)


def main():
    args = parse_args()

    feed = generate_feed()

    selected = feed[
        feed["pneumatic_fraction"]
        == args.fraction
    ].copy()

    print_basis(
        args,
        selected,
    )

    settings = DynamicSettings(
        high_velocity_m_s=args.velocity,
        low_flow_fraction=args.low_flow_fraction,
        pulse_frequency_hz=args.frequency,
        duty_cycle=args.duty,
        dt_s=args.dt,
        duration_s=args.duration,
    )

    sim = DynamicSeparatorSimulation(
        selected,
        settings=settings,
        n_particles=args.particles,
    )

    if args.no_animation:
        sim.run()
    else:
        animate(
            sim,
            args,
        )

    print_results(sim)

    if args.save_results:
        save_results(
            sim,
            args,
        )


if __name__ == "__main__":
    main()
