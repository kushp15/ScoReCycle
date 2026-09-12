from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    SAVE_OUTPUTS,
    PNEUMATIC_FRACTION_ORDER,
)
from feed_model import generate_feed
from simulation import (
    separate,
    sweep_stage,
    best_operating_point,
    stream_composition_table,
)
from diagnostics import (
    print_suspension_diagnostics,
    print_fraction_below_velocity,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def normalize_mass(frame):
    frame = frame.copy()
    total = frame["statistical_mass"].sum()

    if total > 0:
        frame["statistical_mass"] /= total

    return frame


def print_feed_basis(feed):
    print()
    print("=" * 78)
    print("LITERATURE-CALIBRATED 0.25–2.00 mm PNEUMATIC FEED")
    print("=" * 78)

    print()
    print("Mass share by original pneumatic fraction:")
    print(
        feed.groupby(
            "pneumatic_fraction"
        )["statistical_mass"]
        .sum()
        .reindex(PNEUMATIC_FRACTION_ORDER)
    )

    print()
    print("Mass share by source subfraction:")
    print(
        feed.groupby(
            "source_subfraction"
        )["statistical_mass"]
        .sum()
    )

    print()
    print("Conditional composition inside each source subfraction:")
    table = (
        feed.groupby(
            [
                "source_subfraction",
                "material",
            ]
        )["statistical_mass"]
        .sum()
        .unstack(fill_value=0.0)
    )

    table = table.div(
        table.sum(axis=1),
        axis=0,
    )

    print(table)


def run_fraction(
    fraction_name,
    fraction_feed,
):
    fraction_feed = normalize_mass(
        fraction_feed
    )

    print()
    print("=" * 78)
    print(
        f"PNEUMATIC FRACTION: {fraction_name} mm"
    )
    print("=" * 78)

    print()
    print("Feed composition:")
    print(
        fraction_feed.groupby(
            "material"
        )["statistical_mass"]
        .sum()
    )

    print()
    print("Foil liberation state:")
    foil = fraction_feed[
        fraction_feed["material"].isin(
            ["aluminum", "copper"]
        )
    ]

    if not foil.empty:
        state_table = (
            foil.groupby(
                [
                    "material",
                    "liberation_state",
                ]
            )["statistical_mass"]
            .sum()
            .unstack(fill_value=0.0)
        )

        state_table = state_table.div(
            state_table.sum(axis=1),
            axis=0,
        )

        print(state_table)

    print_suspension_diagnostics(
        fraction_feed
    )

    # --------------------------------------------------------
    # Stage 1:
    # light nonmetals -> LIGHT
    # Cu + Al -> HEAVY
    # --------------------------------------------------------
    stage1_sweep = sweep_stage(
        fraction_feed,
        stage_number=1,
    )

    stage1_best = best_operating_point(
        stage1_sweep
    )

    v1 = float(
        stage1_best[
            "air_velocity_m_s"
        ]
    )

    stage1 = separate(
        fraction_feed,
        v1,
    )

    print()
    print("-" * 78)
    print(
        "STAGE 1: NONMETALS -> LIGHT / Cu+Al -> HEAVY"
    )
    print("-" * 78)
    print(
        f"Selected velocity: {v1:.3f} m/s"
    )
    print(
        "Equivalent 100-mm flow: "
        f"{stage1_best['flow_100mm_m3_hr']:.1f} m^3/hr"
    )
    print(
        "Light-material recovery: "
        f"{stage1_best['target_recovery']:.3f}"
    )
    print(
        "Light-stream purity: "
        f"{stage1_best['target_purity']:.3f}"
    )
    print(
        "Metal retention in heavy: "
        f"{stage1_best['metal_retention']:.3f}"
    )
    print(
        f"Stage-1 score: {stage1_best['score']:.3f}"
    )

    print()
    print(
        stream_composition_table(
            stage1
        )
    )

    print_fraction_below_velocity(
        fraction_feed,
        v1,
    )

    # --------------------------------------------------------
    # Stage 2:
    # operates only on the Stage-1 heavy product.
    # --------------------------------------------------------
    stage2_feed = (
        stage1[
            stage1["stream"] == "heavy"
        ]
        .drop(columns="stream")
        .copy()
    )

    stage2_sweep = None
    stage2_best = None
    v2 = None
    stage2 = None

    if not stage2_feed.empty:
        stage2_feed = normalize_mass(
            stage2_feed
        )

        stage2_sweep = sweep_stage(
            stage2_feed,
            stage_number=2,
        )

        stage2_best = best_operating_point(
            stage2_sweep
        )

        v2 = float(
            stage2_best[
                "air_velocity_m_s"
            ]
        )

        stage2 = separate(
            stage2_feed,
            v2,
        )

        print()
        print("-" * 78)
        print(
            "STAGE 2: Al -> LIGHT / Cu -> HEAVY"
        )
        print("-" * 78)
        print(
            f"Selected velocity: {v2:.3f} m/s"
        )
        print(
            "Equivalent 100-mm flow: "
            f"{stage2_best['flow_100mm_m3_hr']:.1f} m^3/hr"
        )
        print(
            "Al recovery: "
            f"{stage2_best['target_recovery']:.3f}"
        )
        print(
            "Al purity: "
            f"{stage2_best['target_purity']:.3f}"
        )
        print(
            "Cu retention in heavy: "
            f"{stage2_best['cu_retention']:.3f}"
        )
        print(
            f"Stage-2 score: {stage2_best['score']:.3f}"
        )

        print()
        print(
            stream_composition_table(
                stage2
            )
        )

    summary = {
        "fraction": fraction_name,
        "stage1_velocity": v1,
        "stage1_score":
            float(stage1_best["score"]),
        "stage1_light_recovery":
            float(stage1_best["target_recovery"]),
        "stage1_light_purity":
            float(stage1_best["target_purity"]),
        "stage1_metal_retention":
            float(stage1_best["metal_retention"]),
        "stage2_velocity":
            None if v2 is None else v2,
        "stage2_score":
            None if stage2_best is None
            else float(stage2_best["score"]),
        "stage2_al_recovery":
            None if stage2_best is None
            else float(stage2_best["target_recovery"]),
        "stage2_al_purity":
            None if stage2_best is None
            else float(stage2_best["target_purity"]),
        "stage2_cu_retention":
            None if stage2_best is None
            else float(stage2_best["cu_retention"]),
    }

    return {
        "summary": summary,
        "feed": fraction_feed,
        "stage1_sweep": stage1_sweep,
        "stage2_sweep": stage2_sweep,
    }


# ============================================================
# PARTICLE-LEVEL SEPARATION PLOTS
# ============================================================
#
# These plots intentionally return to the style of the earlier model:
#   - scatter plots of the simulated particle population
#   - recovery / purity / retention curves versus air velocity
#
# There are NO composition bar charts.
#
# Each comminution fraction is plotted independently because that is
# how the pneumatic-separation literature treats the material.
# ============================================================


def _plot_sample(frame, max_points=6000):
    """
    Keep scatter plots readable while preserving the simulated distribution.

    The simulation itself still uses every parcel. This function only limits
    how many points are DRAWN.
    """
    if len(frame) <= max_points:
        return frame

    return frame.sample(
        n=max_points,
        random_state=42,
    )


def plot_size_density_scatter(
    fraction_feed,
    fraction_name,
):
    """
    Original-style particle distribution:
        x = particle size
        y = effective particle density

    One figure is produced for each comminution fraction.
    """
    plot_df = _plot_sample(
        fraction_feed
    )

    plt.figure(figsize=(9, 6))

    for material, group in plot_df.groupby(
        "material"
    ):
        plt.scatter(
            group["size_mm"],
            group["effective_density_kg_m3"],
            s=12,
            alpha=0.35,
            label=material,
        )

    plt.xlabel("Particle size (mm)")
    plt.ylabel("Effective particle density (kg/m³)")
    plt.title(
        f"Particle Size–Density Distribution: {fraction_name} mm"
    )
    plt.legend(title="Material")
    plt.tight_layout()


def plot_size_suspension_scatter(
    fraction_feed,
    fraction_name,
    stage1_velocity,
    stage2_velocity,
):
    """
    Particle-level aerodynamic map:
        x = particle size
        y = calculated suspension velocity

    Horizontal lines show the selected Stage-1 and Stage-2 air velocities.
    A particle lying BELOW a horizontal airflow line has u_s <= u_air and
    is predicted to enter the light/entrained stream in the present model.
    """
    plot_df = _plot_sample(
        fraction_feed
    )

    plt.figure(figsize=(9, 6))

    for material, group in plot_df.groupby(
        "material"
    ):
        plt.scatter(
            group["size_mm"],
            group["suspension_velocity_m_s"],
            s=12,
            alpha=0.35,
            label=material,
        )

    plt.axhline(
        stage1_velocity,
        linestyle="--",
        linewidth=1.5,
        label=f"Stage 1 air velocity = {stage1_velocity:.2f} m/s",
    )

    if stage2_velocity is not None:
        plt.axhline(
            stage2_velocity,
            linestyle=":",
            linewidth=1.5,
            label=f"Stage 2 air velocity = {stage2_velocity:.2f} m/s",
        )

    plt.xlabel("Particle size (mm)")
    plt.ylabel("Calculated suspension velocity (m/s)")
    plt.title(
        f"Particle Separation Map: {fraction_name} mm"
    )
    plt.legend(
        title="Material / airflow"
    )
    plt.tight_layout()


def plot_stage1_curves(
    stage1_sweep,
    fraction_name,
    selected_velocity,
):
    """
    Same style as the original optimization graph, but now generated
    independently for each comminution fraction.
    """
    plt.figure(figsize=(9, 6))

    plt.plot(
        stage1_sweep["air_velocity_m_s"],
        stage1_sweep["target_recovery"],
        label="Light-material recovery",
    )

    plt.plot(
        stage1_sweep["air_velocity_m_s"],
        stage1_sweep["target_purity"],
        label="Light-stream purity",
    )

    plt.plot(
        stage1_sweep["air_velocity_m_s"],
        stage1_sweep["metal_retention"],
        label="Cu + Al retention in heavy stream",
    )

    plt.axvline(
        selected_velocity,
        linestyle="--",
        label=f"Selected velocity = {selected_velocity:.2f} m/s",
    )

    plt.xlabel("Air velocity (m/s)")
    plt.ylabel("Mass fraction")
    plt.title(
        f"Stage 1 Separation Performance: {fraction_name} mm"
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()


def plot_stage2_curves(
    stage2_sweep,
    fraction_name,
    selected_velocity,
):
    """
    Stage-2 plot reports the physically interpretable quantities directly.

    The objective used internally is:
        J2 = Al recovery × Al purity × Cu retention

    But the graph shows the three underlying terms so the model behavior
    is not hidden by one artificial score.
    """
    if stage2_sweep is None:
        return

    plt.figure(figsize=(9, 6))

    plt.plot(
        stage2_sweep["air_velocity_m_s"],
        stage2_sweep["target_recovery"],
        label="Al recovery to light stream",
    )

    plt.plot(
        stage2_sweep["air_velocity_m_s"],
        stage2_sweep["target_purity"],
        label="Al purity of light stream",
    )

    plt.plot(
        stage2_sweep["air_velocity_m_s"],
        stage2_sweep["cu_retention"],
        label="Cu retention in heavy stream",
    )

    plt.axvline(
        selected_velocity,
        linestyle="--",
        label=f"Selected velocity = {selected_velocity:.2f} m/s",
    )

    plt.xlabel("Air velocity (m/s)")
    plt.ylabel(
        "Fraction   "
        "(J₂ = Al recovery × Al purity × Cu retention)"
    )
    plt.title(
        f"Stage 2 Al/Cu Separation Performance: {fraction_name} mm"
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()


def main():
    feed = generate_feed()

    print_feed_basis(feed)

    results = []

    # Each comminution fraction is processed independently.
    for fraction_name in PNEUMATIC_FRACTION_ORDER:
        fraction_feed = feed[
            feed["pneumatic_fraction"]
            == fraction_name
        ].copy()

        result = run_fraction(
            fraction_name,
            fraction_feed,
        )

        results.append(result)

    summary = pd.DataFrame(
        [
            result["summary"]
            for result in results
        ]
    )

    print()
    print("=" * 78)
    print("SUMMARY BY COMMINUTION FRACTION")
    print("=" * 78)
    print(summary.to_string(index=False))

    # ========================================================
    # CREATE PARTICLE-LEVEL PLOTS FOR EACH SIZE FRACTION
    # ========================================================
    #
    # For each comminution fraction:
    #   1. size vs density scatter
    #   2. size vs suspension velocity scatter
    #   3. Stage-1 recovery/purity/retention curves
    #   4. Stage-2 Al/Cu separation curves
    #
    # The separator physics is therefore visible directly instead of
    # being summarized by bar charts.
    # ========================================================

    for result in results:
        fraction_name = result["summary"]["fraction"]
        fraction_feed = result["feed"]

        plot_size_density_scatter(
            fraction_feed,
            fraction_name,
        )

        plot_size_suspension_scatter(
            fraction_feed,
            fraction_name,
            result["summary"]["stage1_velocity"],
            result["summary"]["stage2_velocity"],
        )

        plot_stage1_curves(
            result["stage1_sweep"],
            fraction_name,
            result["summary"]["stage1_velocity"],
        )

        plot_stage2_curves(
            result["stage2_sweep"],
            fraction_name,
            result["summary"]["stage2_velocity"],
        )

    print()
    print(
        "Particle-level comparison figures were created for each "
        "comminution fraction."
    )
    print(
        "A particle below an airflow line on the suspension-velocity "
        "scatter is predicted to be entrained/light in the current model."
    )

    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(
            exist_ok=True
        )

        summary.to_csv(
            OUTPUT_DIR
            / "simulation_summary.csv",
            index=False,
        )

        feed.to_csv(
            OUTPUT_DIR
            / "latest_particle_results.csv",
            index=False,
        )

        print()
        print("Saved/overwritten:")
        print(
            OUTPUT_DIR
            / "simulation_summary.csv"
        )
        print(
            OUTPUT_DIR
            / "latest_particle_results.csv"
        )

    else:
        print()
        print(
            "CSV saving is OFF. "
            "Set SAVE_OUTPUTS = True in config.py "
            "if you want the latest results saved."
        )

    plt.show()


if __name__ == "__main__":
    main()
