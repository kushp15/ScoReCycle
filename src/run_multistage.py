"""
Three-stage dynamic pneumatic-separation cascade using the original
literature-calibrated particle physics.

This version deliberately DOES NOT force perfect Cu/Al separation.

For every Zhu 2021 pneumatic size class:
    0.25-1.00 mm
    1.00-1.50 mm
    1.50-2.00 mm

the same three-stage logic is simulated independently:

Stage 1
    Remove low-suspension-velocity separator / black mass / other light
    material. Metals reporting to the light stream are counted as real losses.

Stage 2
    Primary Al/Cu separation using Zhu's size-class-specific reported air
    volume. Light product is Al-rich by process intent; heavy product is
    Cu-rich by process intent, but BOTH can contain impurities.

Stage 3
    Reprocess the Stage-2 overlap stream: all unresolved middlings PLUS a
    controlled recycle of borderline light/heavy particles nearest the cut.
    No particle is selected by material identity. Selection is based only on
    its modeled suspension velocity relative to the Stage-2 cut.

This intentionally preserves cross-contamination. Small Cu and large/partially
coated Al can occupy the same aerodynamic region, so both materials can appear
in the Stage-3 feed and both final products.

No particle is reassigned by material label. Its trajectory is determined by
the original model's size, density, foil/composite state, suspension velocity,
pulse history, and separator dynamics.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd

from config import (
    PNEUMATIC_FRACTION_ORDER,
    RANDOM_SEED,
    D_STRAIGHT,
    ZHU_PRIMARY_METAL_FLOW_M3_H,
    ZHU_STAGE3_CLEANING_FLOW_M3_H,
)
from feed_model import generate_feed
from dynamic_separator import DynamicSettings, DynamicSeparatorSimulation
from run_dynamic import animate


DYNAMIC_RESULT_COLUMNS = [
    "x_m",
    "z_m",
    "vx_m_s",
    "vz_m_s",
    "dynamic_stream",
    "exit_time_s",
]


# Stage-1 high-pulse velocity is retained from the original project.
# It is a project calibration/design setting, NOT a value claimed from Zhu.
DEFAULT_STAGE1_VELOCITY_M_S = 1.35


def clean_for_next_stage(df):
    """Remove trajectory columns while preserving all particle physics."""
    return df.drop(
        columns=[c for c in DYNAMIC_RESULT_COLUMNS if c in df.columns],
        errors="ignore",
    ).reset_index(drop=True)


def flow_to_reference_velocity(flow_m3_h):
    """Convert volumetric flow to velocity in the 100-mm straight section."""
    q_m3_s = flow_m3_h / 3600.0
    area = math.pi * D_STRAIGHT**2 / 4.0
    return q_m3_s / area


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run realistic three-stage pneumatic-separation simulations "
            "for one or all Zhu 2021 size classes."
        )
    )

    p.add_argument(
        "--fraction",
        choices=["all", *PNEUMATIC_FRACTION_ORDER],
        default="all",
        help=(
            "Zhu 2021 screened particle-size class. Default 'all' runs "
            "0.25-1.00, 1.00-1.50, and 1.50-2.00 mm independently."
        ),
    )
    p.add_argument(
        "--particles",
        type=int,
        default=1000,
        help="Equal-mass Monte Carlo parcels per size-class simulation.",
    )
    p.add_argument(
        "--no-animation",
        action="store_true",
        help="Run without opening animation windows.",
    )
    p.add_argument(
        "--save-results",
        action="store_true",
        help="Save particle-level stage results and summaries to outputs/.",
    )

    # Keep the original dynamic pulse model.
    p.add_argument("--frequency", type=float, default=4.0)
    p.add_argument("--duty", type=float, default=0.50)
    p.add_argument("--low-flow-fraction", type=float, default=0.35)
    p.add_argument("--dt", type=float, default=0.002)

    p.add_argument(
        "--stage1-velocity",
        type=float,
        default=DEFAULT_STAGE1_VELOCITY_M_S,
        help=(
            "Stage-1 high-pulse reference velocity, m/s. "
            "Project calibration setting; default preserves the original model."
        ),
    )
    p.add_argument("--stage1-duration", type=float, default=10.0)
    p.add_argument("--stage2-duration", type=float, default=12.0)
    p.add_argument("--stage3-duration", type=float, default=12.0)

    return p.parse_args()


def stage_settings(args, velocity, duration):
    return DynamicSettings(
        high_velocity_m_s=velocity,
        low_flow_fraction=args.low_flow_fraction,
        pulse_frequency_hz=args.frequency,
        duty_cycle=args.duty,
        dt_s=args.dt,
        duration_s=duration,
    )


def animation_args(args, title, velocity, duration):
    return SimpleNamespace(
        fraction=title,
        duration=duration,
        velocity=velocity,
        frequency=args.frequency,
        duty=args.duty,
    )


def run_stage(
    feed,
    args,
    fraction,
    stage_number,
    title,
    velocity,
    duration,
    seed,
    flow_m3_h=None,
):
    if feed.empty:
        return pd.DataFrame(columns=list(feed.columns) + DYNAMIC_RESULT_COLUMNS)

    sim = DynamicSeparatorSimulation(
        feed,
        settings=stage_settings(args, velocity, duration),
        n_particles=len(feed),
        seed=seed,
    )

    print()
    print("=" * 86)
    print(
        f"{fraction} mm | STAGE {stage_number}: {title}"
    )
    print("=" * 86)
    print(f"Feed parcels: {len(feed):,}")
    actual_min = feed["size_mm"].min()
    actual_max = feed["size_mm"].max()
    print(
        f"Particle-size range in this stage: "
        f"{actual_min:.4f}-{actual_max:.4f} mm"
    )
    if flow_m3_h is not None:
        print(
            f"Reference air volume: {flow_m3_h:.1f} m^3/h "
            f"-> {velocity:.3f} m/s in 100-mm section"
        )
    else:
        print(
            f"High-pulse reference velocity: {velocity:.3f} m/s "
            "(project Stage-1 setting)"
        )
    print(
        f"Pulse: {args.frequency:.2f} Hz, duty={args.duty:.2f}, "
        f"low/high={args.low_flow_fraction:.2f}"
    )

    if args.no_animation:
        sim.run()
    else:
        animate(
            sim,
            animation_args(
                args,
                (
                    f"{fraction} mm | Stage {stage_number}: {title}"
                ),
                velocity,
                duration,
            ),
        )

    result = sim.results()

    print()
    print("Outlet counts (other_light = unresolved light nonmetal/plastic proxy):")
    print(pd.crosstab(result["material"], result["dynamic_stream"]))

    return result


def material_count(df, material):
    if df.empty or "material" not in df.columns:
        return 0
    return int(df["material"].eq(material).sum())


def purity(df, material):
    if len(df) == 0:
        return np.nan
    return float(df["material"].eq(material).mean())


def recovery(product_df, original_df, material):
    denom = material_count(original_df, material)
    if denom == 0:
        return np.nan
    return material_count(product_df, material) / denom


def pct(x):
    return "n/a" if np.isnan(x) else f"{100*x:.1f}%"


def composition_row(label, df):
    return {
        "stream": label,
        "parcels": len(df),
        "Al": material_count(df, "aluminum"),
        "Cu": material_count(df, "copper"),
        "black_mass": material_count(df, "black_mass"),
        "separator": material_count(df, "separator"),
        "other_light": material_count(df, "other_light"),
    }



def select_overlap_recycle(stage2_result, cut_velocity, band_fraction=0.18):
    """
    Build a physically meaningful Stage-3 overlap stream without using
    material labels.

    We include:
      1) all Stage-2 active/middlings particles;
      2) light and heavy particles whose suspension velocity lies close to the
         Stage-2 cut.

    The overlap band is a model-design parameter, not a literature measurement.
    It represents particles most likely to be misclassified because their
    aerodynamic threshold is close to the operating cut.

    A particle can therefore enter Stage 3 whether it is Cu or Al.
    """
    if stage2_result.empty:
        return clean_for_next_stage(stage2_result)

    us = stage2_result["suspension_velocity_m_s"].to_numpy(dtype=float)
    band = band_fraction * cut_velocity

    near_cut = np.abs(us - cut_velocity) <= band
    unresolved = stage2_result["dynamic_stream"].eq("active").to_numpy()

    mask = near_cut | unresolved

    return clean_for_next_stage(
        stage2_result.loc[mask]
    )


def run_fraction(args, feed, fraction, index):
    selected = feed[
        feed["pneumatic_fraction"].eq(fraction)
    ].copy()

    if selected.empty:
        raise RuntimeError(f"No feed rows found for {fraction} mm.")

    n = min(args.particles, len(selected))
    initial = selected.sample(
        n=n,
        replace=False,
        random_state=RANDOM_SEED + 1000 + index,
    ).reset_index(drop=True)

    q2 = ZHU_PRIMARY_METAL_FLOW_M3_H[fraction]
    q3 = ZHU_STAGE3_CLEANING_FLOW_M3_H[fraction]
    v2 = flow_to_reference_velocity(q2)
    v3 = flow_to_reference_velocity(q3)

    print()
    print("#" * 86)
    print(f"THREE-STAGE SIMULATION — ZHU SIZE CLASS {fraction} mm")
    print("#" * 86)
    print(f"Initial parcels: {len(initial):,}")

    initial_comp = (
        initial["material"]
        .value_counts(normalize=True)
        .mul(100.0)
        .round(1)
    )
    print("Initial model feed composition (parcel/mass proxy):")
    for mat, pct_value in initial_comp.items():
        display = (
            "unresolved light nonmetal/plastic proxy"
            if mat == "other_light"
            else mat
        )
        print(f"  {display}: {pct_value:.1f}%")
    print(
        "Observed sampled size bounds: "
        f"{initial['size_mm'].min():.4f}-"
        f"{initial['size_mm'].max():.4f} mm"
    )
    print(
        f"Stage 2 Zhu primary flow: {q2:.1f} m^3/h "
        f"({v2:.3f} m/s reference velocity)"
    )
    print(
        f"Stage 3 middlings flow: {q3:.1f} m^3/h "
        f"({v3:.3f} m/s reference velocity)"
    )

    # ------------------------------------------------------------
    # Stage 1: nonmetal/light removal
    # ------------------------------------------------------------
    r1 = run_stage(
        initial,
        args,
        fraction=fraction,
        stage_number=1,
        title="light-material / separator removal",
        velocity=args.stage1_velocity,
        duration=args.stage1_duration,
        seed=RANDOM_SEED + 10 * index + 1,
    )

    stage1_light = r1[r1["dynamic_stream"].eq("light")].copy()
    stage1_middlings = r1[r1["dynamic_stream"].eq("active")].copy()
    stage2_feed = clean_for_next_stage(
        r1[r1["dynamic_stream"].eq("heavy")]
    )

    # ------------------------------------------------------------
    # Stage 2: primary size-specific Al/Cu cut from Zhu 2021
    # ------------------------------------------------------------
    r2 = run_stage(
        stage2_feed,
        args,
        fraction=fraction,
        stage_number=2,
        title="primary Al/Cu separation",
        velocity=v2,
        duration=args.stage2_duration,
        seed=RANDOM_SEED + 10 * index + 2,
        flow_m3_h=q2,
    )

    stage2_light = r2[r2["dynamic_stream"].eq("light")].copy()
    stage2_heavy = r2[r2["dynamic_stream"].eq("heavy")].copy()
    stage2_middlings = r2[r2["dynamic_stream"].eq("active")].copy()

    # ------------------------------------------------------------
    # Stage 3: overlap-region recycle.
    #
    # Build the Stage-3 feed WITHOUT looking at material labels. We recycle
    # all unresolved particles plus any Stage-2 light/heavy particles whose
    # suspension velocity lies near the Stage-2 aerodynamic cut.
    #
    # This is exactly where large Al and small Cu should overlap.
    # ------------------------------------------------------------
    stage3_feed = select_overlap_recycle(
        r2,
        cut_velocity=v2,
        band_fraction=0.18,
    )

    # Remove recycled rows from the unrecycled Stage-2 products using the
    # original row index carried by the result table.
    recycle_index = set(stage3_feed.index.tolist())

    # Because clean_for_next_stage resets the index, identify recycle rows
    # using a deterministic physical key generated from unchanged parcel
    # properties.
    key_cols = [
        "material",
        "size_mm",
        "suspension_velocity_m_s",
        "physical_particle_mass_kg",
    ]

    def _row_keys(df):
        if df.empty:
            return pd.Series(dtype=str)
        return (
            df[key_cols]
            .round(12)
            .astype(str)
            .agg("|".join, axis=1)
        )

    recycle_keys = set(_row_keys(stage3_feed))

    light_keys = _row_keys(stage2_light)
    heavy_keys = _row_keys(stage2_heavy)

    stage2_light_kept = stage2_light[
        ~light_keys.isin(recycle_keys)
    ].copy()
    stage2_heavy_kept = stage2_heavy[
        ~heavy_keys.isin(recycle_keys)
    ].copy()

    r3 = run_stage(
        stage3_feed,
        args,
        fraction=fraction,
        stage_number=3,
        title="overlap-region recycle pass",
        velocity=v3,
        duration=args.stage3_duration,
        seed=RANDOM_SEED + 10 * index + 3,
        flow_m3_h=q3,
    )

    if len(r3):
        stage3_light = r3[r3["dynamic_stream"].eq("light")].copy()
        stage3_heavy = r3[r3["dynamic_stream"].eq("heavy")].copy()
        stage3_middlings = r3[r3["dynamic_stream"].eq("active")].copy()
    else:
        stage3_light = r3.copy()
        stage3_heavy = r3.copy()
        stage3_middlings = r3.copy()

    # Final products combine unrecycled Stage-2 products with particles
    # resolved in Stage 3. No material label is used to assign a stream.
    aluminum_product = pd.concat(
        [stage2_light_kept, stage3_light],
        ignore_index=True,
        sort=False,
    )
    copper_product = pd.concat(
        [stage2_heavy_kept, stage3_heavy],
        ignore_index=True,
        sort=False,
    )
    metal_middlings = stage3_middlings.copy()

    stage2_al_purity = purity(stage2_light, "aluminum")
    final_al_purity = purity(aluminum_product, "aluminum")
    stage2_cu_purity = purity(stage2_heavy, "copper")
    final_cu_purity = purity(copper_product, "copper")

    print()
    print("=" * 86)
    print(f"FINAL SUMMARY — {fraction} mm")
    print("=" * 86)

    summary = pd.DataFrame([
        composition_row("Stage 1 light/nonmetal-rich", stage1_light),
        composition_row("Final heavy product (Cu-intended)", copper_product),
        composition_row("Final light product (Al-intended)", aluminum_product),
        composition_row("Metal middlings/recycle", metal_middlings),
        composition_row("Stage 1 middlings/recycle", stage1_middlings),
    ])
    print(summary.to_string(index=False))

    print()
    print("Cu/Al performance:")
    print(
        f"  Stage-2 light product Al purity: "
        f"{pct(stage2_al_purity)}"
    )
    print(
        f"  Final light product Al purity after Stage 3 middlings recycle: "
        f"{pct(final_al_purity)}"
    )
    print(
        f"  Final Al recovery to light product: "
        f"{pct(recovery(aluminum_product, initial, 'aluminum'))}"
    )
    print(
        f"  Stage-2 heavy product Cu purity: "
        f"{pct(stage2_cu_purity)}"
    )
    print(
        f"  Final heavy product Cu purity after Stage 3 middlings recycle: "
        f"{pct(final_cu_purity)}"
    )
    print(
        f"  Final Cu recovery to heavy product: "
        f"{pct(recovery(copper_product, initial, 'copper'))}"
    )

    if not stage3_feed.empty:
        print()
        print("Stage-3 overlap feed composition:")
        print(
            pd.crosstab(
                stage3_feed["material"],
                stage3_feed["liberation_state"],
            )
        )

    # Show explicitly why impurity exists.
    metal_rows = initial[
        initial["material"].isin(["aluminum", "copper"])
    ]
    if not metal_rows.empty:
        print()
        print("Feed metal suspension-velocity ranges:")
        diag = (
            metal_rows.groupby(["material", "liberation_state"])
            ["suspension_velocity_m_s"]
            .agg(["count", "min", "mean", "max"])
            .round(3)
        )
        print(diag)

    metrics = {
        "fraction_mm": fraction,
        "n_initial": len(initial),
        "sampled_size_min_mm": initial["size_mm"].min(),
        "sampled_size_max_mm": initial["size_mm"].max(),
        "stage2_flow_m3_h": q2,
        "stage3_flow_m3_h": q3,
        "stage2_light_al_purity": stage2_al_purity,
        "final_al_purity": final_al_purity,
        "final_al_recovery": recovery(aluminum_product, initial, "aluminum"),
        "stage2_heavy_cu_purity": stage2_cu_purity,
        "final_cu_purity": final_cu_purity,
        "final_cu_recovery": recovery(copper_product, initial, "copper"),
        "stage1_middlings": len(stage1_middlings),
        "metal_middlings": len(metal_middlings),
    }

    if args.save_results:
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        slug = fraction.replace(".", "p")
        initial.to_csv(out / f"{slug}_initial_feed.csv", index=False)
        r1.to_csv(out / f"{slug}_stage1.csv", index=False)
        r2.to_csv(out / f"{slug}_stage2.csv", index=False)
        r3.to_csv(out / f"{slug}_stage3.csv", index=False)
        summary.to_csv(out / f"{slug}_stream_summary.csv", index=False)

    return metrics


def main():
    args = parse_args()

    feed = generate_feed()

    fractions = (
        PNEUMATIC_FRACTION_ORDER
        if args.fraction == "all"
        else [args.fraction]
    )

    all_metrics = []

    for i, fraction in enumerate(fractions, start=1):
        all_metrics.append(
            run_fraction(args, feed, fraction, i)
        )

    if len(all_metrics) > 1:
        print()
        print("#" * 86)
        print("CROSS-SIZE-CLASS SUMMARY")
        print("#" * 86)
        table = pd.DataFrame(all_metrics)
        display_cols = [
            "fraction_mm",
            "n_initial",
            "sampled_size_min_mm",
            "sampled_size_max_mm",
            "stage2_flow_m3_h",
            "stage3_flow_m3_h",
            "stage2_light_al_purity",
            "final_al_purity",
            "final_al_recovery",
            "final_cu_purity",
            "final_cu_recovery",
            "metal_middlings",
        ]
        print(table[display_cols].to_string(index=False))

        if args.save_results:
            Path("outputs").mkdir(exist_ok=True)
            table.to_csv(
                "outputs/all_size_classes_metrics.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
