import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    N_SIMULATION_PARCELS,
    SOURCE_SUBFRACTIONS,
    MATERIAL_PROBABILITIES_BY_SUBFRACTION,
    FOIL_ENDPOINTS,
    SEPARATOR,
    BARE_PROBABILITY,
    RESIDUAL_COATING_MAX_FRACTION,
    RESIDUAL_COATING_MODE_FRACTION,
    BLACK_MASS_DENSITY_SUPPORT_KG_M3,
    BLACK_MASS_SPHERICITY,
    BLACK_MASS_AERO_DIAMETER_MIN_MM,
    BLACK_MASS_AERO_DIAMETER_MAX_MM,
    OTHER_LIGHT_DENSITY_KG_M3,
    OTHER_LIGHT_SPHERICITY,
    OTHER_LIGHT_AERO_DIAMETER_MIN_MM,
    OTHER_LIGHT_AERO_DIAMETER_MAX_MM,
)
from physics import (
    flake_suspension_velocity_eq17,
    pressure_model_velocity_eq9,
    irregular_suspension_velocity,
)

rng = np.random.default_rng(RANDOM_SEED)


def _normalized_subfraction_probabilities():
    names = list(SOURCE_SUBFRACTIONS)
    raw = np.array([
        SOURCE_SUBFRACTIONS[name]["raw_yield_wt_percent"]
        for name in names
    ], dtype=float)

    return names, raw / raw.sum()


def _choose_material(subfraction):
    probs_dict = MATERIAL_PROBABILITIES_BY_SUBFRACTION[
        subfraction
    ]

    materials = list(probs_dict)
    probs = np.array(
        [probs_dict[m] for m in materials],
        dtype=float,
    )
    probs = probs / probs.sum()

    return rng.choice(materials, p=probs)


def _sample_size_mm(subfraction):
    cfg = SOURCE_SUBFRACTIONS[subfraction]

    # The literature provides sieve bins rather than a continuous
    # within-bin PDF. Uniform sampling is the minimum-assumption
    # interpolation inside the measured interval.
    return rng.uniform(
        cfg["low_mm"],
        cfg["high_mm"],
    )


def _foil_row(material, subfraction, size_mm):
    """
    Generate one Cu/Al foil fragment using the original Eq. 17 flake theory.

    The measured Bi et al. bare and fully electrode-bound endpoints are kept.
    The change here is only how much residual electrode is allowed to remain
    after the Zhu-style comminution/peeling step.

    Residual coating is sampled continuously rather than as a binary
    bare-vs-fully-coated state.

    The post-comminution coating envelope is a transparent calibration, not a
    directly measured probability distribution. It is constrained so the
    population behavior matches the experimentally reported direction:
        Al preferentially light/up,
        Cu preferentially heavy/down,
    while still retaining the large-Al / small-Cu overlap.
    """
    endpoint = FOIL_ENDPOINTS[material]

    max_fraction = RESIDUAL_COATING_MAX_FRACTION[material]
    mode_fraction = (
        RESIDUAL_COATING_MODE_FRACTION * max_fraction
    )

    residual_coating_fraction = float(
        rng.triangular(
            0.0,
            mode_fraction,
            max_fraction,
        )
    )

    # Descriptive state labels only. They do not determine particle routing.
    if residual_coating_fraction < 0.05:
        liberation_state = "bare_like"
    elif residual_coating_fraction < 0.60 * max_fraction:
        liberation_state = "partial"
    else:
        liberation_state = "heavily_coated"

    bare_density = endpoint["bare_density"]
    bare_thickness_mm = endpoint["bare_thickness_mm"]
    composite_density = endpoint["composite_density"]
    composite_thickness_mm = endpoint["composite_thickness_mm"]

    # Interpolate total thickness between the measured bare and fully coated
    # physical endpoints.
    total_thickness_mm = (
        bare_thickness_mm
        + residual_coating_fraction
        * (composite_thickness_mm - bare_thickness_mm)
    )

    # Interpolate areal density between the same measured endpoints.
    bare_areal_density = (
        bare_density * bare_thickness_mm / 1000.0
    )
    composite_areal_density = (
        composite_density * composite_thickness_mm / 1000.0
    )
    areal_density = (
        bare_areal_density
        + residual_coating_fraction
        * (composite_areal_density - bare_areal_density)
    )

    effective_density = (
        areal_density / (total_thickness_mm / 1000.0)
    )

    # ORIGINAL THEORY retained.
    u_eq17 = float(
        flake_suspension_velocity_eq17(
            effective_density,
            size_mm,
            total_thickness_mm,
        )
    )

    u_eq9 = float(
        pressure_model_velocity_eq9(
            areal_density
        )
    )

    side_m = size_mm / 1000.0
    physical_mass = areal_density * side_m**2

    return {
        "material": material,
        "liberation_state": liberation_state,
        "residual_coating_fraction": residual_coating_fraction,
        "size_mm": size_mm,
        "aerodynamic_size_mm": size_mm,
        "metal_thickness_mm": bare_thickness_mm,
        "coating_thickness_mm": max(
            total_thickness_mm - bare_thickness_mm,
            0.0,
        ),
        "total_thickness_mm": total_thickness_mm,
        "metal_density_kg_m3": bare_density,
        "coating_density_kg_m3": np.nan,
        "effective_density_kg_m3": effective_density,
        "areal_density_kg_m2": areal_density,
        "suspension_velocity_m_s": u_eq17,
        "eq9_velocity_m_s": u_eq9,
        "physical_particle_mass_kg": physical_mass,
    }


def _separator_row(size_mm):
    density = SEPARATOR["density_kg_m3"]
    thickness_mm = SEPARATOR["thickness_mm"]

    areal_density = density * thickness_mm / 1000.0

    # Separator fragments are thin films. We use the same
    # equivalent-diameter flake framework as a first aerodynamic
    # approximation, but retain Eq. 9 as an independent diagnostic.
    u_eq17 = float(
        flake_suspension_velocity_eq17(
            density,
            size_mm,
            thickness_mm,
        )
    )

    u_eq9 = float(
        pressure_model_velocity_eq9(
            areal_density
        )
    )

    side_m = size_mm / 1000.0
    physical_mass = areal_density * side_m**2

    return {
        "material": "separator",
        "liberation_state": "film",
        "size_mm": size_mm,
        "aerodynamic_size_mm": size_mm,
        "metal_thickness_mm": np.nan,
        "coating_thickness_mm": np.nan,
        "total_thickness_mm": thickness_mm,
        "metal_density_kg_m3": np.nan,
        "coating_density_kg_m3": np.nan,
        "effective_density_kg_m3": density,
        "areal_density_kg_m2": areal_density,
        "suspension_velocity_m_s": u_eq17,
        "eq9_velocity_m_s": u_eq9,
        "physical_particle_mass_kg": physical_mass,
    }


def _black_mass_row(size_mm):
    density = float(
        rng.choice(BLACK_MASS_DENSITY_SUPPORT_KG_M3)
    )

    # IMPORTANT:
    # "size_mm" is the measured comminution/sieve size and is kept intact.
    # It is NOT used as the aerodynamic diameter for liberated black mass.
    #
    # Black mass is a fine powder / porous-agglomerate phase. Treating a
    # 0.25-0.50 mm sieve-class agglomerate as a compact 0.25-0.50 mm dense
    # sphere makes its predicted suspension velocity much too high and causes
    # it to settle, which is contrary to the Stage-1 pneumatic behavior.
    #
    # Use a separate aerodynamic-equivalent diameter for drag. The present
    # 40-100 µm range is an explicit calibration assumption and is exposed in
    # config.py so it can be replaced by measured aerodynamic-size data later.
    aerodynamic_size_mm = min(
        size_mm,
        float(
            rng.uniform(
                BLACK_MASS_AERO_DIAMETER_MIN_MM,
                BLACK_MASS_AERO_DIAMETER_MAX_MM,
            )
        ),
    )

    u = float(
        irregular_suspension_velocity(
            np.array([aerodynamic_size_mm]),
            np.array([density]),
            BLACK_MASS_SPHERICITY,
        )[0]
    )

    # Physical mass remains based on the measured/agglomerate sieve size.
    # The aerodynamic diameter is only an effective drag representation.
    d_m = size_mm / 1000.0
    physical_mass = density * np.pi * d_m**3 / 6.0

    return {
        "material": "black_mass",
        "liberation_state": "free",
        "size_mm": size_mm,
        "aerodynamic_size_mm": aerodynamic_size_mm,
        "metal_thickness_mm": np.nan,
        "coating_thickness_mm": np.nan,
        "total_thickness_mm": np.nan,
        "metal_density_kg_m3": np.nan,
        "coating_density_kg_m3": np.nan,
        "effective_density_kg_m3": density,
        "areal_density_kg_m2": np.nan,
        "suspension_velocity_m_s": u,
        "eq9_velocity_m_s": np.nan,
        "physical_particle_mass_kg": physical_mass,
    }


def _other_light_row(size_mm):
    density = OTHER_LIGHT_DENSITY_KG_M3

    # Keep the measured sieve size for the feed distribution, but do not
    # assume it equals the aerodynamic diameter of an irregular nonmetal
    # fragment.
    aerodynamic_size_mm = min(
        size_mm,
        float(
            rng.uniform(
                OTHER_LIGHT_AERO_DIAMETER_MIN_MM,
                OTHER_LIGHT_AERO_DIAMETER_MAX_MM,
            )
        ),
    )

    u = float(
        irregular_suspension_velocity(
            np.array([aerodynamic_size_mm]),
            np.array([density]),
            OTHER_LIGHT_SPHERICITY,
        )[0]
    )

    d_m = size_mm / 1000.0
    physical_mass = density * np.pi * d_m**3 / 6.0

    return {
        "material": "other_light",
        "liberation_state": "unresolved_nonmetal",
        "size_mm": size_mm,
        "aerodynamic_size_mm": aerodynamic_size_mm,
        "metal_thickness_mm": np.nan,
        "coating_thickness_mm": np.nan,
        "total_thickness_mm": np.nan,
        "metal_density_kg_m3": np.nan,
        "coating_density_kg_m3": np.nan,
        "effective_density_kg_m3": density,
        "areal_density_kg_m2": np.nan,
        "suspension_velocity_m_s": u,
        "eq9_velocity_m_s": np.nan,
        "physical_particle_mass_kg": physical_mass,
    }


def generate_feed():
    """
    Generate a literature-calibrated pneumatic feed.

    Important:
    ----------
    These rows are equal-mass simulation parcels, not literal physical
    particle counts. Published sieve and component data are mass-based.

    Workflow:
        1. sample measured comminution size subfraction by mass
        2. sample material conditional on that size fraction
        3. sample size only inside that measured sieve interval
        4. assign literature-derived material properties
    """
    subfraction_names, subfraction_probs = (
        _normalized_subfraction_probabilities()
    )

    selected_subfractions = rng.choice(
        subfraction_names,
        size=N_SIMULATION_PARCELS,
        p=subfraction_probs,
    )

    rows = []

    for subfraction in selected_subfractions:
        size_mm = _sample_size_mm(subfraction)
        material = _choose_material(subfraction)

        if material in ("aluminum", "copper"):
            row = _foil_row(
                material,
                subfraction,
                size_mm,
            )
        elif material == "separator":
            row = _separator_row(size_mm)
        elif material == "black_mass":
            row = _black_mass_row(size_mm)
        elif material == "other_light":
            row = _other_light_row(size_mm)
        else:
            raise ValueError(
                f"Unsupported material: {material}"
            )

        row["source_subfraction"] = subfraction
        row["pneumatic_fraction"] = (
            SOURCE_SUBFRACTIONS[subfraction][
                "pneumatic_fraction"
            ]
        )

        rows.append(row)

    feed = pd.DataFrame(rows)

    # Equal mass per Monte Carlo parcel makes the empirical frequencies
    # reproduce the literature mass fractions directly.
    feed["statistical_mass"] = (
        1.0 / len(feed)
    )

    # No synthetic "hammer mill" split is performed anymore.
    feed["mill_fraction"] = "pneumatic_feed"

    return feed
