import numpy as np

# ============================================================
# GLOBAL SIMULATION SETTINGS
# ============================================================

RANDOM_SEED = 42

# These are equal-mass Monte Carlo "parcels", not literal particle counts.
# Published sieve fractions are mass fractions, so equal-mass parcels let
# the simulation reproduce the literature mass distributions directly.
N_SIMULATION_PARCELS = 60_000

SAVE_OUTPUTS = False
OUTPUT_SUMMARY_FILE = "simulation_summary.csv"
OUTPUT_PARTICLE_FILE = "latest_particle_results.csv"

# ============================================================
# FLUID / SEPARATOR
# ============================================================

g = 9.81
rho_air = 1.204
mu_air = 1.81e-5

D_STRAIGHT = 0.100
D_EXPANDED = 0.120

VELOCITY_MIN = 0.5
VELOCITY_MAX = 8.0
N_VELOCITIES = 150
AIR_VELOCITIES = np.linspace(
    VELOCITY_MIN,
    VELOCITY_MAX,
    N_VELOCITIES,
)

# ============================================================
# PNEUMATIC SIZE CLASSES — ZHU ET AL. 2021
# ============================================================
#
# This project now uses the sieve distribution reported directly in
# Zhu et al. (Waste Management 131, 2021), Fig. 3(a):
#
#     > 2.0 mm       :  4.18 wt%
#     1.5-2.0 mm     : 18.23 wt%
#     1.0-1.5 mm     : 19.11 wt%
#     0.5-1.0 mm     :  7.47 wt%
#     0.25-0.5 mm    :  0.68 wt%
#     < 0.25 mm      : 50.33 wt%
#
# Zhu combined 0.25-0.5 and 0.5-1.0 mm into the 0.25-1.0 mm
# pneumatic-separation class because the 0.25-0.5 mm fraction was small
# and the two fractions had similar components.
#
# We simulate ONLY the three pneumatic classes reported in the paper:
#
#     0.25-1.00 mm
#     1.00-1.50 mm
#     1.50-2.00 mm
#
# IMPORTANT:
# The sieve boundaries and class mass frequencies are direct inputs from
# Zhu 2021. The paper does not publish a continuous within-bin PDF, so
# particle size is sampled uniformly INSIDE each exact sieve interval.
# That uniform interpolation is a minimum-assumption numerical choice;
# it is not presented as measured sub-bin data.
# ============================================================

SOURCE_SUBFRACTIONS = {
    "0.25-0.50": {
        "low_mm": 0.25,
        "high_mm": 0.50,
        "raw_yield_wt_percent": 0.68,
        "pneumatic_fraction": "0.25-1.00",
        "size_source": "Zhu et al. 2021 Fig. 3(a)",
    },
    "0.50-1.00": {
        "low_mm": 0.50,
        "high_mm": 1.00,
        "raw_yield_wt_percent": 7.47,
        "pneumatic_fraction": "0.25-1.00",
        "size_source": "Zhu et al. 2021 Fig. 3(a)",
    },
    "1.00-1.50": {
        "low_mm": 1.00,
        "high_mm": 1.50,
        "raw_yield_wt_percent": 19.11,
        "pneumatic_fraction": "1.00-1.50",
        "size_source": "Zhu et al. 2021 Fig. 3(a)",
    },
    "1.50-2.00": {
        "low_mm": 1.50,
        "high_mm": 2.00,
        "raw_yield_wt_percent": 18.23,
        "pneumatic_fraction": "1.50-2.00",
        "size_source": "Zhu et al. 2021 Fig. 3(a)",
    },
}

PNEUMATIC_FRACTION_ORDER = [
    "0.25-1.00",
    "1.00-1.50",
    "1.50-2.00",
]

# Zhu 2021 reports the following size-class-specific metal-separation
# air volumes in the mass-balance/conclusion:
#     0.25-1.00 mm -> 75 m^3/h
#     1.00-1.50 mm -> 84 m^3/h
#     1.50-2.00 mm -> 89 m^3/h
#
# These are used as the PRIMARY metal-cut reference flows in Stage 2.
ZHU_PRIMARY_METAL_FLOW_M3_H = {
    "0.25-1.00": 75.0,
    "1.00-1.50": 84.0,
    "1.50-2.00": 89.0,
}

# Stage 3 is a cleaning pass on the Stage-2 LIGHT product, matching Zhu's
# statement that recovered light products can be separated in cycles to
# improve Cu recovery and Al grade.
#
# To avoid inventing a new operating point, the cleaning flows below are
# chosen only from the experimental airflow grid shown in Zhu Fig. 4.
# We use a lower tested cut so Cu contamination is more likely to report
# downward while easier-to-entrain Al remains in the light product.
#
# This stage-3 selection is OUR PROCESS-DESIGN CHOICE constrained to
# airflow values actually tested in the paper; it is not claimed as an
# experimentally optimized third-stage setting.
ZHU_STAGE3_CLEANING_FLOW_M3_H = {
    "0.25-1.00": 60.0,
    "1.00-1.50": 65.0,
    "1.50-2.00": 70.0,
}

# ============================================================
# MATERIAL COMPOSITION CONDITIONAL ON SIZE
# ============================================================
#
# This is deliberately conditional:
#
#       P(material | size fraction)
#
# rather than independently choosing material and size.
#
# 0.50–1.00 and 1–2 mm:
#   Direct component mass percentages from the published table
#   "Component content in the size fractions after shredding"
#   (mobile-phone LIB mechanical preparation study):
#
#   0.5–1 mm:
#       separator 9.68%, Al 64.51%, Cu 13.98%,
#       plastic/others 11.83%
#
#   1–2 mm:
#       separator 14.66%, Al 47.41%, Cu 26.72%,
#       plastic/others 11.21%
#
# 0.25–0.50 mm:
#   Zhang et al. 2014 measured 50.20 wt% Cu and 10.80 wt% Al.
#   The remaining 39.00 wt% is nonmetallic / active-material-rich.
#   To retain an explicit separator class, 9.68 percentage points
#   are assigned to separator using the nearest directly measured
#   0.5–1 mm separator fraction. The remaining 29.32% is represented
#   as free active-material-rich black mass.
#
# This final split is a transparent cross-source interpolation.
# It is intentionally documented here rather than hidden.
# ============================================================

MATERIAL_PROBABILITIES_BY_SUBFRACTION = {
    "0.25-0.50": {
        "aluminum": 0.1080,
        "copper": 0.5020,
        "separator": 0.0968,
        "black_mass": 0.2932,
    },
    "0.50-1.00": {
        "aluminum": 0.6451,
        "copper": 0.1398,
        "separator": 0.0968,
        "other_light": 0.1183,
    },
    # Source reports 1–2 mm as a single class, so both Zhu sub-bins
    # inherit the same measured composition.
    "1.00-1.50": {
        "aluminum": 0.4741,
        "copper": 0.2672,
        "separator": 0.1466,
        "other_light": 0.1121,
    },
    "1.50-2.00": {
        "aluminum": 0.4741,
        "copper": 0.2672,
        "separator": 0.1466,
        "other_light": 0.1121,
    },
}

# ============================================================
# MEASURED FOIL / COMPOSITE ENDPOINTS
# Bi et al., Waste Management & Research 37(4), 2019
# ============================================================

FOIL_ENDPOINTS = {
    "aluminum": {
        "bare_density": 2700.0,
        "bare_thickness_mm": 0.0218,
        "composite_density": 2130.0,
        "composite_thickness_mm": 0.1790,
    },
    "copper": {
        "bare_density": 8900.0,
        "bare_thickness_mm": 0.0120,
        "composite_density": 1600.0,
        "composite_thickness_mm": 0.1580,
    },
}

SEPARATOR = {
    "density_kg_m3": 900.0,
    "thickness_mm": 0.0139,
}

# ============================================================
# SIZE-DEPENDENT LIBERATION PROBABILITY
# ============================================================
#
# Vanderbruggen et al. automated-mineralogy data, "recycling
# process liberation" (B):
#
#   125–500 µm:
#       Al foil liberated 16.4%
#       Cu foil liberated 57.9%
#
#   500–1000 µm:
#       Al foil liberated 17.2%
#       Cu foil liberated 58.2%
#
# We use those values as P(bare foil | material, size fraction).
#
# Their reported liberation characterization used here ends at 1 mm.
# Therefore the 1.0–1.5 and 1.5–2.0 mm classes use the nearest measured
# 0.5–1.0 mm liberation value. This remains an explicit nearest-bin
# extrapolation; the underlying liberation model is otherwise unchanged.
#
# Non-liberated foil is represented using the directly measured
# Bi "metal + electrode" endpoint. We intentionally removed the
# old unsupported 40/40/20 bare/partial/coated probabilities.
# ============================================================

BARE_PROBABILITY = {
    "0.25-0.50": {
        "aluminum": 0.164,
        "copper": 0.579,
    },
    "0.50-1.00": {
        "aluminum": 0.172,
        "copper": 0.582,
    },
    "1.00-1.50": {
        "aluminum": 0.172,
        "copper": 0.582,
    },
    "1.50-2.00": {
        "aluminum": 0.172,
        "copper": 0.582,
    },
}

# ============================================================
# POST-COMMINUTION RESIDUAL-COATING CALIBRATION
# ============================================================
#
# IMPORTANT:
# These are MODEL CALIBRATION ENVELOPES, not direct measured coating-fraction
# distributions.
#
# Why they exist:
# - Zhu 2021 reports that hammer milling peels electrode powder from the foils
#   before the >0.25-mm Cu/Al fractions are pneumatically separated.
# - Zhu's measured process direction is Al-rich light product and Cu-rich
#   heavy product, with overlap caused by particle size.
# - The fully electrode-bound Bi endpoints are retained as physical endpoint
#   references, but using them too often makes the modeled Al population
#   aerodynamically heavier than Cu, which does not match Zhu's population-level
#   separator behavior.
#
# We therefore preserve the same measured bare/composite endpoints and the same
# Eq. 17 suspension theory, but limit how much of the FULL endpoint coating is
# allowed to remain after comminution.
#
# The values below were chosen as a qualitative calibration so that:
#   1) Cu has the higher population-average suspension velocity,
#   2) the fastest Cu remains faster than the fastest Al,
#   3) large/partially coated Al still overlaps small Cu,
#   4) neither product is forced to be pure.
#
# They should be replaced by measured post-comminution coating-retention data
# if/when such data become available.
#
RESIDUAL_COATING_MAX_FRACTION = {
    "aluminum": 0.45,
    "copper": 0.80,
}

# Triangular distribution mode expressed as a fraction of each material's
# allowed residual-coating envelope. A low mode reflects that the Zhu feed has
# already undergone active-material peeling before pneumatic separation.
RESIDUAL_COATING_MODE_FRACTION = 0.10


# ============================================================
# NONMETALLIC PARTICLE PROPERTIES
# ============================================================
#
# Free black mass:
# We use an empirical discrete density mixture instead of one
# invented normal distribution:
#
#   2.10 g/cm3 : midpoint of 1.9–2.3 g/cm3 nonmetal range
#                reported for anode particles by Zhu et al. 2011
#
#   3.06 and 3.54 g/cm3 : measured skeletal densities of
#                industrial NMC black masses (Gerold et al. 2026)
#
# Each value is treated as an empirical support point.
#
# "other_light" corresponds to the published "plastic and others"
# category. Its exact composition is not resolved in that paper.
# It is therefore kept as a separate uncertainty class rather than
# silently calling it black mass.
# ============================================================

BLACK_MASS_DENSITY_SUPPORT_KG_M3 = np.array([
    2100.0,
    3060.0,
    3540.0,
])

BLACK_MASS_SPHERICITY = 0.75

# Black mass must NOT use the coarse sieve opening as its aerodynamic diameter.
# Liberated electrode material behaves as fine powder / porous agglomerates.
# We therefore keep the measured sieve size for feed statistics, but give black
# mass a separate aerodynamic-equivalent size used only in the drag calculation.
#
# Version-1 calibration assumption:
# aerodynamic-equivalent diameter is sampled from 0.04-0.10 mm and is capped
# by the measured sieve size. This is deliberately exposed as a calibration
# parameter rather than being presented as a direct literature measurement.
#
# This change makes the dynamic model reflect the observed Stage-1 behavior:
# liberated black mass is readily entrained upward while Cu/Al foil fragments
# require substantially larger gas velocities.
BLACK_MASS_AERO_DIAMETER_MIN_MM = 0.040
BLACK_MASS_AERO_DIAMETER_MAX_MM = 0.100


OTHER_LIGHT_DENSITY_KG_M3 = 1100.0
OTHER_LIGHT_SPHERICITY = 0.80

# The unresolved light/nonmetal fraction should not be treated as a compact
# sphere whose aerodynamic diameter equals its sieve size. That assumption
# gave it an unrealistically high suspension velocity.
#
# Keep the measured sieve size for the feed distribution, but use a separate
# aerodynamic-equivalent diameter in the drag model. These are transparent
# calibration parameters until direct aerodynamic measurements are available.
OTHER_LIGHT_AERO_DIAMETER_MIN_MM = 0.080
OTHER_LIGHT_AERO_DIAMETER_MAX_MM = 0.200

# ============================================================
# STAGE DEFINITIONS
# ============================================================

LIGHT_STAGE1_MATERIALS = {
    "separator",
    "black_mass",
    "other_light",
}

METAL_MATERIALS = {
    "aluminum",
    "copper",
}
