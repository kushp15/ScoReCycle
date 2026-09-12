"""
Dynamic separator geometry and numerical settings.

Geometry basis
--------------
The separator is represented as a 2-D vertical cross-section of the
variable-diameter column discussed in the Zhu pulsated-pneumatic separator
work.

The geometry used here is:
    lower straight section : D = 0.100 m, L = 0.500 m
    expansion section      : D = 0.100 -> 0.120 m, L = 0.125 m
    contraction section    : D = 0.120 -> 0.100 m, L = 0.125 m
    upper straight section : D = 0.100 m, L = 1.500 m

The exact feed-port height is not established by the literature inputs
currently encoded in this project. FEED_HEIGHT_M is therefore a transparent
simulation/design assumption, not a measured value from the paper.
"""

import numpy as np

# -------------------------
# Separator geometry
# -------------------------
D_LOWER_M = 0.100
D_EXPANDED_M = 0.120

L_LOWER_M = 0.500
L_EXPAND_M = 0.125
L_CONTRACT_M = 0.125
L_UPPER_M = 1.500

TOTAL_HEIGHT_M = (
    L_LOWER_M
    + L_EXPAND_M
    + L_CONTRACT_M
    + L_UPPER_M
)

# Feed injection is deliberately exposed as an assumption.
FEED_HEIGHT_M = 0.350
FEED_WIDTH_FRACTION = 0.70

# -------------------------
# Dynamic airflow
# -------------------------
# Reference velocity is defined in the 100-mm straight section.
DEFAULT_HIGH_VELOCITY_M_S = 2.50

# During valve "off", flow does not instantaneously become zero.
# This first model allows a residual-flow fraction.
DEFAULT_LOW_FLOW_FRACTION = 0.25

DEFAULT_PULSE_FREQUENCY_HZ = 4.0
DEFAULT_DUTY_CYCLE = 0.50

# First-order response of the gas flow to the commanded pulse.
# tau = 0 would be an ideal square wave.
AIR_RESPONSE_TAU_S = 0.040

# -------------------------
# Particle dynamics
# -------------------------
DEFAULT_N_PARTICLES = 500
DEFAULT_DURATION_S = 8.0
DEFAULT_DT_S = 0.002

# Wall collision: reflect the normal component and damp it.
WALL_RESTITUTION = 0.35

# Small random initial particle velocity represents feed-entry disturbance.
INITIAL_VERTICAL_VELOCITY_SD_M_S = 0.08
INITIAL_HORIZONTAL_VELOCITY_SD_M_S = 0.05

# Brownian motion is negligible at these particle sizes, so no Brownian
# forcing is included. A weak stochastic turbulent acceleration can be
# enabled later if supported/calibrated.
TURBULENT_ACCELERATION_SD_M_S2 = 0.0

# -------------------------
# Animation
# -------------------------
ANIMATION_FPS = 30
TRAIL_LENGTH = 0

MATERIAL_MARKERS = {
    "aluminum": "s",
    "copper": "s",
    "separator": "s",
    "black_mass": "o",
    "other_light": "o",
}

# Matplotlib marker size is an on-screen area in points^2. We map each
# parcel's measured size_mm into a readable screen-size range while keeping
# the relative particle-size distribution.
PARTICLE_VISUAL_MIN_PT2 = 10.0
PARTICLE_VISUAL_MAX_PT2 = 220.0
PARTICLE_VISUAL_POWER = 2.0

# Matplotlib will choose colors automatically. We intentionally do not
# hard-code colors so the normal matplotlib theme remains readable.
