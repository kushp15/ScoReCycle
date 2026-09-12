"""
Time-resolved 2-D pulsated pneumatic separator.

The key modeling idea is to preserve the literature-calibrated suspension
velocity already computed for every simulated particle.

Rather than inventing a new drag coefficient for every irregular flake,
this dynamic model calibrates a quadratic drag acceleration so that each
particle's terminal settling/suspension velocity is exactly the
suspension_velocity_m_s from the static literature-based model.

For a stationary gas and a downward-settling particle:

    dv/dt = -g_eff + k * |u_rel| * u_rel

with

    k = g_eff / u_s^2

where u_s is the particle's literature-calibrated suspension velocity.

Therefore, when the relative gas/particle speed equals u_s, drag and
effective gravity balance. This lets the time-domain model inherit the
Zhu/Bi/Haider-Levenspiel suspension physics already encoded in feed_model.py
without silently replacing it with a different particle-shape model.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import (
    g,
    rho_air,
    D_STRAIGHT,
    RANDOM_SEED,
)
from dynamic_config import (
    D_LOWER_M,
    D_EXPANDED_M,
    L_LOWER_M,
    L_EXPAND_M,
    L_CONTRACT_M,
    L_UPPER_M,
    TOTAL_HEIGHT_M,
    FEED_HEIGHT_M,
    FEED_WIDTH_FRACTION,
    DEFAULT_HIGH_VELOCITY_M_S,
    DEFAULT_LOW_FLOW_FRACTION,
    DEFAULT_PULSE_FREQUENCY_HZ,
    DEFAULT_DUTY_CYCLE,
    AIR_RESPONSE_TAU_S,
    DEFAULT_DT_S,
    DEFAULT_DURATION_S,
    WALL_RESTITUTION,
    INITIAL_VERTICAL_VELOCITY_SD_M_S,
    INITIAL_HORIZONTAL_VELOCITY_SD_M_S,
    TURBULENT_ACCELERATION_SD_M_S2,
)
from physics import circular_area


@dataclass
class DynamicSettings:
    high_velocity_m_s: float = DEFAULT_HIGH_VELOCITY_M_S
    low_flow_fraction: float = DEFAULT_LOW_FLOW_FRACTION
    pulse_frequency_hz: float = DEFAULT_PULSE_FREQUENCY_HZ
    duty_cycle: float = DEFAULT_DUTY_CYCLE
    dt_s: float = DEFAULT_DT_S
    duration_s: float = DEFAULT_DURATION_S
    air_response_tau_s: float = AIR_RESPONSE_TAU_S
    wall_restitution: float = WALL_RESTITUTION
    feed_height_m: float = FEED_HEIGHT_M

    def validate(self):
        if self.high_velocity_m_s < 0:
            raise ValueError("high_velocity_m_s must be >= 0")
        if not 0 <= self.low_flow_fraction <= 1:
            raise ValueError("low_flow_fraction must be between 0 and 1")
        if self.pulse_frequency_hz <= 0:
            raise ValueError("pulse_frequency_hz must be > 0")
        if not 0 < self.duty_cycle <= 1:
            raise ValueError("duty_cycle must be in (0, 1]")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        if self.air_response_tau_s < 0:
            raise ValueError("air_response_tau_s must be >= 0")
        if not 0 <= self.wall_restitution <= 1:
            raise ValueError("wall_restitution must be between 0 and 1")
        if not 0 < self.feed_height_m < TOTAL_HEIGHT_M:
            raise ValueError("feed_height_m must lie inside the separator")


def diameter_at_height(z_m):
    """
    Local separator diameter D(z).

    z=0 is the heavy-product/bottom outlet.
    z=TOTAL_HEIGHT_M is the light-product/top outlet.
    """
    z = np.asarray(z_m, dtype=float)

    z1 = L_LOWER_M
    z2 = z1 + L_EXPAND_M
    z3 = z2 + L_CONTRACT_M

    d = np.full_like(z, D_LOWER_M, dtype=float)

    # Expansion: 100 -> 120 mm
    mask_expand = (z >= z1) & (z < z2)
    if np.any(mask_expand):
        xi = (z[mask_expand] - z1) / L_EXPAND_M
        d[mask_expand] = (
            D_LOWER_M
            + xi * (D_EXPANDED_M - D_LOWER_M)
        )

    # Contraction: 120 -> 100 mm
    mask_contract = (z >= z2) & (z < z3)
    if np.any(mask_contract):
        xi = (z[mask_contract] - z2) / L_CONTRACT_M
        d[mask_contract] = (
            D_EXPANDED_M
            - xi * (D_EXPANDED_M - D_LOWER_M)
        )

    # Lower and upper sections remain 100 mm.
    return d


def radius_at_height(z_m):
    return diameter_at_height(z_m) / 2.0


def commanded_flow_rate_m3_s(t_s, settings: DynamicSettings):
    """
    Pulsed volumetric flow command.

    high_velocity_m_s is defined in the 100-mm reference section.
    """
    period = 1.0 / settings.pulse_frequency_hz
    phase = np.mod(t_s, period) / period

    high = settings.high_velocity_m_s
    low = high * settings.low_flow_fraction

    reference_velocity = (
        high if phase < settings.duty_cycle else low
    )

    return (
        reference_velocity
        * circular_area(D_STRAIGHT)
    )


def local_air_velocity_m_s(z_m, flow_rate_m3_s):
    """
    Continuity:
        u(z,t) = Q(t) / A(z)
    """
    d = diameter_at_height(z_m)
    area = np.pi * d**2 / 4.0
    return flow_rate_m3_s / area


def _sample_dynamic_particles(
    feed: pd.DataFrame,
    n_particles: int,
    rng: np.random.Generator,
    feed_height_m: float,
):
    """
    Draw simulation parcels from one selected comminution fraction.

    The source feed is already a mass-calibrated equal-mass Monte Carlo
    population, so uniform row sampling preserves its mass distribution.
    """
    if feed.empty:
        raise ValueError("Selected feed fraction is empty")

    n = min(int(n_particles), len(feed))
    chosen = feed.sample(
        n=n,
        replace=False,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).reset_index(drop=True)

    r_feed = float(radius_at_height(np.array([feed_height_m]))[0])
    half_width = FEED_WIDTH_FRACTION * r_feed

    state = {
        "x": rng.uniform(-half_width, half_width, n),
        "z": np.full(n, feed_height_m, dtype=float),
        "vx": rng.normal(0.0, INITIAL_HORIZONTAL_VELOCITY_SD_M_S, n),
        "vz": rng.normal(0.0, INITIAL_VERTICAL_VELOCITY_SD_M_S, n),
        "status": np.full(n, "active", dtype=object),
        "exit_time_s": np.full(n, np.nan),
    }

    return chosen, state


class DynamicSeparatorSimulation:
    def __init__(
        self,
        feed: pd.DataFrame,
        settings: DynamicSettings | None = None,
        n_particles: int = 500,
        seed: int = RANDOM_SEED,
    ):
        self.settings = settings or DynamicSettings()
        self.settings.validate()

        self.rng = np.random.default_rng(seed)
        self.particles, state = _sample_dynamic_particles(
            feed=feed,
            n_particles=n_particles,
            rng=self.rng,
            feed_height_m=self.settings.feed_height_m,
        )

        self.n = len(self.particles)
        self.x = state["x"]
        self.z = state["z"]
        self.vx = state["vx"]
        self.vz = state["vz"]
        self.status = state["status"]
        self.exit_time_s = state["exit_time_s"]

        self.time_s = 0.0

        # Gas state starts at the low part of the pulse so the first pulse
        # produces a finite transient rather than an instantaneous jump.
        high_q = (
            self.settings.high_velocity_m_s
            * circular_area(D_STRAIGHT)
        )
        self.flow_rate_m3_s = (
            high_q
            * self.settings.low_flow_fraction
        )

        # Particle properties needed by the dynamic force model.
        self.rho_p = (
            self.particles["effective_density_kg_m3"]
            .to_numpy(dtype=float)
        )
        self.u_s = (
            self.particles["suspension_velocity_m_s"]
            .to_numpy(dtype=float)
        )

        # Effective downward gravitational acceleration after buoyancy.
        self.g_eff = g * (
            1.0 - rho_air / self.rho_p
        )

        # Calibrated quadratic drag coefficient per unit mass:
        #     a_drag = k |u_rel| u_rel
        # chosen to recover each particle's existing u_s.
        self.drag_k = (
            self.g_eff
            / np.maximum(self.u_s, 1e-9) ** 2
        )

        self.history_time = []
        self.history_q = []
        self.history_reference_velocity = []
        self.history_active = []
        self.history_light = []
        self.history_heavy = []

    @property
    def active_mask(self):
        return self.status == "active"

    @property
    def light_mask(self):
        return self.status == "light"

    @property
    def heavy_mask(self):
        return self.status == "heavy"

    def _update_flow(self, dt):
        q_command = commanded_flow_rate_m3_s(
            self.time_s,
            self.settings,
        )

        tau = self.settings.air_response_tau_s
        if tau <= 0:
            self.flow_rate_m3_s = q_command
        else:
            # Exact first-order update for constant command over dt.
            alpha = 1.0 - np.exp(-dt / tau)
            self.flow_rate_m3_s += (
                alpha
                * (q_command - self.flow_rate_m3_s)
            )

    def _apply_wall_collisions(self, idx):
        """
        Reflect particles that cross the 2-D side wall.

        This is a geometric boundary model only. Wall friction, sticking,
        electrostatics and particle-particle collisions are not yet modeled.
        """
        if idx.size == 0:
            return

        radii = radius_at_height(self.z[idx])

        too_right = self.x[idx] > radii
        too_left = self.x[idx] < -radii

        if np.any(too_right):
            j = idx[too_right]
            rj = radii[too_right]
            self.x[j] = rj - 1e-6
            self.vx[j] = (
                -np.abs(self.vx[j])
                * self.settings.wall_restitution
            )

        if np.any(too_left):
            j = idx[too_left]
            rj = radii[too_left]
            self.x[j] = -rj + 1e-6
            self.vx[j] = (
                np.abs(self.vx[j])
                * self.settings.wall_restitution
            )

    def _update_outlets(self, idx):
        if idx.size == 0:
            return

        reached_top = self.z[idx] >= TOTAL_HEIGHT_M
        reached_bottom = self.z[idx] <= 0.0

        if np.any(reached_top):
            j = idx[reached_top]
            self.status[j] = "light"
            self.exit_time_s[j] = self.time_s
            self.z[j] = TOTAL_HEIGHT_M
            self.vx[j] = 0.0
            self.vz[j] = 0.0

        if np.any(reached_bottom):
            j = idx[reached_bottom]
            self.status[j] = "heavy"
            self.exit_time_s[j] = self.time_s
            self.z[j] = 0.0
            self.vx[j] = 0.0
            self.vz[j] = 0.0

    def step(self, dt=None):
        dt = (
            self.settings.dt_s
            if dt is None
            else float(dt)
        )

        self._update_flow(dt)

        active = self.active_mask
        idx = np.flatnonzero(active)

        if idx.size:
            z = self.z[idx]

            # Vertical gas velocity from local column area.
            u_air_z = local_air_velocity_m_s(
                z,
                self.flow_rate_m3_s,
            )

            # Gas has no horizontal mean velocity in this first model.
            rel_x = -self.vx[idx]
            rel_z = u_air_z - self.vz[idx]

            rel_mag = np.sqrt(
                rel_x**2 + rel_z**2
            )

            # Calibrated quadratic drag acceleration.
            ax = (
                self.drag_k[idx]
                * rel_mag
                * rel_x
            )

            az = (
                self.drag_k[idx]
                * rel_mag
                * rel_z
                - self.g_eff[idx]
            )

            if TURBULENT_ACCELERATION_SD_M_S2 > 0:
                ax += self.rng.normal(
                    0.0,
                    TURBULENT_ACCELERATION_SD_M_S2,
                    idx.size,
                )
                az += self.rng.normal(
                    0.0,
                    TURBULENT_ACCELERATION_SD_M_S2,
                    idx.size,
                )

            # Semi-implicit Euler:
            # update velocity first, then position.
            self.vx[idx] += ax * dt
            self.vz[idx] += az * dt

            self.x[idx] += self.vx[idx] * dt
            self.z[idx] += self.vz[idx] * dt

            self._apply_wall_collisions(idx)
            self._update_outlets(idx)

        self.time_s += dt

        self.history_time.append(self.time_s)
        self.history_q.append(self.flow_rate_m3_s)
        self.history_reference_velocity.append(
            self.flow_rate_m3_s
            / circular_area(D_STRAIGHT)
        )
        self.history_active.append(int(np.sum(self.active_mask)))
        self.history_light.append(int(np.sum(self.light_mask)))
        self.history_heavy.append(int(np.sum(self.heavy_mask)))

    def run(self, duration_s=None):
        duration = (
            self.settings.duration_s
            if duration_s is None
            else float(duration_s)
        )

        n_steps = int(np.ceil(duration / self.settings.dt_s))

        for _ in range(n_steps):
            if not np.any(self.active_mask):
                break
            self.step()

        return self.results()

    def results(self):
        out = self.particles.copy()
        out["x_m"] = self.x
        out["z_m"] = self.z
        out["vx_m_s"] = self.vx
        out["vz_m_s"] = self.vz
        out["dynamic_stream"] = self.status
        out["exit_time_s"] = self.exit_time_s
        return out

    def performance_summary(self):
        result = self.results()

        def frac(mask):
            return float(np.mean(mask)) if len(result) else 0.0

        light_materials = {
            "separator",
            "black_mass",
            "other_light",
        }

        is_light_material = (
            result["material"]
            .isin(light_materials)
            .to_numpy()
        )
        is_metal = (
            result["material"]
            .isin({"aluminum", "copper"})
            .to_numpy()
        )
        to_light = (
            result["dynamic_stream"]
            .eq("light")
            .to_numpy()
        )
        to_heavy = (
            result["dynamic_stream"]
            .eq("heavy")
            .to_numpy()
        )

        def conditional(numerator, denominator):
            denom = np.sum(denominator)
            if denom == 0:
                return np.nan
            return float(
                np.sum(numerator & denominator)
                / denom
            )

        light_total = np.sum(to_light)
        light_purity = (
            float(
                np.sum(to_light & is_light_material)
                / light_total
            )
            if light_total > 0
            else np.nan
        )

        return {
            "simulated_particles": len(result),
            "time_s": self.time_s,
            "light_exit_fraction": frac(to_light),
            "heavy_exit_fraction": frac(to_heavy),
            "still_active_fraction": frac(
                result["dynamic_stream"]
                .eq("active")
                .to_numpy()
            ),
            "stage1_light_material_recovery":
                conditional(
                    to_light,
                    is_light_material,
                ),
            "stage1_light_stream_purity":
                light_purity,
            "stage1_metal_retention_heavy":
                conditional(
                    to_heavy,
                    is_metal,
                ),
        }

    def material_outlet_table(self):
        result = self.results()

        table = (
            result.groupby(
                ["material", "dynamic_stream"]
            )
            .size()
            .unstack(fill_value=0)
        )

        return table
