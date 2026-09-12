import numpy as np
from config import g, rho_air, mu_air


def circular_area(diameter_m):
    return np.pi * diameter_m**2 / 4.0


def flow_rate_from_velocity(velocity_m_s, diameter_m):
    return velocity_m_s * circular_area(diameter_m) * 3600.0


def flake_equivalent_diameter(a_mm, b_mm, thickness_mm):
    a = np.asarray(a_mm, dtype=float) / 1000.0
    b = np.asarray(b_mm, dtype=float) / 1000.0
    H = np.asarray(thickness_mm, dtype=float) / 1000.0
    volume = a * b * H
    return (6.0 * volume / np.pi)**(1.0 / 3.0)


def flake_shape_factor_K(a_mm, b_mm, thickness_mm):
    a = np.asarray(a_mm, dtype=float) / 1000.0
    b = np.asarray(b_mm, dtype=float) / 1000.0
    H = np.asarray(thickness_mm, dtype=float) / 1000.0

    d_e = flake_equivalent_diameter(a_mm, b_mm, thickness_mm)

    surface_area = (
        2.0 * a * b
        + 2.0 * a * H
        + 2.0 * b * H
    )

    return surface_area / (np.pi * d_e**2)


def flake_suspension_velocity_eq17(
    effective_density_kg_m3,
    size_mm,
    total_thickness_mm,
    C1=1.77,
):
    """
    Zhu equivalent-diameter form used for the foil suspension model.

    For composite/coated fragments:
        - effective density comes from total areal mass / total thickness
        - total thickness includes metal + residual coating
    """
    d_e = flake_equivalent_diameter(
        size_mm,
        size_mm,
        total_thickness_mm,
    )

    K = flake_shape_factor_K(
        size_mm,
        size_mm,
        total_thickness_mm,
    )

    return 3.62 * np.sqrt(
        d_e * (effective_density_kg_m3 - rho_air)
        / (rho_air * C1 * K)
    )


def pressure_model_velocity_eq9(areal_density_kg_m2):
    """
    Diagnostic form of Zhu Eq. 9:

        u = sqrt(1.6 * g * rho * H)

    For a composite fragment, rho*H is replaced by total areal
    density (sum of each layer's rho_i * H_i).
    """
    return np.sqrt(1.6 * g * areal_density_kg_m2)


def drag_coefficient_haider_levenspiel(Re, phi):
    Re = np.maximum(np.asarray(Re, dtype=float), 1e-12)

    A = np.exp(2.3288 - 6.4581 * phi + 2.4486 * phi**2)
    B = 0.0964 + 0.5565 * phi
    C = np.exp(
        4.905
        - 13.8944 * phi
        + 18.4222 * phi**2
        - 10.2599 * phi**3
    )
    D = np.exp(
        1.4681
        + 12.2584 * phi
        - 20.7322 * phi**2
        + 15.8855 * phi**3
    )

    return (
        (24.0 / Re) * (1.0 + A * Re**B)
        + C / (1.0 + D / Re)
    )


def irregular_suspension_velocity(
    size_mm,
    density_kg_m3,
    sphericity,
    tolerance=1e-8,
    max_iterations=500,
):
    d = np.asarray(size_mm, dtype=float) / 1000.0
    rho_p = np.asarray(density_kg_m3, dtype=float)

    volume = np.pi * d**3 / 6.0
    projected_area = np.pi * d**2 / 4.0

    u = np.ones_like(d)

    for _ in range(max_iterations):
        Re = rho_air * u * d / mu_air
        Cd = drag_coefficient_haider_levenspiel(Re, sphericity)

        u_new = np.sqrt(
            2.0
            * (rho_p - rho_air)
            * volume
            * g
            / (Cd * rho_air * projected_area)
        )

        if np.max(np.abs(u_new - u)) < tolerance:
            return u_new

        u = 0.5 * u + 0.5 * u_new

    raise RuntimeError("Irregular-particle suspension solver did not converge.")
