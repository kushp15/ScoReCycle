import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def suspension_diagnostic_table(feed):
    """
    Summarize the particle properties that control pneumatic separation.

    Foils are grouped by material + liberation state.
    Irregular particles are grouped by material.

    The most important outputs are:
      - particle size
      - total foil thickness
      - areal density rho*H
      - effective density
      - Eq. 17 suspension velocity
      - Eq. 9 diagnostic velocity
    """
    df = feed.copy()

    # Give irregular materials a readable state label.
    df["diagnostic_state"] = np.where(
        df["material"].isin(["aluminum", "copper", "separator"]),
        df["liberation_state"],
        "free",
    )

    grouped = df.groupby(
        ["material", "diagnostic_state"],
        dropna=False,
    )

    rows = []

    for (material, state), group in grouped:
        row = {
            "material": material,
            "state": state,
            "particle_count": len(group),
            "statistical_mass_fraction": group["statistical_mass"].sum(),

            "size_mean_mm": group["size_mm"].mean(),
            "size_median_mm": group["size_mm"].median(),
            "size_min_mm": group["size_mm"].min(),
            "size_max_mm": group["size_mm"].max(),

            "effective_density_mean_kg_m3":
                group["effective_density_kg_m3"].mean(),

            "u_s_mean_m_s":
                group["suspension_velocity_m_s"].mean(),
            "u_s_median_m_s":
                group["suspension_velocity_m_s"].median(),
            "u_s_min_m_s":
                group["suspension_velocity_m_s"].min(),
            "u_s_max_m_s":
                group["suspension_velocity_m_s"].max(),
        }

        # Foil-specific quantities. For black mass and PE these remain NaN.
        if group["total_thickness_mm"].notna().any():
            row.update({
                "thickness_mean_mm":
                    group["total_thickness_mm"].mean(),
                "thickness_median_mm":
                    group["total_thickness_mm"].median(),

                "rhoH_mean_kg_m2":
                    group["areal_density_kg_m2"].mean(),
                "rhoH_median_kg_m2":
                    group["areal_density_kg_m2"].median(),
                "rhoH_min_kg_m2":
                    group["areal_density_kg_m2"].min(),
                "rhoH_max_kg_m2":
                    group["areal_density_kg_m2"].max(),

                "eq9_velocity_mean_m_s":
                    group["eq9_velocity_m_s"].mean(),
                "eq9_velocity_median_m_s":
                    group["eq9_velocity_m_s"].median(),
            })
        else:
            row.update({
                "thickness_mean_mm": np.nan,
                "thickness_median_mm": np.nan,
                "rhoH_mean_kg_m2": np.nan,
                "rhoH_median_kg_m2": np.nan,
                "rhoH_min_kg_m2": np.nan,
                "rhoH_max_kg_m2": np.nan,
                "eq9_velocity_mean_m_s": np.nan,
                "eq9_velocity_median_m_s": np.nan,
            })

        rows.append(row)

    table = pd.DataFrame(rows)

    column_order = [
        "material",
        "state",
        "particle_count",
        "statistical_mass_fraction",

        "size_mean_mm",
        "size_median_mm",
        "size_min_mm",
        "size_max_mm",

        "thickness_mean_mm",
        "thickness_median_mm",

        "rhoH_mean_kg_m2",
        "rhoH_median_kg_m2",
        "rhoH_min_kg_m2",
        "rhoH_max_kg_m2",

        "effective_density_mean_kg_m3",

        "u_s_mean_m_s",
        "u_s_median_m_s",
        "u_s_min_m_s",
        "u_s_max_m_s",

        "eq9_velocity_mean_m_s",
        "eq9_velocity_median_m_s",
    ]

    return table[column_order]


def print_suspension_diagnostics(feed):
    """
    Print a compact diagnostic table to the terminal.
    """
    table = suspension_diagnostic_table(feed)

    display_columns = [
        "material",
        "state",
        "particle_count",
        "size_mean_mm",
        "thickness_mean_mm",
        "rhoH_mean_kg_m2",
        "effective_density_mean_kg_m3",
        "u_s_mean_m_s",
        "u_s_median_m_s",
        "u_s_min_m_s",
        "u_s_max_m_s",
        "eq9_velocity_mean_m_s",
    ]

    print()
    print("=" * 100)
    print("SUSPENSION-VELOCITY / rho*H DIAGNOSTICS")
    print("=" * 100)

    print(
        table[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return table


def print_fraction_below_velocity(feed, air_velocity):
    """
    Show how much of each material/state has suspension velocity
    below a selected air velocity and would therefore be entrained
    by the current deterministic separator rule.
    """
    df = feed.copy()

    df["diagnostic_state"] = np.where(
        df["material"].isin(["aluminum", "copper", "separator"]),
        df["liberation_state"],
        "free",
    )

    df["entrained_at_selected_velocity"] = (
        air_velocity >= df["suspension_velocity_m_s"]
    )

    rows = []

    for (material, state), group in df.groupby(
        ["material", "diagnostic_state"]
    ):
        total_mass = group["statistical_mass"].sum()

        entrained_mass = group.loc[
            group["entrained_at_selected_velocity"],
            "statistical_mass",
        ].sum()

        fraction = (
            entrained_mass / total_mass
            if total_mass > 0
            else 0.0
        )

        rows.append({
            "material": material,
            "state": state,
            "mass_fraction_entrained": fraction,
        })

    result = pd.DataFrame(rows)

    print()
    print("=" * 70)
    print(
        f"MASS FRACTION WITH u_s <= {air_velocity:.3f} m/s"
    )
    print("=" * 70)
    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


def plot_suspension_velocity_by_material(feed):
    """
    Plot the suspension-velocity distribution for each major material.
    """
    plt.figure(figsize=(9, 6))

    bins = np.linspace(
        feed["suspension_velocity_m_s"].min(),
        feed["suspension_velocity_m_s"].max(),
        60,
    )

    for material in feed["material"].unique():
        subset = feed[
            feed["material"] == material
        ]

        plt.hist(
            subset["suspension_velocity_m_s"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=material,
        )

    plt.xlabel("Suspension velocity, $u_s$ (m/s)")
    plt.ylabel("Probability density")
    plt.title("Suspension-Velocity Distribution by Material")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_foil_velocity_by_liberation_state(feed):
    """
    Compare bare, partially coated, and coated Al/Cu fragments.
    """
    foil = feed[
        feed["material"].isin(["aluminum", "copper"])
    ].copy()

    plt.figure(figsize=(10, 6))

    labels = []
    data = []

    for material in ["aluminum", "copper"]:
        for state in ["bare", "partial", "coated"]:
            subset = foil[
                (foil["material"] == material)
                & (foil["liberation_state"] == state)
            ]

            if len(subset) > 0:
                labels.append(f"{material}\n{state}")
                data.append(
                    subset["suspension_velocity_m_s"].to_numpy()
                )

    plt.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
    )

    plt.ylabel("Suspension velocity, $u_s$ (m/s)")
    plt.title(
        "Foil Suspension Velocity by Material and Liberation State"
    )
    plt.tight_layout()
    plt.show()


def plot_rhoH_vs_suspension_velocity(feed):
    """
    Directly visualize how areal density rho*H relates to suspension
    velocity for foil fragments.
    """
    foil = feed[
        feed["material"].isin(["aluminum", "copper"])
    ].copy()

    plt.figure(figsize=(9, 6))

    for material in ["aluminum", "copper"]:
        subset = foil[
            foil["material"] == material
        ]

        plt.scatter(
            subset["areal_density_kg_m2"],
            subset["suspension_velocity_m_s"],
            s=8,
            alpha=0.25,
            label=material,
        )

    plt.xlabel(
        r"Areal density, $\rho H$ (kg/m$^2$)"
    )
    plt.ylabel(
        r"Eq. 17 suspension velocity, $u_s$ (m/s)"
    )
    plt.title(
        r"Foil Areal Density ($\rho H$) vs Suspension Velocity"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eq9_vs_eq17(feed):
    """
    Compare the simple rho*H Eq. 9 prediction with the Eq. 17
    equivalent-diameter prediction for the same foil particles.
    """
    foil = feed[
        feed["material"].isin(["aluminum", "copper"])
    ].dropna(
        subset=[
            "eq9_velocity_m_s",
            "suspension_velocity_m_s",
        ]
    )

    plt.figure(figsize=(8, 6))

    for material in ["aluminum", "copper"]:
        subset = foil[
            foil["material"] == material
        ]

        plt.scatter(
            subset["eq9_velocity_m_s"],
            subset["suspension_velocity_m_s"],
            s=8,
            alpha=0.25,
            label=material,
        )

    lower = min(
        foil["eq9_velocity_m_s"].min(),
        foil["suspension_velocity_m_s"].min(),
    )

    upper = max(
        foil["eq9_velocity_m_s"].max(),
        foil["suspension_velocity_m_s"].max(),
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Eq. 9 = Eq. 17",
    )

    plt.xlabel("Eq. 9 velocity from $\\rho H$ (m/s)")
    plt.ylabel("Eq. 17 suspension velocity (m/s)")
    plt.title("Eq. 9 vs Eq. 17 for Foil Fragments")
    plt.legend()
    plt.tight_layout()
    plt.show()
