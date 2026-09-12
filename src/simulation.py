import numpy as np
import pandas as pd

from config import (
    AIR_VELOCITIES,
    D_STRAIGHT,
    LIGHT_STAGE1_MATERIALS,
    METAL_MATERIALS,
)
from physics import flow_rate_from_velocity


def separate(feed, air_velocity):
    result = feed.copy()

    result["stream"] = np.where(
        air_velocity >= result["suspension_velocity_m_s"],
        "light",
        "heavy",
    )

    return result


def material_mass(feed, material):
    return feed.loc[
        feed["material"] == material,
        "statistical_mass",
    ].sum()


def stream_mass(feed, stream):
    return feed.loc[
        feed["stream"] == stream,
        "statistical_mass",
    ].sum()


def group_mass(feed, materials):
    return feed.loc[
        feed["material"].isin(materials),
        "statistical_mass",
    ].sum()


def group_in_stream_mass(feed, materials, stream):
    mask = (
        feed["material"].isin(materials)
        & (feed["stream"] == stream)
    )

    return feed.loc[
        mask,
        "statistical_mass",
    ].sum()


def material_in_stream_mass(
    feed,
    material,
    stream,
):
    mask = (
        (feed["material"] == material)
        & (feed["stream"] == stream)
    )

    return feed.loc[
        mask,
        "statistical_mass",
    ].sum()


def recovery(feed, material, stream):
    denominator = material_mass(feed, material)

    if denominator <= 0:
        return 0.0

    return (
        material_in_stream_mass(
            feed,
            material,
            stream,
        )
        / denominator
    )


def purity(feed, material, stream):
    denominator = stream_mass(feed, stream)

    if denominator <= 0:
        return 0.0

    return (
        material_in_stream_mass(
            feed,
            material,
            stream,
        )
        / denominator
    )


def group_recovery(
    feed,
    materials,
    stream,
):
    denominator = group_mass(feed, materials)

    if denominator <= 0:
        return 0.0

    return (
        group_in_stream_mass(
            feed,
            materials,
            stream,
        )
        / denominator
    )


def group_purity(
    feed,
    materials,
    stream,
):
    denominator = stream_mass(feed, stream)

    if denominator <= 0:
        return 0.0

    return (
        group_in_stream_mass(
            feed,
            materials,
            stream,
        )
        / denominator
    )


def stage1_metrics(separated):
    """
    Stage 1 follows the physical purpose of the original process:

        light nonmetals -> LIGHT
        Cu + Al          -> HEAVY

    The previous model incorrectly treated only black mass as the
    desired light product.

    The score balances:
        light recovery
        light-stream purity
        metal retention in the heavy stream
    """
    light_recovery = group_recovery(
        separated,
        LIGHT_STAGE1_MATERIALS,
        "light",
    )

    light_purity = group_purity(
        separated,
        LIGHT_STAGE1_MATERIALS,
        "light",
    )

    metal_retention = group_recovery(
        separated,
        METAL_MATERIALS,
        "heavy",
    )

    score = (
        light_recovery
        * light_purity
        * metal_retention
    )

    return {
        "target_recovery": light_recovery,
        "target_purity": light_purity,
        "metal_retention": metal_retention,
        "metal_loss": 1.0 - metal_retention,
        "score": score,
    }


def stage2_metrics(separated):
    """
    Stage 2 target:
        Al -> light
        Cu -> heavy
    """
    al_recovery = recovery(
        separated,
        "aluminum",
        "light",
    )

    al_purity = purity(
        separated,
        "aluminum",
        "light",
    )

    cu_retention = recovery(
        separated,
        "copper",
        "heavy",
    )

    score = (
        al_recovery
        * al_purity
        * cu_retention
    )

    return {
        "target_recovery": al_recovery,
        "target_purity": al_purity,
        "cu_retention": cu_retention,
        "score": score,
    }


def sweep_stage(feed, stage_number):
    records = []

    for velocity in AIR_VELOCITIES:
        separated = separate(
            feed,
            velocity,
        )

        if stage_number == 1:
            metrics = stage1_metrics(
                separated
            )
        elif stage_number == 2:
            metrics = stage2_metrics(
                separated
            )
        else:
            raise ValueError(
                "stage_number must be 1 or 2"
            )

        records.append({
            "air_velocity_m_s": velocity,
            "flow_100mm_m3_hr":
                flow_rate_from_velocity(
                    velocity,
                    D_STRAIGHT,
                ),
            **metrics,
        })

    return pd.DataFrame(records)


def best_operating_point(sweep):
    return sweep.loc[
        sweep["score"].idxmax()
    ]


def stream_composition_table(
    separated_feed
):
    if separated_feed.empty:
        return pd.DataFrame()

    table = (
        separated_feed
        .groupby(
            ["stream", "material"]
        )["statistical_mass"]
        .sum()
        .unstack(fill_value=0.0)
    )

    return table.div(
        table.sum(axis=1),
        axis=0,
    )
