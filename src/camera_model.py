
"""Synthetic camera model for closed-loop separator studies.

This module intentionally separates TRUE simulated particle state from what
the controller is allowed to observe.

The camera sees:
    - an imperfect material classification
    - an imperfect projected particle-size measurement

It does NOT see:
    - true density
    - true coating fraction
    - true suspension velocity

Those remain latent plant states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CAMERA_CLASSES = [
    "aluminum",
    "copper",
    "separator",
    "black_mass",
    "other_light",
]


DEFAULT_CONFUSION = {
    "aluminum": {
        "aluminum": 0.92,
        "copper": 0.05,
        "separator": 0.01,
        "black_mass": 0.00,
        "other_light": 0.02,
    },
    "copper": {
        "aluminum": 0.05,
        "copper": 0.92,
        "separator": 0.01,
        "black_mass": 0.00,
        "other_light": 0.02,
    },
    "separator": {
        "aluminum": 0.02,
        "copper": 0.01,
        "separator": 0.92,
        "black_mass": 0.01,
        "other_light": 0.04,
    },
    "black_mass": {
        "aluminum": 0.00,
        "copper": 0.00,
        "separator": 0.02,
        "black_mass": 0.90,
        "other_light": 0.08,
    },
    "other_light": {
        "aluminum": 0.02,
        "copper": 0.01,
        "separator": 0.05,
        "black_mass": 0.07,
        "other_light": 0.85,
    },
}


class SyntheticCamera:
    def __init__(
        self,
        confusion=None,
        relative_size_sd=0.04,
        seed=2026,
    ):
        self.confusion = confusion or DEFAULT_CONFUSION
        self.relative_size_sd = float(relative_size_sd)
        self.rng = np.random.default_rng(seed)

    def _classify(self, material):
        probs = self.confusion.get(material)
        if probs is None:
            return material
        labels = list(probs)
        p = np.array([probs[k] for k in labels], dtype=float)
        p = p / p.sum()
        return self.rng.choice(labels, p=p)

    def observe_particles(self, particles, max_particles=None):
        """
        Return a camera-observed particle table.

        The true material identity is deliberately removed from the returned
        table so downstream control code cannot accidentally use it.
        """
        if particles.empty:
            return pd.DataFrame(
                columns=["camera_material", "camera_size_mm"]
            )

        observed = particles
        if max_particles is not None and len(observed) > max_particles:
            idx = self.rng.choice(
                len(observed),
                size=int(max_particles),
                replace=False,
            )
            observed = observed.iloc[idx]

        labels = [
            self._classify(m)
            for m in observed["material"].astype(str)
        ]

        true_size = observed["size_mm"].to_numpy(dtype=float)
        noise = self.rng.normal(
            0.0,
            self.relative_size_sd,
            len(observed),
        )
        measured_size = np.maximum(
            0.01,
            true_size * (1.0 + noise),
        )

        return pd.DataFrame(
            {
                "camera_material": labels,
                "camera_size_mm": measured_size,
            }
        )

    def summarize(self, observation):
        if observation.empty:
            return {
                "n_seen": 0,
                "aluminum_fraction": 0.0,
                "copper_fraction": 0.0,
                "mean_size_mm": np.nan,
            }

        counts = observation["camera_material"].value_counts(
            normalize=True
        )

        return {
            "n_seen": int(len(observation)),
            "aluminum_fraction": float(counts.get("aluminum", 0.0)),
            "copper_fraction": float(counts.get("copper", 0.0)),
            "mean_size_mm": float(
                observation["camera_size_mm"].mean()
            ),
        }
