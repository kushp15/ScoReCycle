# Realistic three-stage pneumatic-separation model

This folder returns to the original literature-calibrated particle model.
It does **not** force copper and aluminum into perfect products.

## Exact Zhu 2021 screened size classes

The simulation uses the particle-size distribution shown in Zhu et al. 2021,
Fig. 3(a):

| sieve class | reported mass frequency |
|---|---:|
| >2.0 mm | 4.18% |
| 1.5–2.0 mm | 18.23% |
| 1.0–1.5 mm | 19.11% |
| 0.5–1.0 mm | 7.47% |
| 0.25–0.5 mm | 0.68% |
| <0.25 mm | 50.33% |

The pneumatic model independently simulates:

1. **0.25–1.00 mm** (the paper combines 0.25–0.5 and 0.5–1.0 mm)
2. **1.00–1.50 mm**
3. **1.50–2.00 mm**

The sieve boundaries and class frequencies are direct literature inputs.
The paper does not provide a continuous probability density inside each sieve
interval, so sizes are sampled uniformly *inside the exact reported limits*.
That is an interpolation, not invented measured data.

## Particle physics retained from the original model

The original theory is retained:

- Cu and Al are modeled as flaky particles.
- Zhu's equivalent-diameter suspension-velocity model is used.
- Bare and electrode-bound/composite foil states remain in the model.
- The size-dependent liberation probabilities already used in the original
  project are retained.
- Black mass and unresolved light material retain their separate aerodynamic
  treatment.
- Dynamic drag is calibrated to each parcel's calculated suspension velocity.
- No particle is routed by its label. A copper parcel can report to the light
  stream and an aluminum parcel can report to the heavy stream.

This is important: **small Cu fragments and large/heavily coated Al fragments
can overlap aerodynamically.** The model therefore produces cross-contamination
instead of perfect Cu/Al separation.

## Three stages for every size class

### Stage 1 — nonmetal/light removal
The original project setting of 1.35 m/s high-pulse reference velocity is
retained. This is a model calibration/design setting, not a Zhu measured value.

### Stage 2 — primary Al/Cu cut
Uses the paper's size-specific air volumes:

- 0.25–1.00 mm: **75 m³/h**
- 1.00–1.50 mm: **84 m³/h**
- 1.50–2.00 mm: **89 m³/h**

### Stage 3 — resolve Stage-2 middlings

Stage 3 now reprocesses only the particles that remain suspended/unresolved
after Stage 2. It does **not** replace or artificially clean the Stage-2 product
streams.

Particles that resolve upward are added to the Al-intended light product.
Particles that resolve downward are added to the Cu-intended heavy product.
Particles still active remain explicit middlings/recycle.

To avoid inventing a new airflow value, Stage 3 uses lower flow values from
the airflow grid tested in Zhu Fig. 4:

- 0.25–1.00 mm: **70 m³/h**
- 1.00–1.50 mm: **80 m³/h**
- 1.50–2.00 mm: **80 m³/h**

Choosing the lower tested point for the middlings pass is a project design
choice, not a claim that Zhu experimentally optimized a third stage at those
values.

## Run all three size classes

```bash
python3 run_multistage.py
```

or:

```bash
python3 run_multistage.py --fraction all
```

This runs three independent three-stage simulations. Every animation title
states the exact particle-size class.

Run one size class:

```bash
python3 run_multistage.py --fraction 0.25-1.00
python3 run_multistage.py --fraction 1.00-1.50
python3 run_multistage.py --fraction 1.50-2.00
```

Headless validation:

```bash
python3 run_multistage.py --fraction all --no-animation
```

Save particle-level results:

```bash
python3 run_multistage.py --fraction all --no-animation --save-results
```

## Interpretation

Do not interpret a stream name such as "Aluminum-rich" or "Copper-rich" as a
pure material. Those names describe the intended product stream. The printed
composition and recovery/purity values are the model results.

`active` particles are unresolved middlings. They are never silently counted
as clean product.

The model is intended to show physically believable trends and impurity
mechanisms, not to reproduce Zhu's recovery numbers by force-fitting.


## Partial-coating / overlap revision

The Cu/Al particle model is no longer binary "bare" versus "fully coated."

The original measured bare and fully electrode-bound endpoints are retained,
as is the same equivalent-diameter suspension-velocity theory. Non-bare
particles now receive a continuous residual-coating fraction between those
measured endpoints.

Because the current literature set does not provide a measured probability
distribution for residual coating thickness, the code uses a transparent
triangular interpolation over 0–100% of the measured endpoint difference,
with its mode at 35%. This is explicitly a modeling interpolation, not a claim
of measured coating fractions.

The purpose is to represent the physically plausible continuum:

- small/light Al: easiest to entrain;
- large or partially coated Al: higher suspension threshold;
- small Cu: can overlap with large/partially coated Al;
- large/heavier Cu: generally occupies the higher suspension-velocity side.

Stage 3 now receives the aerodynamic overlap region, not a stream selected by
material identity. It includes all unresolved Stage-2 particles and particles
from either outlet whose predicted suspension velocity lies near the Stage-2
cut. This allows both Cu and Al to appear in the final recycle/separation pass.


## Final calibration revision

### Why the aluminum population was previously settling too quickly

The preceding version allowed too much of the fully electrode-bound cathode
endpoint to remain on aluminum after comminution. Because the cathode coating
adds substantial areal mass, the high-coating Al tail could exceed the Cu
population and dominate the heavy product.

The underlying suspension theory was not changed. The correction is to the
post-comminution feed-state model.

### Residual coating

Measured bare and fully coated Bi endpoints remain the physical endpoints.
For the Zhu-style post-comminution feed, residual coating is now sampled over
a calibrated fraction of that full endpoint:

- Al: 0–45% of the full measured coating increment
- Cu: 0–80% of the full measured coating increment

These percentages are **calibration envelopes, not direct measured coating
fractions**. They are deliberately exposed in `config.py`. They were selected
to satisfy the experimentally observed population ordering while preserving
overlap:

- Cu population-average suspension velocity > Al population-average
  suspension velocity;
- the fastest Cu remains faster than the fastest Al;
- large/partially coated Al overlaps small Cu;
- neither material is routed by its label.

If direct coating-retention measurements become available, these calibration
values should be replaced.

### Third-stage airflow

The third stage is intentionally slower than the primary metal cut:

| Size class | Stage 2 | Stage 3 |
|---|---:|---:|
| 0.25–1.00 mm | 75 m3/h | 60 m3/h |
| 1.00–1.50 mm | 84 m3/h | 65 m3/h |
| 1.50–2.00 mm | 89 m3/h | 70 m3/h |

The Stage-2 values are from Zhu 2021. The Stage-3 values are **our process
design settings**, chosen lower so borderline Cu is more likely to settle while
some large Al also settles. They are not claimed as literature optima.

The default low/high pulse ratio was also changed from 0.25 to 0.35. This is a
model calibration parameter, not a measured Zhu value. It prevents the low
part of the pulse from unrealistically dominating the trajectory and causing
too many Al fragments to fall.

### `other_light`

`other_light` is an internal category for unresolved light nonmetal/plastic
material from the source composition data. It is not a single chemical
species. In printed output it is described as an
"unresolved light nonmetal/plastic proxy."
