# Dynamic Pulsated Pneumatic Separator

## What to run

Run the dynamic simulation with:

```bash
python3 run_dynamic.py --fraction 0.25-1.00
```

Other size fractions:

```bash
python3 run_dynamic.py --fraction 1.00-1.50
python3 run_dynamic.py --fraction 1.50-2.00
```

Example with a different pulse:

```bash
python3 run_dynamic.py \
  --fraction 0.25-1.00 \
  --velocity 2.8 \
  --frequency 5 \
  --duty 0.45
```

Save results:

```bash
python3 run_dynamic.py \
  --fraction 0.25-1.00 \
  --save-results
```

Headless / fast test:

```bash
python3 run_dynamic.py \
  --fraction 0.25-1.00 \
  --particles 200 \
  --duration 2 \
  --no-animation
```

## What is now dynamic

The model no longer immediately assigns a stream from the rule

`air velocity >= suspension velocity`.

Instead, each simulated parcel has a position and velocity:

- `x`, `z`
- `vx`, `vz`

and moves through time under:

- gravity
- buoyancy
- aerodynamic drag
- pulsated local airflow
- side-wall collisions

The gas velocity varies with separator diameter by continuity:

\[
u(z,t)=Q(t)/A(z)
\]

The variable-diameter separator therefore changes the local gas velocity.

## Drag model used in the dynamic simulation

Each parcel already has a suspension velocity calculated from the
literature-grounded particle model.

The dynamic drag term is calibrated so the time-domain model recovers that
same suspension velocity:

\[
a_D = k |u_r| u_r
\]

with

\[
k = g_\mathrm{eff}/u_s^2
\]

and

\[
g_\mathrm{eff}=g(1-\rho_g/\rho_p).
\]

This is intentional: it avoids replacing the existing Zhu / Bi /
Haider-Levenspiel suspension-velocity models with an unsupported new drag
law for thin flakes.

## Outlet convention

- Reaching the top of the column -> `light`
- Reaching the bottom of the column -> `heavy`
- Still inside when the run ends -> `active`

## Important limitations in Version 1

These are not hidden assumptions:

1. The feed-port height is currently a design assumption.
2. Particle-particle collisions are not yet modeled.
3. Wall friction/sticking/electrostatic effects are not yet modeled.
4. Mean gas flow is one-dimensional and vertical.
5. The valve-to-flow response is represented by a first-order lag.
6. Turbulent dispersion is currently off.
7. This is a 2-D particle trajectory model, not CFD/DEM.
8. Each size fraction is still run independently, matching the project
   modeling decision.

These limitations make this a physics-based reduced-order separator model,
not a replacement for CFD/DEM or experimental validation.


## Black-mass aerodynamic correction

The measured sieve size and the aerodynamic-equivalent diameter are now kept as
two different variables for liberated black mass.

`size_mm` remains the literature-derived comminution/sieve size.

`aerodynamic_size_mm` is the effective diameter used in the black-mass drag
calculation. Version 1 samples it from 0.04 to 0.10 mm. This is a transparent
calibration assumption, not a claimed direct measurement.

This correction is necessary because using a coarse sieve-class size together
with skeletal density treated black mass as a compact dense granule and caused
it to settle. In the corrected simulation, liberated black mass has a much
lower suspension velocity and is readily carried to the top/light outlet under
the Stage-1 airflow.


## Particle rendering update

The animation now uses only circles and squares:

- aluminum: square
- copper: square
- separator film: square
- black mass: circle
- unresolved nonmetal / `other_light`: circle

Marker area now follows each parcel's predicted measured `size_mm`. Because
matplotlib requires marker size in screen points rather than physical
millimetres, the physical distribution is mapped to a readable display range.
The relative size ordering still comes from the simulated particle-size
distribution.

## Why `other_light` previously fell to the bottom

The previous model treated `other_light` as a compact irregular particle with
density 1100 kg/m^3 and aerodynamic diameter equal to the measured sieve size.
That produced a high suspension velocity and made those parcels settle.

The revised model keeps the measured sieve size unchanged but uses a separate
aerodynamic-equivalent diameter of 0.08-0.20 mm for drag. This is an explicit
calibration assumption pending direct aerodynamic measurements of that mixed
fraction.


## Multi-stage Cu/Al separation

Run the new cascade with:

```bash
python3 run_multistage.py --fraction 0.25-1.00
```

For a fast non-animated check:

```bash
python3 run_multistage.py --fraction 0.25-1.00 --no-animation
```

The cascade uses three physical separation passes:

1. **Stage 1 — light-material removal**
   - default high-pulse velocity: 1.35 m/s
   - black mass, separator film, and other light material preferentially move
     toward the top/light product
   - Al/Cu continue as the heavy metal-rich stream

2. **Stage 2 — aluminum pre-separation**
   - default high-pulse velocity: 2.50 m/s
   - removes the lower-suspension-velocity Al population before the Cu cut
   - heavy + unresolved/middling particles continue to Stage 3

3. **Stage 3 — copper recovery**
   - default high-pulse velocity: 3.60 m/s
   - preferentially entrains Cu from the remaining Al/Cu population
   - high-suspension-velocity Al is retained

Why multiple metal stages are necessary:
the current particle model predicts a bimodal Al population. Some Al fragments
have lower suspension velocities than Cu, while other Al fragments have higher
suspension velocities than Cu. Therefore, a single velocity cut cannot cleanly
place all Al on one side and all Cu on the other.

The default velocities are model-driven operating points, not experimentally
validated optima. They are exposed as command-line arguments:

```bash
python3 run_multistage.py \
    --fraction 0.25-1.00 \
    --stage1-velocity 1.35 \
    --stage2-velocity 2.50 \
    --stage3-velocity 3.60
```
