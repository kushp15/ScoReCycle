# Pneumatic Separator Simulation

This folder contains the current battery-recycling pneumatic-separation model.

## Files

- `config.py`  
  All assumptions and adjustable parameters.

- `physics.py`  
  Suspension-velocity equations, drag model, flow conversions.

- `feed_model.py`  
  Generates hammer-milled synthetic particles, including variable foil
  thickness, residual electrode coating, effective density, and areal density.

- `simulation.py`  
  Performs separation and air-velocity sweeps.

- `main.py`  
  Runs the complete simulation.

- `outputs/`  
  Optional output folder.

## Run

Open this folder in VS Code, then in the terminal:

```bash
python3 main.py
```

## CSV behavior

By default:

```python
SAVE_OUTPUTS = False
```

so **no CSV files are written**.

If you change it to:

```python
SAVE_OUTPUTS = True
```

the model writes only:

- `outputs/simulation_summary.csv`
- `outputs/latest_particle_results.csv`

Those same two files are overwritten each run rather than creating a growing
set of duplicate files.

## Hammer-mill model

The current first-pass split is:

- `< 0.25 mm` = powdered fraction
- `>= 0.25 mm` = shredded fraction

The powdered fraction bypasses the pneumatic foil separator in this version.
The shredded fraction enters the pneumatic separation stages.

## Foil density/thickness model

Foils are no longer represented by density alone.

Each Al or Cu fragment has:

- metal thickness
- residual coating thickness
- total thickness
- coating/liberation state
- areal density
- effective composite density
- suspension velocity

For a composite foil:

```text
areal density = rho_metal * H_metal + rho_coating * H_coating
```

and:

```text
effective density = areal density / total thickness
```

The Eq. 17 suspension model uses the effective density and total thickness.

The simpler Eq. 9 value is also calculated for foil fragments as a diagnostic.

## Important

The liberation probabilities, size distributions, and many spread parameters
are still assumptions. They should later be replaced with experimental
hammer-mill data or literature distributions.

## Suspension diagnostics

`diagnostics.py` contains optional functions for checking whether the assumed
particle distributions produce physically sensible separation windows.

Useful imports in `main.py`:

```python
from diagnostics import (
    print_suspension_diagnostics,
    print_fraction_below_velocity,
    plot_suspension_velocity_by_material,
    plot_foil_velocity_by_liberation_state,
    plot_rhoH_vs_suspension_velocity,
    plot_eq9_vs_eq17,
)
```

After `shredded` has been created, call:

```python
diagnostic_table = print_suspension_diagnostics(shredded)

plot_suspension_velocity_by_material(shredded)
plot_foil_velocity_by_liberation_state(shredded)
plot_rhoH_vs_suspension_velocity(shredded)
plot_eq9_vs_eq17(shredded)
```

After Stage 1 selects `stage1_velocity`, call:

```python
print_fraction_below_velocity(
    shredded,
    stage1_velocity,
)
```

This reveals exactly which materials and foil liberation states are being
entrained at the selected air velocity.

