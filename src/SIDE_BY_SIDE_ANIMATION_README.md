
# Full fixed-vs-closed-loop ML comparison animation

This version places **four synchronized panels in every particle-size row**:

| Column | Display |
|---|---|
| 1 | Fixed-control separator particle trajectories |
| 2 | Camera-informed closed-loop ML separator particle trajectories |
| 3 | Actual pulsated reference airflow velocity: fixed vs ML |
| 4 | Cumulative outlet accumulation: fixed vs ML |

The three rows remain:

- 0.25–1.00 mm
- 1.00–1.50 mm
- 1.50–2.00 mm

## Pulsated-airflow graph

The airflow graph does **not** simply plot the requested high-flow setting.
It plots the simulated, first-order-filtered reference-section gas velocity:

```text
actual Q(t) -> actual u_ref(t)
```

Therefore you can see:

- the 4 Hz pulsation;
- the fixed-control pulse amplitude/duty;
- changes in ML-selected high airflow;
- changes in ML-selected duty cycle;
- the gas-response transient.

Vertical dotted lines mark the boundaries between camera/control windows.

## Outlet-accumulation graph

The final graph records cumulative parcels reaching each physical outlet over
the entire changing-feed experiment:

- Fixed top/light
- Fixed bottom/heavy
- ML top/light
- ML bottom/heavy

Counts do not reset when a new feed/control window begins. This makes it easy
to compare how rapidly the two control strategies accumulate material at the
top and bottom exits over time.

## Fair comparison

For each row and each control window:

1. both columns receive the same true Cu/Al particles;
2. both use the same initial-position and initial-velocity random seed;
3. both use the same separator physics;
4. the fixed side remains at its reference setting;
5. only the ML side receives the noisy feed-camera estimate and outlet-camera
   feedback and can change the control setting.

## Run interactively

```bash
python3 run_side_by_side_ml_animation.py --quick
```

## Save an animated GIF

```bash
python3 run_side_by_side_ml_animation.py \
    --quick \
    --save-gif \
    --no-show
```

For a faster demonstration:

```bash
python3 run_side_by_side_ml_animation.py \
    --quick \
    --windows 3 \
    --particles-per-window 70 \
    --camera-sample 55 \
    --window-duration 3.0
```

Outputs are saved under:

```text
outputs_side_by_side/
```
