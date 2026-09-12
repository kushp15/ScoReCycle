
# Hackathon optimization proof of concept

This version is designed to communicate the idea to judges rather than merely
show a generic ML controller.

## Process story

### Stage 2 — make an Al-rich top product

The Stage-2 controller:

1. observes the incoming feed with a camera;
2. starts at the Zhu size-class reference flow;
3. changes **high-pulse volumetric flow only in small steps**;
4. watches both the top and bottom outlet cameras;
5. increases flow when Al still appears in the bottom and the Al top product
   remains clean;
6. backs off strongly when Cu appears in the top product.

The goal is to improve **Al amount and Al purity in the top/light product
without sacrificing Cu just to gain more Al**.

### Stage 3 — clean the Stage-2 heavy product

The actual Stage-2 heavy stream from each branch becomes that branch's Stage-3
feed.

The Stage-3 controller:

1. uses a feed camera on the Stage-2 heavy stream;
2. watches both Stage-3 outlets;
3. gives special importance to the bottom camera;
4. slowly increases flow when the bottom Cu product still contains Al and the
   learned model predicts a useful purity gain;
5. allows more Cu to leave through the top than Stage 2 does;
6. backs off when Cu loss becomes excessive.

The Stage-3 goal is **Cu purity in the bottom product**.

A possible future Stage 4 could reprocess the Stage-3 top stream, but it is
intentionally not included in this hackathon demo.

## What is machine learning doing?

The ML model is a fast surrogate trained from the existing dynamic separator
simulations. It predicts outlet probabilities for a small candidate change in
airflow.

The ML model **does not replace the separator physics** and it cannot see true
density, coating fraction, or true suspension velocity.

The cameras provide imperfect estimates of:
- material class;
- particle size.

The feedback controller combines:
- ML prediction;
- feed-camera information;
- top-camera product quality;
- bottom-camera product quality.

## What changes in the animation?

For each of the three Zhu size classes, the animation contains:

1. fixed/open-loop separator particles;
2. closed-loop ML separator particles;
3. pulsated airflow comparison;
4. outlet accumulation comparison;
5. the product-purity objective.

During Stage 2 the quality plot is **Al purity in the top product**.

During Stage 3 the quality plot switches to **Cu purity in the bottom product**.

The dotted vertical line is the Stage-2 -> Stage-3 handoff.

## Run it

Fast hackathon version:

```bash
python3 run_hackathon_optimization_demo.py --quick
```

Save a GIF:

```bash
python3 run_hackathon_optimization_demo.py \
    --quick \
    --save-gif \
    --no-show
```

For a lighter laptop demo:

```bash
python3 run_hackathon_optimization_demo.py \
    --quick \
    --particles 70 \
    --stage2-duration 6 \
    --stage3-duration 6 \
    --fps 8
```

Outputs go to:

```text
outputs_hackathon_demo/
```

## Important scientific caveat

This is a digital-twin proof of concept. Camera accuracy, controller thresholds,
airflow step size, and Stage-3 loss tolerance are project assumptions until
they are measured on hardware.

The surrogate's held-out accuracy means it approximates the deterministic
simulation. It is not a claim of real-world ML accuracy.


## Important heavy-product interpretation

In the dynamic model, dense Cu-rich particles often remain suspended/held
around the variable-diameter region rather than physically reaching the
bottom boundary during the finite pulse window.

For this proof of concept, when a stage ends the airflow is considered shut
off and all material that **did not leave through the top** is discharged as
the heavy/bottom product.

Therefore:

```text
Stage heavy product = retained active material + any bottom-exited material
```

The bottom camera is modeled as looking at that retained heavy-side population
during operation. This is especially important for Stage 3, because its
control objective is to improve the Cu purity of the material that will remain
and be discharged from the bottom when the stage ends.


## Stage-3 starting flow in this demo

Stage 3 currently starts at the same size-class reference flow used for the
Stage-2 primary metal cut (75, 84, or 89 m3/h depending on size class).

This is a **project/demo starting rule**, not a Zhu Stage-3 literature optimum.

The earlier project-only 60/65/70 m3/h cleaning settings were too weak in the
time-domain model to move the retained material out of the variable-diameter
region. The next development step is to replace this fixed Stage-3 starting
rule with a prediction based on the Stage-2 bottom-camera composition and
particle-size state.
