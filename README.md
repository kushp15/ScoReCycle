# ScoReCycle

Adaptive, camera-in-the-loop pneumatic separation for recovering aluminum and
copper foil from shredded lithium-ion battery scrap. Built by the **Teenage
CMUtant Ninja Turtles** for HackCMU 2026.

Particles suspended in an air column each have a balance ("suspension")
velocity. Since separation depends on density **and** thickness together, not
density alone, different materials can end up sharing the same suspension
velocity and get stuck together. ScoReCycle watches the separator with
cameras and nudges the airflow in real time to move stuck material to a
velocity where it separates cleanly, instead of trying to eliminate the
overlap outright.

## See the demo / slide deck

The pitch deck (`scorecycle-deck.html`) is published with GitHub Pages and
rebuilds automatically on every push to `main`:

**Live deck:** https://kushp15.github.io/ScoReCycle/scorecycle-deck.html

To view it locally instead, just open the file directly in a browser:

```bash
open scorecycle-deck.html   # macOS
```

## Repository layout

```
scorecycle-deck.html   Slide deck / demo, served via GitHub Pages
Sources.docx           Literature references cited in the deck
src/                    Pneumatic separator simulation and ML controller
```

## Simulation code (`src/`)

`src/` contains the physics simulation and closed-loop ML controller behind
the deck's results. Start with the docs below, in this order:

| File | What it covers |
|---|---|
| [`src/README.md`](src/README.md) | Base pneumatic separator model: config, physics, feed generation, how to run it |
| [`src/DYNAMIC_SIMULATION_README.md`](src/DYNAMIC_SIMULATION_README.md) | Time-domain / variable-diameter separator simulation |
| [`src/CLOSED_LOOP_ML_README.md`](src/CLOSED_LOOP_ML_README.md) | Closed-loop ML controller that adjusts airflow from camera feedback |
| [`src/HACKATHON_DEMO_README.md`](src/HACKATHON_DEMO_README.md) | The optimization proof-of-concept demo used for judging, and how to run it |
| [`src/SIDE_BY_SIDE_ANIMATION_README.md`](src/SIDE_BY_SIDE_ANIMATION_README.md) | Open-loop vs. closed-loop side-by-side animation |
| [`src/REALISTIC_MODEL_README.md`](src/REALISTIC_MODEL_README.md) | Notes on making the model more physically realistic |

Quick start for the base model:

```bash
cd src
python3 main.py
```

## Sources

All literature grounding the physics and material-composition assumptions is
collected in [`Sources.docx`](Sources.docx) and summarized in
[`src/literature_basis.md`](src/literature_basis.md) /
[`src/ZHU_2021_DIRECT_INPUTS.md`](src/ZHU_2021_DIRECT_INPUTS.md). Key references:

- Zhu, X.; Zhang, C.; Feng, P.; Yang, X.; Yang, X. *A Novel Pulsated
  Pneumatic Separation with Variable-Diameter Structure and Its Application
  in the Recycling Spent Lithium-Ion Batteries.* Waste Manage. 2021, 131,
  20–30. DOI: [10.1016/j.wasman.2021.05.027](https://doi.org/10.1016/j.wasman.2021.05.027)
- Bi, H.; Zhu, H.; Zu, L.; He, S.; Gao, Y.; Gao, S. *Pneumatic Separation and
  Recycling of Anode and Cathode Materials from Spent Lithium Iron Phosphate
  Batteries.* Waste Manage. Res. 2019, 37 (4), 374–385. DOI:
  [10.1177/0734242X18823939](https://doi.org/10.1177/0734242X18823939)
- Vanderbruggen, A.; Gugala, E.; Blannin, R.; Bachmann, K.; Serna-Guerrero,
  R.; Rudolph, M. *Automated Mineralogy as a Novel Approach for the
  Compositional and Textural Characterization of Spent Lithium-Ion
  Batteries.* Miner. Eng. 2021, 169, 106924. DOI:
  [10.1016/j.mineng.2021.106924](https://doi.org/10.1016/j.mineng.2021.106924)

## Important caveat

This is a digital-twin proof of concept. Camera accuracy, controller
thresholds, airflow step sizes, and several distribution parameters in the
simulation are project assumptions until they are measured on real hardware.
See the caveats sections in the individual `src/*_README.md` files for
model-specific limitations.
