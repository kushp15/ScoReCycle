# Literature basis for the revised feed generator

The simulation now treats **comminution size fraction as the first conditioning variable**.

It does **not** generate one global normal particle-size distribution and then classify those
particles afterward.

## Pneumatic fractions

The original pulsated pneumatic separator uses:
- 0.25–1.0 mm
- 1.0–1.5 mm
- 1.5–2.0 mm

Zhang et al. (2014) measured the following post-comminution yields:
- 0.25–0.5 mm = 2.54 wt%
- 0.5–1.0 mm = 5.81 wt%
- 1–2 mm = 7.86 wt%

Inside the 0.25–2 mm pneumatic feed this becomes:
- 0.25–1.0 mm = 51.51 mass%
- 1–2 mm = 48.49 mass%

The 1–2 mm source bin is divided equally between 1–1.5 and 1.5–2 mm because the
source does not resolve that interval further. This is interpolation, not measured data.

## Conditional material composition

For 0.5–1 and 1–2 mm, the mobile-phone LIB shredding study reporting
"Component content in the size fractions after shredding" supplies direct phase mass fractions.

For 0.25–0.5 mm, Zhang et al. (2014) supplies direct Al and Cu contents:
- Cu = 50.20 wt%
- Al = 10.80 wt%

The remainder is nonmetallic/active-material rich. The separator share is assigned using
the nearest measured 0.5–1 mm separator fraction (9.68 percentage points), leaving
29.32% as active-material-rich free black mass. This is a transparent cross-source
interpolation and should be replaced if a better direct phase table for 0.25–0.5 mm is found.

## Liberation

Vanderbruggen et al. automated-mineralogy data provide mass-based liberation degrees:
- 125–500 µm: Al 16.4%, Cu 57.9%
- 500–1000 µm: Al 17.2%, Cu 58.2%

The simulation uses those as P(bare foil | material, size). Non-liberated material is
represented by the measured "metal + electrode" composite endpoints from Bi et al. (2019).

The earlier unsupported 40/40/20 bare/partial/coated probability model has been removed.

## Physical properties

Bi et al. (2019):
- Cu bare: 0.0120 mm, 8900 kg/m3
- Al bare: 0.0218 mm, 2700 kg/m3
- Cu + electrode: 0.1580 mm, 1600 kg/m3
- Al + electrode: 0.1790 mm, 2130 kg/m3
- separator: 0.0139 mm, 900 kg/m3

## Important limitation

Published sieve fractions and component percentages are mass-based. The Monte Carlo rows
are therefore **equal-mass simulation parcels**, not literal equal-count physical particles.

This prevents a common error: treating a 50 mass% fraction as if it meant 50% of the actual
number of particles.
