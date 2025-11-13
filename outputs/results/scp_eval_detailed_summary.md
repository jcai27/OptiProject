# SCP Evaluation Summary by Class

Single-instance training on `SCP_500_1000_3`; evaluation uses 60s / 5k-node caps per instance. Results compare vanilla PySCIPOpt vs. learned policies.

## Overall metrics

- Instances: **39**
- Objective improved / worse / tied: **21 / 14 / 4**
- Avg objective (baseline → learned): **-625.90 → -634.23**
- Median objective (baseline → learned): **-646.00 → -646.00**
- Avg nodes (baseline → learned): **100.0 → 97.0**
- Median nodes (baseline → learned): **18.0 → 3.0**
- Avg solve time (baseline → learned): **57.44s → 56.96s**

## Class breakdown

### Class 1000_1000

- Instances: **10**
- Objective improved / worse / tied: **8 / 2 / 0**
- Avg objective (baseline → learned): **-563.70 → -591.20**
- Median objective (baseline → learned): **-554.00 → -609.50**
- Avg nodes (baseline → learned): **19.4 → 5.8**
- Median nodes (baseline → learned): **18.5 → 3.0**
- Avg solve time (baseline → learned): **57.93s → 56.86s**

| Instance | Status (Base/ML) | Objective (Base) | Objective (ML) | Δ Objective | Nodes (Base) | Nodes (ML) | Δ Nodes | Time (Base s) | Time (ML s) |
|---|---|---|---|---|---|---|---|---|---|
| SCP_1000_1000_1 | timelimit / timelimit | -591.0 | -604.0 | -13.0 | 25 | 3 | -22 | 59.15 | 56.37 |
| SCP_1000_1000_10 | timelimit / timelimit | -550.0 | -546.0 | 4.0 | 24 | 14 | -10 | 57.41 | 56.47 |
| SCP_1000_1000_2 | timelimit / timelimit | -541.0 | -606.0 | -65.0 | 15 | 3 | -12 | 57.91 | 58.16 |
| SCP_1000_1000_3 | timelimit / timelimit | -550.0 | -615.0 | -65.0 | 11 | 2 | -9 | 57.92 | 56.61 |
| SCP_1000_1000_4 | timelimit / timelimit | -561.0 | -614.0 | -53.0 | 19 | 2 | -17 | 57.81 | 56.63 |
| SCP_1000_1000_5 | timelimit / timelimit | -558.0 | -614.0 | -56.0 | 17 | 2 | -15 | 57.78 | 56.67 |
| SCP_1000_1000_6 | timelimit / timelimit | -565.0 | -613.0 | -48.0 | 31 | 3 | -28 | 58.84 | 56.54 |
| SCP_1000_1000_7 | timelimit / timelimit | -548.0 | -619.0 | -71.0 | 18 | 5 | -13 | 57.33 | 58.26 |
| SCP_1000_1000_8 | timelimit / timelimit | -635.0 | -532.0 | 103.0 | 6 | 11 | 5 | 57.77 | 56.58 |
| SCP_1000_1000_9 | timelimit / timelimit | -538.0 | -549.0 | -11.0 | 28 | 13 | -15 | 57.36 | 56.36 |

### Class 1000_2000

- Instances: **10**
- Objective improved / worse / tied: **5 / 3 / 2**
- Avg objective (baseline → learned): **-683.40 → -679.30**
- Median objective (baseline → learned): **-690.00 → -672.00**
- Avg nodes (baseline → learned): **1.0 → 1.0**
- Median nodes (baseline → learned): **1.0 → 1.0**
- Avg solve time (baseline → learned): **57.33s → 56.89s**

| Instance | Status (Base/ML) | Objective (Base) | Objective (ML) | Δ Objective | Nodes (Base) | Nodes (ML) | Δ Nodes | Time (Base s) | Time (ML s) |
|---|---|---|---|---|---|---|---|---|---|
| SCP_1000_2000_1 | timelimit / timelimit | -660.0 | -661.0 | -1.0 | 1 | 1 | 0 | 58.45 | 56.69 |
| SCP_1000_2000_10 | timelimit / timelimit | -663.0 | -663.0 | 0.0 | 1 | 1 | 0 | 56.82 | 56.22 |
| SCP_1000_2000_2 | timelimit / timelimit | -663.0 | -663.0 | 0.0 | 1 | 1 | 0 | 58.55 | 58.26 |
| SCP_1000_2000_3 | timelimit / timelimit | -699.0 | -702.0 | -3.0 | 1 | 1 | 0 | 57.07 | 56.50 |
| SCP_1000_2000_4 | timelimit / timelimit | -704.0 | -707.0 | -3.0 | 1 | 1 | 0 | 57.00 | 56.52 |
| SCP_1000_2000_5 | timelimit / timelimit | -712.0 | -646.0 | 66.0 | 1 | 1 | 0 | 56.78 | 56.52 |
| SCP_1000_2000_6 | timelimit / timelimit | -691.0 | -670.0 | 21.0 | 1 | 1 | 0 | 56.64 | 56.78 |
| SCP_1000_2000_7 | timelimit / timelimit | -653.0 | -698.0 | -45.0 | 1 | 1 | 0 | 58.50 | 58.21 |
| SCP_1000_2000_8 | timelimit / timelimit | -689.0 | -709.0 | -20.0 | 1 | 1 | 0 | 56.70 | 56.57 |
| SCP_1000_2000_9 | timelimit / timelimit | -700.0 | -674.0 | 26.0 | 1 | 1 | 0 | 56.82 | 56.59 |

### Class 500_1000

- Instances: **9**
- Objective improved / worse / tied: **3 / 5 / 1**
- Avg objective (baseline → learned): **-560.78 → -560.89**
- Median objective (baseline → learned): **-557.00 → -562.00**
- Avg nodes (baseline → learned): **384.2 → 404.0**
- Median nodes (baseline → learned): **367.0 → 400.0**
- Avg solve time (baseline → learned): **57.00s → 57.21s**

| Instance | Status (Base/ML) | Objective (Base) | Objective (ML) | Δ Objective | Nodes (Base) | Nodes (ML) | Δ Nodes | Time (Base s) | Time (ML s) |
|---|---|---|---|---|---|---|---|---|---|
| SCP_500_1000_1 | timelimit / timelimit | -545.0 | -562.0 | -17.0 | 360 | 330 | -30 | 58.40 | 56.70 |
| SCP_500_1000_10 | timelimit / timelimit | -581.0 | -566.0 | 15.0 | 332 | 424 | 92 | 56.20 | 56.81 |
| SCP_500_1000_2 | timelimit / timelimit | -531.0 | -576.0 | -45.0 | 467 | 409 | -58 | 56.79 | 57.40 |
| SCP_500_1000_4 | timelimit / timelimit | -573.0 | -572.0 | 1.0 | 476 | 424 | -52 | 56.76 | 57.13 |
| SCP_500_1000_5 | timelimit / timelimit | -556.0 | -556.0 | 0.0 | 346 | 283 | -63 | 56.71 | 57.09 |
| SCP_500_1000_6 | timelimit / timelimit | -582.0 | -551.0 | 31.0 | 441 | 616 | 175 | 58.39 | 58.54 |
| SCP_500_1000_7 | timelimit / timelimit | -545.0 | -531.0 | 14.0 | 283 | 380 | 97 | 56.80 | 57.05 |
| SCP_500_1000_8 | timelimit / timelimit | -577.0 | -574.0 | 3.0 | 367 | 370 | 3 | 56.56 | 57.03 |
| SCP_500_1000_9 | timelimit / timelimit | -557.0 | -560.0 | -3.0 | 386 | 400 | 14 | 56.42 | 57.17 |

### Class 500_2000

- Instances: **10**
- Objective improved / worse / tied: **5 / 4 / 1**
- Avg objective (baseline → learned): **-689.20 → -698.20**
- Median objective (baseline → learned): **-690.50 → -696.50**
- Avg nodes (baseline → learned): **23.9 → 7.8**
- Median nodes (baseline → learned): **17.5 → 3.0**
- Avg solve time (baseline → learned): **57.45s → 56.92s**

| Instance | Status (Base/ML) | Objective (Base) | Objective (ML) | Δ Objective | Nodes (Base) | Nodes (ML) | Δ Nodes | Time (Base s) | Time (ML s) |
|---|---|---|---|---|---|---|---|---|---|
| SCP_500_2000_1 | timelimit / timelimit | -693.0 | -704.0 | -11.0 | 55 | 16 | -39 | 58.20 | 58.50 |
| SCP_500_2000_10 | timelimit / timelimit | -656.0 | -690.0 | -34.0 | 1 | 1 | 0 | 58.08 | 56.36 |
| SCP_500_2000_2 | timelimit / timelimit | -706.0 | -700.0 | 6.0 | 19 | 1 | -18 | 56.39 | 56.99 |
| SCP_500_2000_3 | timelimit / timelimit | -715.0 | -715.0 | 0.0 | 1 | 18 | 17 | 56.57 | 56.93 |
| SCP_500_2000_4 | timelimit / timelimit | -685.0 | -672.0 | 13.0 | 43 | 16 | -27 | 56.57 | 56.77 |
| SCP_500_2000_5 | timelimit / timelimit | -667.0 | -682.0 | -15.0 | 16 | 18 | 2 | 58.31 | 56.65 |
| SCP_500_2000_6 | timelimit / timelimit | -688.0 | -733.0 | -45.0 | 68 | 5 | -63 | 56.73 | 56.24 |
| SCP_500_2000_7 | timelimit / timelimit | -646.0 | -682.0 | -36.0 | 20 | 1 | -19 | 57.02 | 58.11 |
| SCP_500_2000_8 | timelimit / timelimit | -709.0 | -693.0 | 16.0 | 15 | 1 | -14 | 59.07 | 56.20 |
| SCP_500_2000_9 | timelimit / timelimit | -727.0 | -711.0 | 16.0 | 1 | 1 | 0 | 57.58 | 56.42 |
