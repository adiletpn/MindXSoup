# Startup timing observations ? 2026-10-30

Source: memory diagnostic version 1 on Kaggle T4. These are cold diagnostic process timings, not a decomposition of the official Soup training subprocess. Imports are sequential; later imports may already be cached by earlier imports. Model loading includes first-download and CUDA initialization costs.

| Stage | Seconds |
|---|---:|
| import torch | 5.057 |
| import transformers | 9.724 |
| import trl | 2.237 |
| import peft | 13.557 |
| import bitsandbytes | 0.000 |
| import soup_cli | 0.000 |
| load_model | 30.172 |
| prepare_and_lora | 0.139 |

Do not sum these stages and compare directly with the organizer estimate of ~20 seconds. The official harness warms downloads first. Official wall-minus-train-loop observations below include imports, model loading, setup, compilation and saving.

## Official warmed benchmark, seed 1

| Run | Wall s | Loop s | Outside loop s | apply s (included, not additional) |
|---|---:|---:|---:|---:|
| base_1 | 85.02 | 57.24 | 27.78 | None |
| base_2 | 85.49 | 59.08 | 26.41 | None |
| solution_1 | 73.53 | 47.16 | 26.37 | 9.03 |
| solution_2 | 73.08 | 46.56 | 26.52 | 8.93 |

The residual is about 26-28 s in these runs. It is not all import time; the available logs do not provide timestamps sufficient to attribute every second. Solution apply() takes about 9 s, some of which may be imports that the base performs elsewhere. No claimed optimization is inferred from subtraction alone.
