# CAS-UQ: Closed-ended Medical VQA benchmark

This repository contains the code for the **7 models × 3 datasets = 21
closed-ended settings** in the shared-greedy-label HAC versus CAS benchmark.
It contains the MedVLM runtime, greedy answer and label generation, HAC/VASE
and PTrue baselines, closed CAS feature extraction, protocol audits, the main
comparison, and paper metric tables. It includes no datasets, model weights,
patient images, or historical prediction files.

## Protocol

Each question has one deterministic greedy answer. HAC and CAS use the same
answer and correctness label. CAS compares yes/no answer support under the
real image, text-only input, black image, noise image, and blurred image.
The extractor writes 17 CAS features; the **primary comparison uses nine
aggregate features** (`CAS-LogReg-cas_all`). HAC uses 10 high-temperature
samples only for uncertainty estimation, not correctness labels. See
[docs/PROTOCOL.md](docs/PROTOCOL.md), [docs/LABEL_POLICY.md](docs/LABEL_POLICY.md),
and [docs/METRICS.md](docs/METRICS.md).

## Install and inputs

Python 3.10+ and a compatible PyTorch installation are required. The bundled
`src/medvlm` package is based on the MIT-licensed MedVLM checkout; see
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hedge,internvl]'
export CAS_MODEL_ROOT=/path/to/models
export CAS_DATA_ROOT=/path/to/datasets
```

`CAS_MODEL_ROOT` contains the local model subdirectories named in
`src/medvlm/api.py`. Hugging Face model keys can use the model registry.
`CAS_DATA_ROOT` contains `Slake1.0` and
`vqa_med_2019/VQAMed2019Test`; VQA-RAD can be loaded through MedVLM from
Hugging Face. Model downloads and gated model access are the user's
responsibility.

## Reproduce the 21-setting main experiment

Run from this repository root. The runner processes seven models on
VQA-RAD, SLAKE, and VQA-Med-2019, **closed questions only**. It uses seed 42,
HAC N=10, temperature 0.7, cluster threshold 0.85, five stratified folds,
1,000 bootstrap draws for the main comparison, and 2,000 for paper tables.
It runs models sequentially on the selected device.

```bash
python scripts/run_main_experiment.py --dry-run
python scripts/run_main_experiment.py --cuda-visible-devices 0
```

The runner invokes `generate_greedy_answers.py`, `label_greedy_answers.py`,
`compute_hac_vase_from_greedy.py`, `compute_ptrue_from_greedy.py`,
`compute_cas_closed_from_greedy.py`, `audit_protocol_inputs.py`,
`evaluate_compare_hac_cas.py`, and `build_paper_metric_tables.py` in order.
Use `--evaluate-only` to rebuild results from existing JSONL inputs. It
checks that all 21 settings have greedy, label, HAC, PTrue, and CAS files.

Main outputs are `04_results/closed/json/summary.json` and
`04_results/paper_metrics/benchmark_main_metrics.csv`. The latter has
210 rows: 21 settings × 10 methods. Data and output directories are excluded
by `.gitignore`.

## Verification and limits

The evaluation code was run against the original cached inputs. All 21
closed settings matched the original sample counts, accuracy, method AUROCs,
and primary CAS deltas. The paper metrics builder matched the original
closed-setting metric rows. In those caches, CAS beat the HAC-family best in
21/21 settings, with mean AUROC difference +0.1434. This verifies evaluation
from caches, not a fresh GPU inference run.

Ten model/sample targets in six original files had greedy answers other than
yes/no, so closed CAS recorded errors for them. The comparison uses the
matched successful HAC/CAS samples. The original inputs and weights are not
part of this repository.
