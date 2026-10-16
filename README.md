# CAS-UQ: Counterfactual Answer Support for Medical VQA

Code for the greedy-label HAC versus CAS benchmark. The repository contains the
closed-answer and open-answer CAS feature extractors, shared greedy target and
label generation, HAC/VASE comparison, evaluation, ablations, and the MedVLM
runtime used by these scripts. It contains **no datasets, model weights, patient
images, or historical prediction files**.

## Method and protocol

For each question, one deterministic greedy answer is the shared target for
CAS and HAC. CAS measures support for that fixed answer under the real image
and four counterfactual inputs: text only, black image, noise image, and blurred
image. The closed extractor scores yes/no candidates. The open extractor
teacher-forces the greedy answer tokens and checks token alignment before
computing likelihoods. HAC samples supply uncertainty signals only; they do
not supply correctness labels. See [docs/PROTOCOL.md](docs/PROTOCOL.md),
[docs/LABEL_POLICY.md](docs/LABEL_POLICY.md), and
[docs/METRICS.md](docs/METRICS.md).

## Installation

Python 3.10+ and a compatible PyTorch installation are required. The bundled
`src/medvlm` code is derived from the MIT-licensed MedVLM package; see
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hedge,internvl]'
```

The `hedge` extra is needed for HAC/VASE. Model downloads and access to gated
models are the user's responsibility. Set `CAS_MODEL_ROOT` to a directory
containing the local model subdirectories named in `src/medvlm/api.py`, or use
the Hugging Face model keys in that registry. Set `CAS_DATA_ROOT` for local
SLAKE and VQA-Med datasets; VQA-RAD can be loaded by MedVLM from Hugging Face.
The expected local layout is `CAS_DATA_ROOT/Slake1.0` and
`CAS_DATA_ROOT/vqa_med_2019/VQAMed2019Test` (analogous paths for 2020/2021).

## Minimal workflow

Run from the repository root. Replace the model and dataset keys with those
available in your environment. A small pilot uses `--max-samples 10`; omit it
for the full data. Each stage writes JSONL outputs into ignored directories.

```bash
export CAS_MODEL_ROOT=/path/to/models
export CAS_DATA_ROOT=/path/to/datasets
python scripts/generate_greedy_answers.py --models qwen3vl_2b --datasets vqa_rad --qtypes closed open --max-samples 10
python scripts/label_greedy_answers.py --models qwen3vl_2b --datasets vqa_rad --qtypes closed open
python scripts/compute_cas_closed_from_greedy.py --models qwen3vl_2b --datasets vqa_rad --max-samples 10
python scripts/compute_cas_open_from_greedy.py --models qwen3vl_2b --datasets vqa_rad --max-samples 10
python scripts/compute_hac_vase_from_greedy.py --models qwen3vl_2b --datasets vqa_rad --qtypes closed open --max-samples 10
python scripts/audit_protocol_inputs.py --model qwen3vl_2b --dataset vqa_rad --qtype closed
python scripts/audit_protocol_inputs.py --model qwen3vl_2b --dataset vqa_rad --qtype open --output 05_audits/protocol_audit_open.json
python scripts/evaluate_compare_hac_cas.py --qtype closed --cas-dir 03_runs/cas_closed/features --output-dir 04_results/closed
python scripts/evaluate_compare_hac_cas.py --qtype open --cas-dir 03_runs/cas_open/features --output-dir 04_results/open
```

Check each script's `--help` before running. The closed extractor does not
accept `--strict-token-alignment`; that flag applies to the open extractor and
is enabled by default. The automatic open-answer label is normalized exact
match, so task-specific human or external labels should be supplied with
`label_greedy_answers.py --external-labels` when that is the evaluation target.

## Full main experiment (24 closed / 21 open settings)

The latest main summary covers seven models on VQA-RAD, SLAKE, and
VQA-Med-2019 for both question types, plus LLaVA-NeXT-7B on the three closed
tasks. `run_main_experiment.py` runs greedy answers, labels, HAC/VASE, PTrue
for the seven main models, CAS features, protocol audits, comparison summaries,
and paper metric tables in that order. It uses seed 42, HAC N=10 / temperature 0.7 / cluster threshold
0.85, five stratified folds, and 1,000 bootstrap draws for the main comparison.
Run it with an environment that has the eight models and three datasets:

```bash
python scripts/run_main_experiment.py --dry-run
python scripts/run_main_experiment.py --cuda-visible-devices 0
```

This runner uses one GPU selection sequentially. `--evaluate-only` computes
summaries from existing JSONL inputs and checks that all 45
model/dataset/task settings have greedy, label, HAC, and CAS files, plus the
42 PTrue files used in the original full metric table. It accepts
existing error rows in CAS and evaluates only matched successful samples,
following the original comparison script. The result files are
`04_results/closed/json/summary.json`, `04_results/open/json/summary.json`,
and `04_results/paper_metrics/benchmark_main_metrics.csv`.

The closed CAS extractor writes 17 features, while the **primary main
comparison uses nine aggregate features** (`CAS-LogReg-cas_all`). The
17-feature variant is a separate feature ablation. Open primary
`CAS-LogReg-cas_only` is pure CAS; `CAS+Likelihood-LogReg-full` is a separate
augmented result.

## Repository layout

- `scripts/compute_cas_closed_from_greedy.py`: closed CAS features.
- `scripts/compute_cas_open_from_greedy.py`: open CAS features.
- `scripts/cas_runtime.py`, `scripts/cas_open_likelihood.py`, `confidence/grounding.py`: probe implementation.
- `scripts/benchmark_common.py`: shared data paths and row format.
- `scripts/generate_greedy_answers.py`, `scripts/label_greedy_answers.py`: shared targets and labels.
- `scripts/compute_hac_vase_from_greedy.py`, `scripts/evaluate_compare_hac_cas.py`: comparison.
- `scripts/run_main_experiment.py`: exact latest main-experiment scope and command chain.
- `scripts/build_paper_metric_tables.py`, `scripts/run_cas_feature_ablation.py` and plotting scripts: analysis.
- `src/medvlm`: model and dataset runtime.

## Reproducibility status

This release is a code package, not a claim that every historical experiment
finished successfully. The original full-scope output audit found six closed
CAS model/dataset files with ten distinct failed model/sample targets whose
greedy answers were not yes/no. No historical output files
are included here. `scripts/assert_full_scope_complete.py` is a strict audit
for the five-model full-scope protocol; run it only after producing that full
scope. Its expected counts do not apply to the ten-sample example above.

For release verification, the evaluation scripts in this repository were run
against the original cached JSONL files. All 24 closed and 21 open settings
matched the original sample counts, accuracy, AUROC values, and primary CAS
delta. The paper metrics builder reproduced 429 rows with matching model,
dataset, task, method, sample counts, AUROC, accuracy, and AURC. This verifies
evaluation from caches; it does not verify a fresh model inference run.
