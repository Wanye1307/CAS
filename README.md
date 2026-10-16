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

## Repository layout

- `scripts/compute_cas_closed_from_greedy.py`: closed CAS features.
- `scripts/compute_cas_open_from_greedy.py`: open CAS features.
- `scripts/cas_runtime.py`, `scripts/cas_open_likelihood.py`, `confidence/grounding.py`: probe implementation.
- `scripts/benchmark_common.py`: shared data paths and row format.
- `scripts/generate_greedy_answers.py`, `scripts/label_greedy_answers.py`: shared targets and labels.
- `scripts/compute_hac_vase_from_greedy.py`, `scripts/evaluate_compare_hac_cas.py`: comparison.
- `scripts/build_paper_metric_tables.py`, `scripts/run_cas_feature_ablation.py` and plotting scripts: analysis.
- `src/medvlm`: model and dataset runtime.

## Reproducibility status

This release is a code package, not a claim that every historical experiment
finished successfully. The original full-scope output audit found six closed
CAS model/dataset files with missing or error rows. No historical output files
are included here. `scripts/assert_full_scope_complete.py` is a strict audit
for the five-model full-scope protocol; run it only after producing that full
scope. Its expected counts do not apply to the ten-sample example above.
