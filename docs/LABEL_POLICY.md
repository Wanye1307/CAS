# Label Policy

## Policy

Each benchmark row uses exactly one correctness label per sample. That label
must judge the deterministic greedy answer.

For HAC rows, the deterministic greedy answer is the `greedy_answer` generated
inside the VASE/HEDGE run before high-temperature sampling. The
`original_high_temp` and `distorted_high_temp` generations are never label
sources.


## Current Benchmark Label Rule

Closed labels use normalized yes/no exact matching against the shared greedy
prediction. Open labels use normalized exact matching against the shared greedy
answer text: answer-prefix stripping, lowercase, and terminal-period
normalization are allowed; substring containment is not used.

## Required Normalized Label Schema

Each normalized label file under `02_inputs/labels/` should contain JSONL rows
with:

- `model`
- `dataset`
- `qtype`
- `question_id`
- `image_id`
- `question`
- `ground_truth`
- `prediction`
- `is_correct`
- `judge_model`
- `judge_method`
- `source_path`
- `source_index`

## Label Audits

For each model/dataset/qtype, record:

- number of rows,
- number correct,
- number incorrect,
- accuracy,
- duplicate question IDs,
- missing labels,
- label source,
- whether labels judge the same answer used by HAC and CAS.

## Pending Source Mapping

The final source mapping is not frozen yet. Candidate sources must be reviewed
before being copied into `02_inputs/labels/`.
