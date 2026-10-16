# Shared greedy-label policy

Every closed-question sample has one deterministic greedy target. Its
normalized yes/no prediction is compared with the normalized yes/no ground
truth. The label is shared by HAC, PTrue, and CAS. High-temperature answers
are used for uncertainty estimation only.

`scripts/label_greedy_answers.py` writes one JSONL row per successful greedy
sample under `02_inputs/labels/`. The row records model, dataset, question
type, sample/question/image identifiers, ground truth, target prediction,
`is_correct`, judge method, and source. An external label file can be passed
with `--external-labels`, provided it judges the same greedy target.

`scripts/audit_protocol_inputs.py` checks that greedy, label, HAC, and CAS
rows refer to the same target and correctness label. A CAS row with an
unparseable yes/no target has status `error` and is excluded by the strict
HAC/CAS intersection.
