# Benchmark Protocol

## Goal

Compare HAC and CAS/V6 uncertainty scores for medical VQA under one shared
correctness-label protocol.

## Correctness Labels

Primary correctness labels must be derived from a single deterministic greedy
answer per sample. This shared greedy answer is generated once by
`scripts/generate_greedy_answers.py` and then reused by HAC, closed CAS, open
CAS, and label normalization.

It is not the first high-temperature sample and is never replaced by a
sampling-majority answer.

Disallowed as primary labels:

- HAC n=20 sampling majority.
- HAC/VASE high-temperature sample majority.
- Any label source that judges a different answer from the greedy answer.
- Any label source that differs between HAC and CAS for the same setting.

## HAC

Primary HAC uses the shared greedy target plus one VASE/HEDGE sampling bundle
per sample:

- `target_answer`: the shared deterministic original-image answer; used for the
  benchmark prediction and correctness label.
- `original_high_temp`: `N=10` original-image high-temperature answers; used to
  compute HAC confidence.
- `distorted_high_temp`: `N=10` distorted-image high-temperature answers; used
  with the original-image samples to compute VASE.
- `VASE`: hallucination/risk score from clean and distorted semantic
  distributions.
- `HAC-Platt` and `HAC-Gate`: fitted from HAC confidence and VASE.

HAC confidence is:

```text
confidence = count(original_high_temp answer matches greedy_answer) / 10
```

For closed-ended rows, matching uses normalized yes/no. Invalid or unparsable
high-temperature answers count as non-matches and stay in the denominator. For
open-ended rows, matching uses the same semantic clustering policy used by the
VASE/HEDGE run, with the benchmark default embedding threshold `0.85`.

HAC may not use high-temperature generations as correctness labels.

## CAS/V6

Closed CAS/V6:

- Uses the shared greedy yes/no answer.
- Compares answer support under real image and four fixed counterfactual inputs:
  `textonly`, `black`, `noise`, and `blur`.
- Primary signal is `CAS-LogReg-cas_all`.
- Must align `is_correct` to the shared greedy correctness label.

Open CAS/V6:

- Uses the shared greedy answer text.
- Teacher-forces the same answer tokens under real image and four fixed
  counterfactual inputs: `text`, `black`, `noise`, and `blur`.
- Primary signal is pure `CAS-LogReg-cas_only`.
- `CAS+Likelihood-full` is ablation/exploratory only.
- Must align `is_correct` to the shared greedy correctness label.

## Primary Metrics

Primary comparison:

```text
delta = fixed_CAS_AUROC - HAC_best_AUROC
```

where:

- `HAC_best_AUROC` is the better AUROC of HAC-Platt and HAC-Gate.
- `fixed_CAS_AUROC` uses the predefined fixed CAS configuration.

## Exploratory Metrics

CV-selected CAS is exploratory unless explicitly promoted later.

## Reproducibility Requirement

Every final result must have:

- input paths,
- command,
- output path,
- label source,
- audit notes.
