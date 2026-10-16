# Metrics

## Primary Metrics

- AUROC for error detection.
- Accuracy of the greedy answers.
- Delta between fixed CAS and HAC-best.

## HAC Metrics

HAC candidates:

- HAC-Platt
- HAC-Gate

HAC-best:

```text
max(AUROC(HAC-Platt), AUROC(HAC-Gate))
```

## CAS Metrics

Fixed CAS candidates are protocol-defined before looking at results.

Current candidate definitions:

- closed: `CAS-LogReg-cas_all`
- open: `CAS-LogReg-cas_full`

These are not final until the benchmark scope is frozen.

## Exploratory Metrics

- CV-selected CAS best.
- Best raw CAS signal.
- Other calibration or selective metrics if needed.

Exploratory metrics must be marked as exploratory in final tables.
