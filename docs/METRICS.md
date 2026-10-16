# Closed-question metrics

- `accuracy`: correctness of the shared deterministic greedy answer.
- `AUROC`: error detection, where label 1 means an incorrect answer and
  larger scores mean greater risk.
- `HAC-best`: the higher AUROC of HAC-Platt and HAC-Gate within each setting.
- `CAS-primary`: `CAS-LogReg-cas_all`, fit from the predefined nine CAS
  aggregate features under five stratified folds.
- `delta`: CAS-primary AUROC minus HAC-best AUROC in the same setting.
- `n_intersection`: samples with successful HAC and CAS rows and matching
  correctness labels; failed CAS rows do not enter the metric.

The main result contains 21 model/dataset settings. The paper metrics builder
also reports selective prediction metrics and other baselines on the matched
rows.
