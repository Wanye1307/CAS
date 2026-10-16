# Closed-question benchmark protocol

## Scope

Seven model keys (`qwen3vl_2b`, `qwen3vl_8b`,
`internvl3_2b_local`, `internvl3_8b_local`, `lingshu_7b`,
`medgemma_4b`, `medvlm_r1`) are evaluated on closed questions from
VQA-RAD, SLAKE, and VQA-Med-2019: 21 model/dataset settings.

## Shared answer and label

One deterministic greedy answer is generated for each sample. Its normalized
yes/no value is compared with the normalized ground truth to produce the
correctness label. HAC and CAS use that same target and label. A
high-temperature sample or sampling majority is never used as the label.
If a greedy answer cannot be normalized to yes/no, closed CAS records an
error and the HAC/CAS comparison excludes that sample from the strict matched
intersection.

## HAC/VASE

Per sample, ten original-image and ten distorted-image high-temperature
answers are drawn. HAC confidence is the fraction of ten original answers
matching the shared greedy yes/no target. Invalid sampled answers remain in
the denominator. VASE is calculated from original and distorted samples.
HAC-Platt and HAC-Gate are fit in the same cross-validation protocol.

## CAS

CAS scores the shared greedy yes/no answer under the original image and four
counterfactual inputs: text only, black image, noise image, and blurred image.
The extractor writes 17 features. The fixed primary detector uses nine
aggregate features named `CAS-LogReg-cas_all` in the evaluation code.

## Evaluation

The comparison uses the intersection of valid HAC and CAS rows with matching
correctness labels. The primary metric is error-detection AUROC and its
per-setting difference from the better of HAC-Platt and HAC-Gate. Five
stratified folds use seed 42. The main comparison computes 1,000 paired
bootstrap draws for each setting.
