# DiaGraMo — Locked Research Specification

## 1. Official Research Direction

The primary research architecture is:

**Multi-scale Relative Stroke Transformer + Kinematic LightGBM**

Repository: `Alexx-crearor/DiaGraMo_project`

Base branch: `study/raw-temporal-research-v2`

The objective is to investigate whether learned temporal handwriting representations provide information complementary to handcrafted Kinematic features.

Do not rebuild the existing network.

## 2. Locked Architecture

### Branch A: Kinematic

- 14 tasks × 74 features.
- Protected LightGBM baseline.
- Output: `pK`.

### Branch B: Raw Temporal

- Original online handwriting `.svc` signals.
- 21 preprocessed temporal channels.
- Stroke-aware, non-overlapping windows of up to 128 samples.
- Multi-scale CNN with kernels 5, 13, and 29.
- Four retained 32-dimensional tokens per window.
- Window-token aggregation into segment representations.
- Two-head relative attention over bounded groups of segment representations.
- Task identity embeddings and learned aggregation.
- Participant-level classification head.
- Output: `pT`.

**Critical definitions:**

1. Attention is performed over bounded segment-level representations, not the complete original time series.
2. Segments are computational units derived from pen-state and timestamp boundaries. Do not call them clinically annotated handwriting strokes.
3. Participant aggregation uses learned pooling, not a task-level Transformer.
4. Do not confuse the Stroke Transformer pooling strategy with duration-weighted pooling in `RecoveryModel`.

### Fusion

The frozen Phase 4 Lite experiment uses:

`pF = 0.5 * pK + 0.5 * pT`

No trainable fusion parameters.

## 3. Scientific Contributions to Investigate

### C1 — Structured Temporal Representation

Hypothesis: explicit temporal relationships and segment-level structure provide discriminative information beyond handcrafted Kinematic summaries.

Evidence required:

- Controlled relative-time and relative-spatial ablations.
- Matched local encoder and aggregation controls.
- Capacity and computational-budget comparisons.
- Order-sensitive perturbation experiments with explicit discussion of out-of-distribution effects.
- Assessment of whether differences can be explained by parameter count, pooling, optimization, or calibration.

Do not equate attention weights with a causal explanation.

### C2 — Complementary Motor Evidence

Hypothesis: the Temporal branch provides incremental predictive evidence beyond LightGBM.

Evidence required:

- Participant-matched Kinematic-only, Temporal-only and fixed-fusion comparisons.
- Complete six-fold OOF evaluation.
- Corrected versus harmed predictions.
- Fold-wise consistency.
- Paired bootstrap confidence intervals.
- ROC-AUC, Macro-F1, sensitivity, specificity, Brier score and calibration diagnostics.
- A train-only calibrated Kinematic control in any subsequent independent experiment.

Do not fit calibration or fusion parameters on existing outer-test predictions.

### C3 — Cross-Domain Robustness and Explainability

This is a future research direction, not an established contribution.

External-domain, language, recruitment-site or acquisition-condition generalization requires appropriate independent data.

Do not claim clinical biomarkers, motor invariance or cross-language robustness without direct evidence.

## 4. Immediate Execution Priority

Complete the existing frozen Phase 4 Lite evaluation.

The original protocol must remain unchanged:

- Seed: 42.
- Epochs: 28.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.001.
- Dropout: 0.1.
- Embedding dimension: 32.
- Attention heads: 2.
- Effective batch: 8.
- Original participant-independent six-fold assignments.
- Fixed 50:50 probability fusion.

Resume existing checkpoints where valid.

Do not retrain successful folds.

Do not change the architecture, model-selection rule, checkpoint selection, threshold or fusion weight based on partial outer results.

Inspect local state before training. If GPU execution is unavailable, report the blocker rather than pretending training has completed.

## 5. Mandatory Data Validity Audit

Audit the strong association between participant-ID prefixes and the dataset's `group` labels.

Determine whether recruitment source, collection batch, recording hardware or site metadata may explain this association.

Never use participant ID or ID prefix as predictive features.

Distinguish data-source confounding, direct feature leakage, and researcher-level selection bias.

Maintain the distinction between participant-independent model fitting and independent validation of model-selection decisions.

The completed Phase 4 Lite study remains exploratory internal CV, not externally validated evidence.

## 6. Stop Condition and Research Decision

After six folds are complete, report the full results before initiating another experiment.

If fixed fusion demonstrates stable complementary value, propose a separate train-only fusion study.

If results are weak, inconsistent or confounded, document this and recommend further validation rather than automatically adding model complexity.

Do not start SSL, larger Transformers, selective fusion, architecture grids or threshold tuning in the frozen experiment.

## 7. Required Final Output

Produce a Vietnamese scientific report containing:

1. Exact implemented architecture.
2. Dataset and preprocessing audit.
3. Complete OOF evaluation and matched baselines.
4. Controlled ablation interpretation.
5. Complementarity/error analysis.
6. Statistical uncertainty and calibration limitations.
7. Data confounding and validation limitations.
8. Decision on whether to pursue C1/C2-focused follow-up experiments.
9. Research contributions supported by actual evidence, clearly separated from untested hypotheses.

Preserve historical artifacts, checkpoints and baseline results. Do not push or merge automatically.

**Scientific validity takes precedence over obtaining a higher F1 score.**