# VACF derived artifacts — public release

This compact package contains derived artifacts supporting the Val152 analyses reported in the VACF paper. It is designed for public release without redistributing the large full-text evidence payloads from benchmark/Web sources.

## Included

### Val152 matched analysis
- `VAL152_ALL_METRICS_PAIRED_STATS_FINAL.json` — matched means, 10,000-sample paired-bootstrap intervals, and win/loss/tie counts for Question, Evidence, Verdict, and Justification.
- `VAL152_XXP_VACF_PER_CLAIM_EVIDENCE.csv` — 152 claim IDs with per-claim XxP and VACF Evidence Scores, deltas, outcomes, and evidence counts. Claim text is intentionally omitted.
- `VAL152_EVIDENCE_SCORE_SUMMARY.json` — compact aggregate cross-check for the three evidence-score comparisons.

### Component / ablation analysis
- `VAL152_VISUAL_CONTEXTUAL_PER_CLAIM_EVIDENCE.csv` — per-claim Evidence Scores for Visual-only versus Contextual-only retrieval.
- `VAL152_RESTORATION_PER_CLAIM_EVIDENCE.csv` — per-claim Evidence Scores for full VACF versus VACF without restoration.
- `VAL152_RESTORATION_AUDIT.json` — structural audit showing whether source set/order/count changed under restoration.

### Retrieval provenance
- `VAL152_SOURCE_SELECTION_PUBLIC.json` — frozen full-Val152 source selections, URL rank, provenance stream/expert, and source score; claim text removed.
- `VAL152_RETRIEVAL_COMPOSITION.csv` — visual/contextual occupancy distribution used in the paper.

### Efficiency
- `EFF152_COMPARISON_FINAL.json` — matched aggregate timing and GPU-memory comparison.
- `EFF152_VACF.json` and `EFF152_XXP.json` — per-claim efficiency benchmark records.

### Verification
- `scripts/summarize_public_artifacts.py` — standard-library script that recomputes the main Evidence-Score means and win/loss/tie counts from the public CSVs and prints the exact manuscript bootstrap result stored in the paired-statistics artifact.
- `SHA256SUMS.txt` — integrity hashes for all release files.

## Why the raw Stage8B JSON files are not redistributed

The private evaluator outputs contain large blocks of predicted evidence, benchmark reference evidence, and evaluator feedback. Those files are useful for internal audit, but are not required for reproducing the aggregate results and may unnecessarily redistribute benchmark/Web text. This public package therefore exposes the derived per-claim scores and audits while keeping raw text payloads out of the release.

## Artifact mapping

The uploaded private files named `VAL152_EVAL_VIS_CTX_STAGE8B_RESULTS.json`, `VAL152_EVAL_WOREST_STAGE8B_RESULTS.json`, and `VAL152_FINAL_EVAL_STAGE8B_RESULTS.json` use generic internal method keys (`XXP_FULL` and `CANDIDATE_X`). In this release they are relabeled according to the experimental conditions they represent:

- `VAL152_FINAL_EVAL_STAGE8B_RESULTS.json`: XxP vs full VACF.
- `VAL152_EVAL_VIS_CTX_STAGE8B_RESULTS.json`: Visual-only vs Contextual-only.
- `VAL152_EVAL_WOREST_STAGE8B_RESULTS.json`: VACF without restoration vs full VACF.

The public CSVs use these descriptive names directly.

## Legacy-artifact note

Two legacy private files named `VAL152_CANDIDATE_Y.json` and `VAL152_STAGE8A_FROZEN_EVIDENCE.json` report `n = 64` and are intermediate 64-claim artifacts. They should not be presented as full-Val152 release artifacts. Full-Val152 frozen source selection is represented here by `VAL152_SOURCE_SELECTION_PUBLIC.json` (`n = 152`).
