# Reproducibility map

This repository is organized around the stages reported in the VACF paper.

## 1. Image-conditioned dual ranking

Primary script:

- `src/retrieval/stage7_run_expert.py`

Run once with the SigLIP-base configuration and once with the SigLIP2-large
configuration. The script uses 60-token non-overlapping chunks, computes
image-text similarities, aggregates the top three chunk scores at URL level,
uses the maximum URL score over multiple claim images, and retains the Top-100
URLs per branch.

## 2. Protected asymmetric fusion / frozen source selection

- `src/retrieval/stage8a_freeze_val152.py`
- `src/retrieval/stage8a_freeze_test352.py`

The visual ranking is scanned first and contextual URLs fill remaining slots
without reranking or displacing admitted visual sources.

## 3. Source-faithful restoration

- `src/restoration/stage8c_restore_context_val152.py`
- `src/restoration/stage8c_restore_context_test352.py`

Restoration localizes the selected ranking chunks in raw pieces from the same
originating branch, expands by local neighbors and early source pieces, and
packs evidence under the 1,200-token source budget.

## 4. Downstream verification

See `src/downstream/`. The matched Val152 scripts keep the downstream verifier
fixed across compared retrieval conditions.

## 5. Official/seeded evaluation

See `src/evaluation/`. These scripts depend on evaluation utilities from the
organizer repository listed under `external/`.

## 6. Ablation and diagnostics

See `src/ablations/`.

## 7. Efficiency benchmark

See `src/efficiency/benchmark_val152_efficiency.py` and
`src/efficiency/run_eff152_sequential.sh`.

## Reproducibility limitation of the supplied archive

The source ZIP contains the experiment code but not the claim-level final
prediction/evaluation artifacts and not the script that originally created some
workspace-specific intermediate state (`STATE.json` and certain per-claim XxP
prediction files). Consequently, the archive is valuable as the exact source
record but is not yet a one-command, from-scratch reproducibility package.

For publication, retain these exact scripts and additionally archive the final
claim-level derived artifacts (preferably with a Zenodo DOI).
