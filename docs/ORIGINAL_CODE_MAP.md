# VACF Code Archive

## Core retrieval
- `stage8b_handoff/stage7_run_expert.py`
  - SigLIP-base / SigLIP2-large retrieval
  - 60-token chunks
  - image-conditioned text scoring
  - URL ranking

- `vacf_core/stage8a_freeze_val152.py`
  - protected visual anchoring
  - contextual URL completion
  - frozen URL selection

- `vacf_core/stage8c_restore_context_val152.py`
  - source-faithful restoration
  - raw source-piece matching
  - contextual expansion
  - token-budget packing

## Downstream verification
See `downstream/`.

## Official/seeded evaluation
See `evaluation/`.

## Ablations and development experiments
See `ablations_and_analysis/`.

## Efficiency benchmarking
See `efficiency/`.

## Reference systems
- `xxp_official/`
- `upstream_modified/`

## Reproducibility
The Git commit, full repository bundle and local non-secret patch are
stored under `git/`.

API keys and credentials are intentionally excluded.
