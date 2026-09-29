# VACF — Visual-Anchored Context Fusion

Research-code release for **Visual-Anchored Context Fusion (VACF)**, a
training-free retrieval framework for source-faithful evidence construction in
multimodal fact verification.

VACF uses two frozen image-conditioned retrieval streams over complementary
AVerImaTeC knowledge-store branches. A visual-source ranking forms a protected
prefix, contextual retrieval fills only remaining evidence slots, and
source-faithful restoration expands text only from the same selected URL and
originating branch.

## Repository contents

```text
src/
  retrieval/      image-conditioned ranking and protected source selection
  restoration/    source-faithful local context restoration
  downstream/     fixed downstream verifier scripts used in matched analyses
  evaluation/     official/seeded evaluation scripts
  ablations/      component and diagnostic experiments
  efficiency/     latency and GPU-memory benchmarks
configs/           example path configuration
external/          third-party repository URLs and exact commits
data/              instructions for obtaining benchmark resources
results/           aggregate manuscript tables (not claim-level artifacts)
docs/              reproducibility and data-availability notes
```

## Core models

- `google/siglip-base-patch16-384` — visual-anchor stream
- `google/siglip2-large-patch16-512` — contextual-completion stream
- `google/gemma-3-27b-it` — benchmark semantic evaluator
- `Qwen/Qwen3-VL-8B-Instruct` — fixed downstream verifier in matched runs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then obtain the AVerImaTeC benchmark resources and clone the external
repositories listed in `external/README.md`.

## Important reproducibility note

The uploaded experiment archive was **not a self-contained clean-room package**:
it preserved exact scripts from the original `/workspace` instance, and many
scripts contain absolute `/workspace/...` defaults. The exact code is retained
here for scientific provenance. See `docs/PATHS_AND_INPUTS.md` for path mapping.

The source archive also did not include final claim-level prediction/evaluation
artifacts. The aggregate paper values are provided under `results/`, while
`docs/MISSING_ARTIFACTS.md` lists the derived artifacts that should be archived
with the public release (ideally via Zenodo).

## External code

Third-party source repositories are not copied into this cleaned release. Exact
URLs and commits are recorded under `external/`.

## Citation

See `CITATION.cff`. The paper citation/DOI can be added after publication.

## License

Author-owned code in this repository is released under the MIT License. See `LICENSE`. Third-party repositories are not vendored and remain subject to their own licenses and terms; see `LICENSE_NOTICE.md` and `external/README.md`.
