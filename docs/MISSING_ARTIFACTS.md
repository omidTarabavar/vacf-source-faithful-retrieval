# Missing artifacts from the supplied ZIP

The supplied `VACF_CODE_FINAL_20260923.zip` contained source code and Git
provenance, but no final JSON/JSONL/CSV result artifacts other than a file
manifest. Therefore this cleaned repository cannot honestly claim that all
underlying derived evaluation data are included yet.

Recommended artifacts to add before/alongside publication:

1. Val152 frozen source-selection JSON.
2. Val152 source-faithful restored evidence JSON.
3. Val152 fixed-question downstream XxP output.
4. Val152 fixed-question downstream VACF output.
5. Per-claim Evidence/Verdict/Justification scores used for paired analysis.
6. Paired-bootstrap output or the per-claim table from which it is recomputed.
7. VACF efficiency benchmark output.
8. Test352 submission file if sharing is permitted by the task rules.

These can live in a Zenodo release instead of GitHub if size or benchmark terms
make GitHub unsuitable.
