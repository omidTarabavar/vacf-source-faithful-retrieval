# Results included in this release package

The CSV files in this directory transcribe the **aggregate values reported in
the manuscript**. They are included to make the public repository easy to
inspect, but they are not a substitute for claim-level prediction/evaluation
artifacts.

## Important missing artifacts from the supplied code archive

The uploaded source archive did **not** contain the final JSON/JSONL/CSV
prediction and evaluation outputs. For the strongest reproducibility release,
add the final claim-level artifacts used to produce the paper tables, such as:

- frozen Val152 VACF source selections;
- restored Val152 evidence;
- frozen/restored Test352 submission evidence (where redistribution is allowed);
- matched Val152 XxP and VACF downstream outputs;
- per-claim Evidence / Verdict / Justification evaluation outputs;
- paired-bootstrap input/output records;
- efficiency benchmark JSON/CSV output.

If these artifacts are too large for GitHub, archive them in Zenodo and link
the DOI from the repository README and manuscript Data Availability section.
