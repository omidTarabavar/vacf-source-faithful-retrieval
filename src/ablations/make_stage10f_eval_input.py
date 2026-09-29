import json
import hashlib
from pathlib import Path

BASE = Path(
    "/workspace/V5B_FREEZE/"
    "CANDIDATE_Y_64_FROZEN.json"
)

V5B = Path(
    "/workspace/STAGE10E_RESULTS/"
    "CAL64_V5B_PROTECTED_FUSION.json"
)

OUT = Path(
    "/workspace/STAGE10F_MATCHED_EVAL_INPUT.json"
)

base = json.loads(
    BASE.read_text(encoding="utf-8")
)

v5b = json.loads(
    V5B.read_text(encoding="utf-8")
)

baseline_rows = base["methods"]["CANDIDATE_X"]

v5b_rows = v5b[
    "methods"
]["V5B_PROTECTED_FUSION"]

assert len(baseline_rows) == 64
assert len(v5b_rows) == 64

baseline_by_id = {
    r["claim_id"]: r
    for r in baseline_rows
}

v5b_by_id = {
    r["claim_id"]: r
    for r in v5b_rows
}

assert set(baseline_by_id) == set(v5b_by_id)

ordered_base = []
ordered_v5b = []

for br in baseline_rows:
    cid = br["claim_id"]
    vr = v5b_by_id[cid]

    assert (
        br["claim_text"]
        == vr["claim_text"]
    ), cid

    assert 1 <= len(br["evidence"]) <= 10
    assert 1 <= len(vr["evidence"]) <= 10

    for e in br["evidence"]:
        assert e.get("url")
        assert (e.get("text") or "").strip()

    for e in vr["evidence"]:
        assert e.get("url")
        assert (e.get("text") or "").strip()

    bcopy = dict(br)
    bcopy["method"] = "CANDIDATE_Y_BASELINE"

    vcopy = dict(vr)
    vcopy["method"] = "V5B_PROTECTED_FUSION"

    ordered_base.append(bcopy)
    ordered_v5b.append(vcopy)

out = {
    "status": "STAGE10F_MATCHED_EVAL_INPUT",
    "cohort": "Cal64",
    "n": 64,

    "IMPORTANT_LABEL_MAPPING": {
        "XXP_FULL":
            "Candidate-Y baseline; NOT actual XxP",
        "CANDIDATE_X":
            "V5-B Protected Fusion"
    },

    "qrels_used_for_selection_or_hydration":
        False,

    "methods": {
        "XXP_FULL":
            ordered_base,

        "CANDIDATE_X":
            ordered_v5b
    }
}

OUT.write_text(
    json.dumps(
        out,
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)

raw = OUT.read_bytes()

print("claims =", len(ordered_base))
print("baseline evidence =", sum(
    len(x["evidence"])
    for x in ordered_base
))
print("V5B evidence =", sum(
    len(x["evidence"])
    for x in ordered_v5b
))

print(
    "SHA256 =",
    hashlib.sha256(raw).hexdigest()
)

print(
    "STAGE10F_MATCHED_INPUT_OK"
)
