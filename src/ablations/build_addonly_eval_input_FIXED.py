import json
from pathlib import Path

BASE = Path(
    "/workspace/STAGE10F_MATCHED_EVAL_INPUT_NORMALIZED.json"
)

ADD = Path(
    "/workspace/STAGE10F_RESULTS/"
    "CAL64_ADDONLY_FUSION.json"
)

OUT = Path(
    "/workspace/STAGE10F_MATCHED_EVAL_INPUT_ADDONLY_FIXED.json"
)

base = json.loads(BASE.read_text())
add = json.loads(ADD.read_text())

xxp = base["methods"]["XXP_FULL"]

candidate = add["methods"]["CANDIDATE_Y_ADDONLY"]


assert len(xxp) == 64
assert len(candidate) == 64

assert {
    x["claim_id"] for x in xxp
} == {
    x["claim_id"] for x in candidate
}


out = {
    "status": "READY",
    "cohort": "RouterCal64",
    "n": 64,
    "methods": {
        "XXP_FULL": xxp,
        "CANDIDATE_X": candidate
    }
}


OUT.write_text(
    json.dumps(
        out,
        ensure_ascii=False,
        indent=2
    )
)


print("DONE")
print(OUT)
print("XXP =",len(xxp))
print("ADDONLY =",len(candidate))
