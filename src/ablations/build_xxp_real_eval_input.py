import json
from pathlib import Path

SRC = Path(
    "/workspace/STAGE8B_HANDOFF_FINAL/frozen/"
    "STAGE8A_FROZEN_EVIDENCE.json"
)

OUT = Path(
    "/workspace/STAGE10F_MATCHED_EVAL_INPUT_XXP_REAL.json"
)

x = json.loads(SRC.read_text())

xxp = x["methods"]["XXP_FULL"]

assert len(xxp) == 64

out = {
    "status": "READY",
    "cohort": "RouterCal64",
    "n": 64,
    "methods": {
        "XXP_FULL": xxp,
        "CANDIDATE_X": xxp
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
print("claims =", len(xxp))
