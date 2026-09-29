import json
from pathlib import Path

SRC = Path(
    "/workspace/STAGE10F_RESULTS/"
    "CAL64_ADDONLY_FUSION.json"
)

OUT = Path(
    "/workspace/STAGE10F_MATCHED_EVAL_INPUT_ADDONLY.json"
)

x = json.loads(SRC.read_text())

rows = x["methods"]["CANDIDATE_Y_ADDONLY"]

# evaluator expects list under method name CANDIDATE_X
out = {
    "methods": {
        "CANDIDATE_X": rows,
        "XXP_FULL": rows
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
print("claims =",len(rows))
