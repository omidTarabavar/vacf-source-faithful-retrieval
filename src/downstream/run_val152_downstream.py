import sys
import json
import os
import random
import argparse
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/XXP_OFFICIAL")
from GenerationLLMHandler import GenerationLLMHandler


LOCKED = Path(
    "/workspace/PAPER_VAL152_LOCKED_20260917/"
    "VAL152_XXP_REAL_VS_CANDIDATE_Y.json"
)

QUESTIONS = Path("/workspace/XXP_VAL152/submission.json")

VAL_JSON = Path(
    "/workspace/AVerImaTec_Shared_Task/"
    "data/data_clean/split_data/val.json"
)

MODEL = (
    "/workspace/.hf_home/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    required=True,
    choices=["xxp", "vacf"],
)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()

if args.method == "xxp":
    METHOD_KEY = "XXP_FULL"
    OUT = Path("/workspace/VAL152_XXP_QWEN_FINAL.json")
else:
    METHOD_KEY = "CANDIDATE_X"
    OUT = Path("/workspace/VAL152_VACF_QWEN_FINAL.json")


# ---------------------------------------------------------
# Reproducibility — same as Test352 downstream
# ---------------------------------------------------------

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------
# Load frozen inputs
# ---------------------------------------------------------

locked = json.loads(LOCKED.read_text(encoding="utf-8"))
question_rows = json.loads(QUESTIONS.read_text(encoding="utf-8"))
val = json.loads(VAL_JSON.read_text(encoding="utf-8"))

method_rows = locked["methods"][METHOD_KEY]

assert len(method_rows) == 152
assert len(question_rows) == 152
assert len(val) == 152


# ---------------------------------------------------------
# Exact alignment checks
# ---------------------------------------------------------

for i in range(152):

    expected_cid = f"averi_val_{i:04d}"

    assert method_rows[i]["claim_id"] == expected_cid, (
        i,
        method_rows[i]["claim_id"],
        expected_cid,
    )

    qid = int(question_rows[i]["id"])
    assert qid == i, (i, qid)

    claim_val = (
        val[i].get("claim_text")
        or val[i].get("claim")
        or ""
    ).strip()

    claim_locked = method_rows[i]["claim_text"].strip()

    assert claim_val == claim_locked, (
        i,
        claim_val,
        claim_locked,
    )


# ---------------------------------------------------------
# Resolve validation claim images to absolute paths.
# We pass absolute paths to GenerationLLMHandler and use "/"
# as image_root, avoiding assumptions about the image folder.
# ---------------------------------------------------------

needed = set()

for row in val:
    for ref in (row.get("claim_images", []) or []):
        if isinstance(ref, str):
            needed.add(Path(ref).name)

print("Unique claim image basenames:", len(needed), flush=True)

search_roots = [
    Path("/workspace/AVerImaTec_Shared_Task/data"),
    Path("/workspace/Knowledge_Store/val_extracted"),
]

found = {}

for root in search_roots:
    if not root.exists():
        continue

    for dirpath, dirs, files in os.walk(root):

        # Avoid irrelevant large converted datastore trees when possible.
        dirs[:] = [
            d for d in dirs
            if d not in {
                "converted_datastore",
                "__pycache__",
            }
        ]

        for fn in files:
            if fn in needed and fn not in found:
                found[fn] = str(Path(dirpath) / fn)

        if len(found) == len(needed):
            break

    if len(found) == len(needed):
        break


missing = sorted(needed - set(found))

print("Resolved claim images:", len(found), "/", len(needed), flush=True)

if missing:
    print("MISSING IMAGE BASENAMES:", flush=True)
    for x in missing[:20]:
        print(" ", x, flush=True)
    raise RuntimeError(
        f"Could not resolve {len(missing)} validation claim images"
    )


def abs_claim_images(row):
    out = []

    for ref in (row.get("claim_images", []) or []):
        if not isinstance(ref, str):
            raise TypeError(
                f"Unexpected claim image ref type: {type(ref)}"
            )

        out.append(found[Path(ref).name])

    return out


# ---------------------------------------------------------
# Build immutable skeleton from:
# questions = original matched XxP questions
# evidence  = locked XxP/VACF evidence
# ---------------------------------------------------------

fresh_rows = []

for i in range(152):
    fresh_rows.append({
        "id": i,
        "questions": question_rows[i]["questions"],
        "evidence": method_rows[i]["evidence"],
        "verdict": "",
        "justification": "",
    })


# Resume only from this method's own output.
if OUT.exists():

    rows = json.loads(OUT.read_text(encoding="utf-8"))

    assert len(rows) == 152

    # Protect frozen questions/evidence against accidental drift.
    for i in range(152):
        assert rows[i]["id"] == fresh_rows[i]["id"]
        assert rows[i]["questions"] == fresh_rows[i]["questions"]
        assert rows[i]["evidence"] == fresh_rows[i]["evidence"]

    print("RESUME:", OUT, flush=True)

else:
    rows = fresh_rows


def atomic_save(obj):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, OUT)


# ---------------------------------------------------------
# Preflight only
# ---------------------------------------------------------

print("METHOD:", args.method, flush=True)
print("METHOD_KEY:", METHOD_KEY, flush=True)
print("OUTPUT:", OUT, flush=True)
print(
    "Question counts first/last:",
    len(rows[0]["questions"]),
    len(rows[-1]["questions"]),
    flush=True,
)
print(
    "Evidence counts first/last:",
    len(rows[0]["evidence"]),
    len(rows[-1]["evidence"]),
    flush=True,
)
print(
    "Images first/last:",
    len(abs_claim_images(val[0])),
    len(abs_claim_images(val[-1])),
    flush=True,
)

if args.check_only:
    print("CHECK_OK", flush=True)
    raise SystemExit(0)


# ---------------------------------------------------------
# Load exact downstream model used previously
# ---------------------------------------------------------

print("Loading:", MODEL, flush=True)

gen = GenerationLLMHandler(
    image_root="/",
    model_name=MODEL,
)

print("MODEL READY", flush=True)


# ---------------------------------------------------------
# Downstream verdict + justification
# ---------------------------------------------------------

for pos, row in enumerate(rows, 1):

    idx = int(row["id"])

    if row.get("verdict") and row.get("justification"):
        print(
            f"[CACHE] {pos}/152 id={idx} "
            f"verdict={row['verdict']}",
            flush=True,
        )
        continue

    gold_row = val[idx]

    claim = (
        gold_row.get("claim_text")
        or gold_row.get("claim")
        or ""
    ).strip()

    claim_imgs = abs_claim_images(gold_row)

    evid_texts = []

    for e in row.get("evidence", [])[:10]:
        t = (e.get("text", "") or "").strip()
        if t:
            evid_texts.append(t)

    if evid_texts:
        joined_evidence = "\n".join(evid_texts)
    else:
        joined_evidence = "No relevant evidence was retrieved."

    print(
        f"[RUN] {pos}/152 id={idx} "
        f"evidence={len(row.get('evidence', []))} "
        f"images={len(claim_imgs)}",
        flush=True,
    )

    verdict = gen.generate_prediction(
        claim=claim,
        retrieved_evidences=joined_evidence,
        image_refs=claim_imgs,
        max_tokens=64,
    )

    justification = gen.generate_justification(
        claim=claim,
        retrieved_evidences=joined_evidence,
        verdict=verdict,
        image_refs=claim_imgs,
        max_tokens=256,
    )

    if not justification:
        raise RuntimeError(
            f"Empty justification at id={idx}"
        )

    row["verdict"] = verdict
    row["justification"] = justification

    atomic_save(rows)

    print(
        f"[DONE] {pos}/152 id={idx} "
        f"verdict={verdict} "
        f"just_chars={len(justification)}",
        flush=True,
    )


atomic_save(rows)

from collections import Counter

print("\nFINAL:", OUT)
print("N:", len(rows))
print(
    "missing verdict:",
    sum(not r.get("verdict") for r in rows),
)
print(
    "missing justification:",
    sum(not r.get("justification") for r in rows),
)
print(
    "verdict distribution:",
    dict(Counter(r["verdict"] for r in rows)),
)
