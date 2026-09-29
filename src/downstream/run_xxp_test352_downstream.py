import sys
import json
import os
import random
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/XXP_OFFICIAL")
from GenerationLLMHandler import GenerationLLMHandler

SKELETON = Path("/workspace/TEST352_XXP/predictions/submission.json")
TEST_JSON = Path("/workspace/AVERIMATEC_TEST_OFFICIAL/extracted/test_data/test.json")
IMAGE_ROOT = "/workspace/AVERIMATEC_TEST_OFFICIAL/extracted/test_data/images"

OUT = Path("/workspace/TEST352_XXP_QWEN_FINAL.json")

MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

skeleton = json.loads(SKELETON.read_text(encoding="utf-8"))
test = json.loads(TEST_JSON.read_text(encoding="utf-8"))

assert len(skeleton) == 352
assert len(test) == 352

# Resume if a partial output already exists.
if OUT.exists():
    rows = json.loads(OUT.read_text(encoding="utf-8"))
    assert len(rows) == 352
    print("RESUME:", OUT, flush=True)
else:
    rows = skeleton

def atomic_save(obj):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(tmp, OUT)

print("Loading:", MODEL, flush=True)

gen = GenerationLLMHandler(
    image_root=IMAGE_ROOT,
    model_name=MODEL
)

print("MODEL READY", flush=True)

for pos, row in enumerate(rows, 1):
    idx = int(row["id"])

    # Skip already completed rows on resume.
    if row.get("verdict") and row.get("justification"):
        print(
            f"[CACHE] {pos}/352 id={idx} "
            f"verdict={row['verdict']}",
            flush=True
        )
        continue

    gold_row = test[idx]

    claim = (
        gold_row.get("claim_text")
        or gold_row.get("claim")
        or ""
    ).strip()

    claim_imgs = gold_row.get("claim_images", []) or []

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
        f"[RUN] {pos}/352 id={idx} "
        f"evidence={len(row.get('evidence', []))} "
        f"images={len(claim_imgs)}",
        flush=True
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

    # Save after EVERY claim so interruption is safe.
    atomic_save(rows)

    print(
        f"[DONE] {pos}/352 id={idx} "
        f"verdict={verdict} "
        f"just_chars={len(justification)}",
        flush=True
    )

atomic_save(rows)

print("\nFINAL:", OUT)
print("N:", len(rows))
print(
    "missing verdict:",
    sum(not r.get("verdict") for r in rows)
)
print(
    "missing justification:",
    sum(not r.get("justification") for r in rows)
)

from collections import Counter
print(
    "verdict distribution:",
    dict(Counter(r["verdict"] for r in rows))
)
