import sys
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ROOT = "/workspace/AVerImaTec_Shared_Task"
PREP = f"{ROOT}/prepare_submission"

sys.path.insert(0, PREP)
sys.path.insert(0, ROOT)

from ref_eval import textual_val_single
import utils

VAL = Path(f"{ROOT}/data/data_clean/split_data/val.json")

PRED = {
    "BASELINE": Path("/workspace/VAL152_OFFICIAL_BASELINE_PREDICTIONS.json"),
    "XXP": Path("/workspace/VAL152_XXP_QWEN_FINAL.json"),
    "VACF": Path("/workspace/VAL152_VACF_QWEN_FINAL.json"),
}

OUT = Path("/workspace/VAL152_QJ_EVAL_RESULTS.json")

MODEL = (
    "/workspace/FINAL_HF_CACHE/hub/"
    "models--google--gemma-3-27b-it/"
    "snapshots/005ad3404e59d6023443cb575daa05336842228a"
)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

gold = json.loads(VAL.read_text(encoding="utf-8"))
pred = {
    k: json.loads(p.read_text(encoding="utf-8"))
    for k,p in PRED.items()
}

assert len(gold) == 152
for k in pred:
    assert len(pred[k]) == 152

# XxP and VACF questions MUST be identical.
assert all(
    pred["XXP"][i]["questions"] == pred["VACF"][i]["questions"]
    for i in range(152)
)

if OUT.exists():
    results = json.loads(OUT.read_text(encoding="utf-8"))
    print("RESUME:", OUT, flush=True)
else:
    results = {
        "status": "RUNNING",
        "model": MODEL,
        "question": {
            "BASELINE": {},
            "XXP_SHARED_WITH_VACF": {}
        },
        "justification": {
            "BASELINE": {},
            "XXP": {},
            "VACF": {}
        }
    }

def save():
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    os.replace(tmp, OUT)

print("Loading evaluator:", MODEL, flush=True)

processor = AutoProcessor.from_pretrained(
    MODEL,
    local_files_only=True,
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).eval()

judge = {
    "model": model,
    "processor": processor,
}

print("Evaluator loaded on:", model.device, flush=True)

# ---------------------------------------------------------
# QUESTION SCORE
# Baseline separately; XxP/VACF once because identical.
# ---------------------------------------------------------

question_jobs = {
    "BASELINE": pred["BASELINE"],
    "XXP_SHARED_WITH_VACF": pred["XXP"],
}

for method, rows in question_jobs.items():

    print("\nQUESTION METHOD:", method, flush=True)

    for i in range(152):

        cid = f"averi_val_{i:04d}"

        if cid in results["question"][method]:
            print(f"[CACHE-Q] {method} {i+1}/152 {cid}", flush=True)
            continue

        gt_questions = [
            q["question"]
            for q in gold[i]["questions"]
        ]

        pred_questions = rows[i]["questions"]

        # Claim-specific deterministic seed.
        random.seed(500000 + i)
        torch.manual_seed(500000 + i)
        torch.cuda.manual_seed_all(500000 + i)

        feedback, raw_score = textual_val_single(
            gt_questions,
            pred_questions,
            ROOT,
            "gemma",
            judge,
            "question",
            False,
        )

        score = utils.ques_recall_compute(
            raw_score,
            len(gt_questions),
            len(pred_questions),
        )

        results["question"][method][cid] = {
            "score": float(score),
            "raw_score": raw_score,
            "feedback": feedback,
        }

        save()

        print(
            f"[Q] {method} {i+1}/152 {cid} "
            f"score={float(score):.6f}",
            flush=True,
        )

# ---------------------------------------------------------
# JUSTIFICATION SCORE
# ---------------------------------------------------------

for method in ["BASELINE", "XXP", "VACF"]:

    rows = pred[method]

    print("\nJUSTIFICATION METHOD:", method, flush=True)

    for i in range(152):

        cid = f"averi_val_{i:04d}"

        if cid in results["justification"][method]:
            print(f"[CACHE-J] {method} {i+1}/152 {cid}", flush=True)
            continue

        gt_just = gold[i]["justification"]
        pred_just = rows[i]["justification"]

        # Same seed per claim across methods.
        random.seed(600000 + i)
        torch.manual_seed(600000 + i)
        torch.cuda.manual_seed_all(600000 + i)

        feedback, raw_score = textual_val_single(
            gt_just,
            pred_just,
            ROOT,
            "gemma",
            judge,
            "justification",
            False,
        )

        score = utils.justi_recall_compute(
            feedback,
            raw_score,
        )

        results["justification"][method][cid] = {
            "score": float(score),
            "raw_score": raw_score,
            "feedback": feedback,
        }

        save()

        print(
            f"[J] {method} {i+1}/152 {cid} "
            f"score={float(score):.6f}",
            flush=True,
        )

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def mean(section, method):
    vals = [
        v["score"]
        for v in results[section][method].values()
    ]
    assert len(vals) == 152
    return sum(vals) / len(vals)

q_baseline = mean("question", "BASELINE")
q_xxp = mean("question", "XXP_SHARED_WITH_VACF")

j_baseline = mean("justification", "BASELINE")
j_xxp = mean("justification", "XXP")
j_vacf = mean("justification", "VACF")

results["summary"] = {
    "question": {
        "BASELINE": q_baseline,
        "XXP": q_xxp,
        "VACF": q_xxp,
    },
    "justification": {
        "BASELINE": j_baseline,
        "XXP": j_xxp,
        "VACF": j_vacf,
    }
}

results["status"] = "COMPLETE"
save()

print("\n================ FINAL ================")
print("Question BASELINE:", q_baseline)
print("Question XxP     :", q_xxp)
print("Question VACF    :", q_xxp)
print("Justif BASELINE  :", j_baseline)
print("Justif XxP       :", j_xxp)
print("Justif VACF      :", j_vacf)
print("OUTPUT:", OUT)
