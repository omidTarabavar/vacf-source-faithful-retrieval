import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, obj):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    os.replace(tmp, path)


def bootstrap(a, b, n=10000, seed=808):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a

    rng = np.random.default_rng(seed)
    vals = np.empty(n)

    for i in range(n):
        ix = rng.integers(0, len(d), len(d))
        vals[i] = d[ix].mean()

    return {
        "delta": float(d.mean()),
        "lo": float(np.quantile(vals, 0.025)),
        "hi": float(np.quantile(vals, 0.975)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--repo",
        default="/workspace/AVerImaTec_Shared_Task"
    )
    ap.add_argument(
        "--frozen",
        default="/workspace/TEST352_STAGE8A/STAGE8A_FROZEN_EVIDENCE.json"
    )
    ap.add_argument(
        "--test",
        default="/workspace/AVERIMATEC_TEST_OFFICIAL/extracted/test_data/test.json"
    )
    ap.add_argument(
        "--out",
        default="/workspace/TEST352_STAGE8B_RESULTS"
    )
    ap.add_argument(
        "--cache",
        default="/workspace/FINAL_HF_CACHE"
    )
    ap.add_argument(
        "--model",
        default="google/gemma-3-27b-it"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(
        Path(args.frozen).read_text(encoding="utf-8")
    )

    test = json.loads(
        Path(args.test).read_text(encoding="utf-8")
    )

    cand_rows = frozen["methods"]["CANDIDATE_X"]
    xxp_rows = frozen["methods"]["XXP_FULL"]

    candidate = {x["claim_id"]: x for x in cand_rows}
    xxp = {x["claim_id"]: x for x in xxp_rows}

    ids = [x["claim_id"] for x in cand_rows]

    assert len(ids) == 352
    assert set(ids) == set(xxp)

    if args.limit > 0:
        ids = ids[:args.limit]

    # Critical: official modules infer paths from current working directory.
    prep = repo / "prepare_submission"
    os.chdir(prep)

    sys.path.insert(0, str(prep))
    sys.path.insert(1, str(repo))

    from ref_eval import val_evid_idv, compute_image_scores
    import utils

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    print("Loading evaluator:", args.model, flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache
    )

    processor = AutoProcessor.from_pretrained(
        args.model,
        cache_dir=args.cache
    )

    judge = {
        "model": model.eval(),
        "processor": processor,
    }

    print(
        "Evaluator loaded on:",
        next(model.parameters()).device,
        flush=True
    )

    # ---------------------------------------------------------
    # Gold evidence cache
    # ---------------------------------------------------------

    gt_path = out / "GT_EVID_TEST352.json"

    if gt_path.exists():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
    else:
        gt = {}

    for pos, cid in enumerate(ids, 1):

        if cid in gt:
            continue

        idx = int(cid.rsplit("_", 1)[1])
        row = test[idx]

        actual_claim = (
            row.get("claim_text")
            or row.get("claim")
            or ""
        ).strip()

        expected_claim = candidate[cid]["claim_text"].strip()

        if actual_claim != expected_claim:
            raise RuntimeError(
                f"TEST MAPPING FAILURE for {cid}\n"
                f"expected={expected_claim}\n"
                f"actual={actual_claim}"
            )

        converted = [
            utils.convert_qa_format(
                q,
                judge,
                "gemma",
                str(repo)
            )
            for q in row["questions"]
        ]

        gt[cid] = converted
        atomic_json(gt_path, gt)

        print(
            f"[GOLD] {pos}/{len(ids)} {cid} "
            f"questions={len(converted)}",
            flush=True
        )

    # ---------------------------------------------------------
    # Method evaluation
    # ---------------------------------------------------------

    result_path = out / "STAGE8B_RESULTS.json"

    if result_path.exists():
        results = json.loads(
            result_path.read_text(encoding="utf-8")
        )
    else:
        results = {
            "status": "RUNNING",
            "evaluator": args.model,
            "official_functions": [
                "val_evid_idv",
                "compute_image_scores",
                "get_auto_recall",
                "convert_qa_format"
            ],
            "methods": {
                "XXP_FULL": {},
                "CANDIDATE_X": {}
            }
        }

    methods = {
        "XXP_FULL": xxp,
        "CANDIDATE_X": candidate,
    }

    for method, rows in methods.items():

        print("\nMETHOD:", method, flush=True)

        for pos, cid in enumerate(ids, 1):

            if cid in results["methods"][method]:
                print(
                    f"[CACHE] {method} {pos}/{len(ids)} {cid}",
                    flush=True
                )
                continue

            pred_evid = rows[cid]["evidence"][:10]
            ref_evid = gt[cid]

            feedback, text_score = val_evid_idv(
                judge,
                "gemma",
                pred_evid,
                ref_evid,
                False,
                True
            )

            image_scores = compute_image_scores(
                judge,
                "gemma",
                pred_evid,
                ref_evid,
                text_score
            )

            precision, recall, f1 = utils.get_auto_recall(
                feedback,
                image_scores,
                len(ref_evid),
                len(pred_evid)
            )

            evidence_score = 0.0 if recall is None else float(recall)

            results["methods"][method][cid] = {
                "evidence_score": evidence_score,
                "precision": None if precision is None else float(precision),
                "f1": None if f1 is None else float(f1),
                "n_reference": len(ref_evid),
                "n_prediction": len(pred_evid),
                "text_score": text_score,
                "image_scores": image_scores,
                "feedback": feedback,
            }

            atomic_json(result_path, results)

            print(
                f"[EVAL] {method} "
                f"{pos}/{len(ids)} {cid} "
                f"EvidenceScore={evidence_score:.6f}",
                flush=True
            )

    # Do not finalize on smoke subsets.
    if len(ids) < 352:
        print(
            f"\nSMOKE_OK: completed {len(ids)} claim(s).",
            flush=True
        )
        return

    x = [
        results["methods"]["XXP_FULL"][cid]["evidence_score"]
        for cid in ids
    ]

    y = [
        results["methods"]["CANDIDATE_X"][cid]["evidence_score"]
        for cid in ids
    ]

    paired = bootstrap(x, y)

    summary = {
        "n": 352,
        "XXP_FULL_EVIDENCE_SCORE": float(np.mean(x)),
        "CANDIDATE_X_EVIDENCE_SCORE": float(np.mean(y)),
        "candidate_minus_xxp": paired,
        "wins": int(sum(b > a for a, b in zip(x, y))),
        "losses": int(sum(b < a for a, b in zip(x, y))),
        "ties": int(sum(b == a for a, b in zip(x, y))),
    }

    results["status"] = "STAGE8B_OFFICIAL_STYLE_EVIDENCE_OK"
    results["summary"] = summary

    atomic_json(result_path, results)

    md = f"""# Stage 8B — Matched AVerImaTeC Evidence Evaluation

Cohort: Test352 hidden evaluation
N: 352

Evaluator: `{args.model}`

Official AVerImaTeC evidence functions are used unchanged.

| Method | Evidence Score |
|---|---:|
| XxP Full | {summary['XXP_FULL_EVIDENCE_SCORE']:.6f} |
| Candidate-X | {summary['CANDIDATE_X_EVIDENCE_SCORE']:.6f} |

Candidate-X - XxP:
{paired['delta']:+.6f}

Paired bootstrap 95% CI:
[{paired['lo']:+.6f}, {paired['hi']:+.6f}]

Wins / Losses / Ties:
{summary['wins']} / {summary['losses']} / {summary['ties']}

IMPORTANT:
This is a matched official-style Evidence Score evaluation on the Test352 cohort.
"""

    (out / "STAGE8B_OFFICIAL_EVIDENCE.md").write_text(
        md,
        encoding="utf-8"
    )

    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
