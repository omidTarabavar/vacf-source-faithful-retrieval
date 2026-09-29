import argparse
import gc
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL = "Qwen/Qwen3-Reranker-4B"

SEM_FILE = Path(
    "/workspace/STAGE10B_RESULTS/"
    "CAL64_QWEN_SEMANTIC.json"
)

OUTDIR = Path("/workspace/STAGE10C_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTDIR / "checkpoint.json"
FINAL = OUTDIR / "CAL64_QWEN_RERANKER.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

TOPN = 100

TASK = (
    "Given a fact-checking claim, determine whether the document "
    "contains evidence useful for verifying or refuting the claim. "
    "Relevant evidence may identify the depicted person, object, "
    "event, source, date, location, context, correction, or factual "
    "relationship."
)


def canon(u):
    u = (u or "").strip()

    if not u:
        return ""

    try:
        p = urlsplit(u)
    except Exception:
        return u.lower().rstrip("/")

    host = p.netloc.lower()

    if host.startswith("www."):
        host = host[4:]

    path = re.sub(
        r"/{2,}",
        "/",
        p.path or "/"
    )

    if path != "/":
        path = path.rstrip("/")

    q = urlencode(sorted(
        (k, v)
        for k, v in parse_qsl(
            p.query,
            keep_blank_values=True
        )
        if not k.lower().startswith("utm_")
    ))

    key = host + path

    if q:
        key += "?" + q

    return key


PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. "
    'The answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


def format_pair(query, document):

    body = (
        f"<Instruct>: {TASK}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )

    return PREFIX + body + SUFFIX


@torch.inference_mode()
def score_pairs(
    model,
    tokenizer,
    true_id,
    false_id,
    query,
    candidates,
    batch_size,
    max_length
):
    scores = []

    for start in range(
        0,
        len(candidates),
        batch_size
    ):

        batch_rows = candidates[
            start:start + batch_size
        ]

        texts = []

        for row in batch_rows:

            document = (
                f"URL: {row['url']}\n"
                f"Passage: {row['best_passage']}"
            )

            texts.append(
                format_pair(
                    query,
                    document
                )
            )

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        logits = model(
            **inputs
        ).logits[:, -1, :]

        yes = logits[:, true_id]
        no = logits[:, false_id]

        pair = torch.stack(
            [no, yes],
            dim=1
        )

        probs = F.softmax(
            pair.float(),
            dim=1
        )[:, 1]

        scores.extend(
            probs.cpu().tolist()
        )

        del inputs
        del logits
        del yes
        del no
        del pair
        del probs

    return scores


def save_checkpoint(results):

    CHECKPOINT.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "model": MODEL,
                "topn": TOPN,
                "results": results
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1536
    )

    args = parser.parse_args()


    src = json.loads(
        SEM_FILE.read_text(
            encoding="utf-8"
        )
    )

    sem = src["results"]

    assert len(sem) == 64


    print("Claims =", len(sem))
    print("TOPN =", TOPN)
    print("Batch size =", args.batch_size)
    print("Max length =", args.max_length)


    print("Loading tokenizer:", MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        padding_side="left"
    )


    print("Loading model:", MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).cuda().eval()


    print("MODEL_DEVICE =", model.device)


    FALSE_ID = tokenizer.convert_tokens_to_ids(
        "no"
    )

    TRUE_ID = tokenizer.convert_tokens_to_ids(
        "yes"
    )


    print("FALSE_ID =", FALSE_ID)
    print("TRUE_ID =", TRUE_ID)


    if CHECKPOINT.exists():

        saved = json.loads(
            CHECKPOINT.read_text(
                encoding="utf-8"
            )
        )

        results = saved.get(
            "results",
            {}
        )

        print(
            "Checkpoint:",
            len(results),
            "/64"
        )

    else:
        results = {}


    for n, (cid, row) in enumerate(
        sem.items(),
        1
    ):

        if cid in results:

            print(
                f"[CACHE] {n}/64 {cid}",
                flush=True
            )

            continue


        candidates = row[
            "top250_qwen"
        ][:TOPN]


        if not candidates:
            raise RuntimeError(
                f"No candidates for {cid}"
            )


        print(
            f"[CLAIM] {n}/64 "
            f"{cid} "
            f"candidates={len(candidates)}",
            flush=True
        )


        rr_scores = score_pairs(
            model=model,
            tokenizer=tokenizer,
            true_id=TRUE_ID,
            false_id=FALSE_ID,
            query=row["claim_text"],
            candidates=candidates,
            batch_size=args.batch_size,
            max_length=args.max_length
        )


        scored = []


        for sem_rank, (
            candidate,
            rr_score
        ) in enumerate(
            zip(
                candidates,
                rr_scores
            ),
            1
        ):

            scored.append({
                "url":
                    candidate["url"],

                "best_passage":
                    candidate["best_passage"],

                "semantic_rank":
                    sem_rank,

                "semantic_score":
                    candidate["score"],

                "reranker_score":
                    float(rr_score)
            })


        reranked = sorted(
            scored,
            key=lambda x:
                x["reranker_score"],
            reverse=True
        )


        final_rows = []

        for rank, x in enumerate(
            reranked,
            1
        ):

            final_rows.append({
                "rank":
                    rank,

                **x
            })


        results[cid] = {
            "claim_id":
                cid,

            "claim_text":
                row["claim_text"],

            "num_candidates":
                len(candidates),

            "reranked":
                final_rows
        }


        print(
            f"[DONE] {cid} "
            f"top1={final_rows[0]['url'][:100]} "
            f"score={final_rows[0]['reranker_score']:.6f}",
            flush=True
        )


        save_checkpoint(
            results
        )


        gc.collect()
        torch.cuda.empty_cache()


    assert len(results) == 64


    ordered = {
        cid: results[cid]
        for cid in sem
    }


    sizes = [
        row["num_candidates"]
        for row in ordered.values()
    ]


    FINAL.write_text(
        json.dumps(
            {
                "status":
                    "STAGE10C_COMPLETE",

                "cohort":
                    "Cal64",

                "n":
                    64,

                "qrels_used":
                    False,

                "model":
                    MODEL,

                "topn":
                    TOPN,

                "results":
                    ordered
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


    lines = [
        "=== STAGE 10C CAL64 QWEN RERANKER ===",
        "status = COMPLETE",
        "claims = 64",
        "qrels_used = False",
        f"model = {MODEL}",
        f"TopN = {TOPN}",
        "",
        f"candidates min = {min(sizes)}",
        f"candidates max = {max(sizes)}",
        f"pairs total = {sum(sizes)}",
        "",
        "STAGE10C_CAL64_RERANKER_OK"
    ]


    summary = "\n".join(
        lines
    )


    SUMMARY.write_text(
        summary + "\n",
        encoding="utf-8"
    )


    CHECKPOINT.write_text(
        json.dumps(
            {
                "status":
                    "COMPLETE",

                "model":
                    MODEL,

                "topn":
                    TOPN,

                "results":
                    ordered
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


    print()
    print(summary)
    print()
    print("FINAL =", FINAL)
    print("SUMMARY =", SUMMARY)


if __name__ == "__main__":
    main()
