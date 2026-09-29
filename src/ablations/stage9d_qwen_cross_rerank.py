import json
import math
import re
import gc
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL = "Qwen/Qwen3-Reranker-4B"

SRC = Path(
    "/workspace/STAGE9C_RESULTS/"
    "STAGE9C_QWEN_RERANK.json"
)

OUTDIR = Path(
    "/workspace/STAGE9D_RESULTS"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT = OUTDIR / "checkpoint.json"
FINAL = OUTDIR / "STAGE9D_RERANK.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

TOPN = 100
BATCH = 16
MAXLEN = 1536

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


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    padding_side="left"
)

print("Loading:", MODEL)

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

    return (
        PREFIX
        + body
        + SUFFIX
    )


@torch.inference_mode()
def score_pairs(
    query,
    candidates
):
    scores = []

    for start in range(
        0,
        len(candidates),
        BATCH
    ):

        batch_rows = candidates[
            start:start+BATCH
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
            max_length=MAXLEN,
            return_tensors="pt"
        )

        inputs = {
            k: v.cuda()
            for k, v in inputs.items()
        }

        logits = model(
            **inputs
        ).logits[:, -1, :]

        yes = logits[:, TRUE_ID]
        no = logits[:, FALSE_ID]

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
        del pair
        del probs

    return scores


src = json.loads(
    SRC.read_text(
        encoding="utf-8"
    )
)

src_results = src["results"]


if CHECKPOINT.exists():

    old = json.loads(
        CHECKPOINT.read_text(
            encoding="utf-8"
        )
    )

    results = old.get(
        "results",
        {}
    )

    print(
        "Checkpoint:",
        len(results),
        "/",
        len(src_results)
    )

else:
    results = {}


for n, (cid, row) in enumerate(
    src_results.items(),
    1
):

    if cid in results:

        print(
            f"[CACHE] "
            f"{n}/{len(src_results)} "
            f"{cid}"
        )

        continue


    candidates = row[
        "top250_qwen"
    ][:TOPN]


    print(
        f"[CLAIM] "
        f"{n}/{len(src_results)} "
        f"{cid} "
        f"candidates={len(candidates)}",
        flush=True
    )


    rr_scores = score_pairs(
        row["claim"],
        candidates
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


    rerank_position = {
        canon(x["url"]):
            rank

        for rank, x
        in enumerate(
            reranked,
            1
        )
    }


    teacher_urls = row[
        "teacher_urls"
    ]


    pure_reranker_ranks = [
        rerank_position.get(
            canon(url)
        )
        for url in teacher_urls
    ]


    # -------------------------------------------------
    # Rank fusion benchmark
    # -------------------------------------------------

    fusion_results = {}

    K = 20

    for lam in [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0
    ]:

        fused = []

        for x in scored:

            sem_rank = x[
                "semantic_rank"
            ]

            rr_rank = (
                rerank_position[
                    canon(x["url"])
                ]
            )

            fusion_score = (
                1.0
                / (K + sem_rank)
                +
                lam
                / (K + rr_rank)
            )

            fused.append({
                **x,
                "reranker_rank":
                    rr_rank,
                "fusion_score":
                    fusion_score
            })


        fused.sort(
            key=lambda x:
                x["fusion_score"],
            reverse=True
        )


        fmap = {
            canon(x["url"]):
                rank

            for rank, x
            in enumerate(
                fused,
                1
            )
        }


        ranks = [
            fmap.get(
                canon(url)
            )
            for url in teacher_urls
        ]


        fusion_results[
            str(lam)
        ] = {
            "teacher_ranks":
                ranks,

            "ranking":
                fused
        }


    print(
        f"[DONE] {cid} "
        f"semantic="
        f"{row['qwen_teacher_ranks']} "
        f"reranker="
        f"{pure_reranker_ranks}",
        flush=True
    )


    results[cid] = {
        "claim":
            row["claim"],

        "teacher_urls":
            teacher_urls,

        "semantic_teacher_ranks":
            row["qwen_teacher_ranks"],

        "reranker_teacher_ranks":
            pure_reranker_ranks,

        "reranked":
            reranked,

        "fusion":
            fusion_results
    }


    CHECKPOINT.write_text(
        json.dumps(
            {
                "status":
                    "RUNNING",
                "results":
                    results
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


    gc.collect()
    torch.cuda.empty_cache()


def collect_ranks(mode):
    ranks = []

    for row in results.values():

        if mode == "semantic":

            ranks.extend(
                row[
                    "semantic_teacher_ranks"
                ]
            )

        elif mode == "reranker":

            ranks.extend(
                row[
                    "reranker_teacher_ranks"
                ]
            )

        else:

            ranks.extend(
                row["fusion"][
                    mode
                ]["teacher_ranks"]
            )

    return ranks


def stats(ranks):
    out = {}

    for k in [
        1,
        3,
        5,
        10,
        20,
        30,
        50,
        100
    ]:

        n = sum(
            r is not None
            and r <= k
            for r in ranks
        )

        out[k] = n

    mrr = sum(
        0 if r is None else 1/r
        for r in ranks
    ) / len(ranks)

    return out, mrr


modes = [
    "semantic",
    "reranker",
    "0.25",
    "0.5",
    "1.0",
    "2.0",
    "4.0"
]


lines = [
    "=== STAGE 9D CROSS-ENCODER RERANK ===",
    f"teacher_claims = {len(results)}",
    "",
]


best_mode = None
best_key = None


for mode in modes:

    ranks = collect_ranks(
        mode
    )

    st, mrr = stats(
        ranks
    )

    lines.append(
        f"[{mode}]"
    )

    lines.append(
        f"Recall@5  = "
        f"{st[5]}/{len(ranks)}"
    )

    lines.append(
        f"Recall@10 = "
        f"{st[10]}/{len(ranks)}"
    )

    lines.append(
        f"Recall@20 = "
        f"{st[20]}/{len(ranks)}"
    )

    lines.append(
        f"Recall@50 = "
        f"{st[50]}/{len(ranks)}"
    )

    lines.append(
        f"MRR = {mrr:.6f}"
    )

    lines.append("")


    key = (
        st[10],
        st[5],
        st[20],
        mrr
    )

    if (
        best_key is None
        or key > best_key
    ):
        best_key = key
        best_mode = mode


lines.append(
    f"BEST_MODE = {best_mode}"
)


# Specific exact-Qwen sanity claim.
if "averi_train_0668" in results:

    r = results[
        "averi_train_0668"
    ]

    lines += [
        "",
        "0668 brute-force reference = [2, 3]",
        "0668 semantic optimized = "
        + str(
            r[
                "semantic_teacher_ranks"
            ]
        ),
        "0668 reranker = "
        + str(
            r[
                "reranker_teacher_ranks"
            ]
        )
    ]


summary = "\n".join(
    lines
)

SUMMARY.write_text(
    summary + "\n",
    encoding="utf-8"
)


FINAL.write_text(
    json.dumps(
        {
            "status":
                "STAGE9D_COMPLETE",

            "model":
                MODEL,

            "topn":
                TOPN,

            "best_mode":
                best_mode,

            "results":
                results
        },
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


print()
print(summary)
print()
print("STAGE9D_COMPLETE")
print("FINAL =", FINAL)
print("SUMMARY =", SUMMARY)
