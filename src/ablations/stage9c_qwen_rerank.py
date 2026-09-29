import json
import re
import gc
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL = "Qwen/Qwen3-Embedding-4B"

BM25_FILE = Path(
    "/workspace/STAGE9B_RESULTS/BM25_PREFILTER_RESULTS.json"
)

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

OUTDIR = Path("/workspace/STAGE9C_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTDIR / "checkpoint.json"
FINAL = OUTDIR / "STAGE9C_QWEN_RERANK.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

TOP_URLS = 250
PER_URL = 16
WINDOW = 448
OVERLAP = 64
BATCH = 64
MAX_LENGTH = 512

TASK = (
    "Given a fact-checking claim, retrieve web evidence that helps "
    "verify or refute it. Prefer evidence identifying the relevant "
    "person, object, event, source, date, location, factual context, "
    "correction, or relationship."
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

    path = re.sub(r"/{2,}", "/", p.path or "/")

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


def words(text):
    return re.findall(
        r"[a-z0-9]+",
        (text or "").lower()
    )


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def last_token_pool(hidden, mask):
    left_padding = (
        mask[:, -1].sum()
        == mask.shape[0]
    )

    if left_padding:
        return hidden[:, -1]

    seq = mask.sum(dim=1) - 1

    return hidden[
        torch.arange(
            hidden.shape[0],
            device=hidden.device
        ),
        seq
    ]


@torch.inference_mode()
def encode(
    model,
    tokenizer,
    texts,
    is_query=False
):
    all_emb = []

    for start in range(
        0,
        len(texts),
        BATCH
    ):

        batch = texts[
            start:start+BATCH
        ]

        if is_query:
            batch = [
                f"Instruct: {TASK}\nQuery: {x}"
                for x in batch
            ]

        inp = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inp = {
            k: v.to(model.device)
            for k, v in inp.items()
        }

        out = model(**inp)

        emb = last_token_pool(
            out.last_hidden_state,
            inp["attention_mask"]
        )

        emb = F.normalize(
            emb.float(),
            p=2,
            dim=1
        )

        all_emb.append(
            emb.cpu()
        )

    return torch.cat(
        all_emb,
        dim=0
    ).numpy()


def chunk_text(tokenizer, text):
    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not ids:
        return []

    if len(ids) <= WINDOW:
        return [text.strip()]

    chunks = []

    step = WINDOW - OVERLAP

    for start in range(
        0,
        len(ids),
        step
    ):

        part = ids[
            start:start+WINDOW
        ]

        if not part:
            break

        x = tokenizer.decode(
            part,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        if x:
            chunks.append(x)

        if start + WINDOW >= len(ids):
            break

    return chunks


def cheap_chunk_score(
    text,
    query_terms
):
    toks = words(text)

    if not toks:
        return 0.0

    tset = set(toks)

    score = 0.0

    for term in query_terms:

        if term in tset:
            score += 2.0

            score += min(
                toks.count(term),
                3
            ) * 0.25

    useful = [
        t for t in query_terms
        if len(t) >= 4
    ]

    low = text.lower()

    for a, b in zip(
        useful,
        useful[1:]
    ):
        if f"{a} {b}" in low:
            score += 1.0

    return score


def select_chunks_for_url(
    tokenizer,
    raw_pieces,
    query_terms
):
    chunks = []

    for piece in raw_pieces:
        chunks.extend(
            chunk_text(
                tokenizer,
                piece
            )
        )

    if not chunks:
        return []

    scored = [
        (
            cheap_chunk_score(
                x,
                query_terms
            ),
            i,
            x
        )
        for i, x in enumerate(chunks)
    ]

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    chosen = []
    seen = set()

    # Top lexical chunks.
    for _, i, x in scored[:10]:
        if x not in seen:
            chosen.append((i, x))
            seen.add(x)

    # Preserve title / lead context.
    for i in range(
        min(3, len(chunks))
    ):
        x = chunks[i]

        if x not in seen:
            chosen.append((i, x))
            seen.add(x)

    # Add distributional coverage through long pages.
    if len(chunks) > 3:

        extra_indices = [
            len(chunks) // 3,
            (2 * len(chunks)) // 3,
            len(chunks) - 1
        ]

        for i in extra_indices:

            x = chunks[i]

            if x not in seen:
                chosen.append((i, x))
                seen.add(x)

    chosen = chosen[:PER_URL]

    return [
        x for _, x in chosen
    ]


bm25 = json.loads(
    BM25_FILE.read_text(
        encoding="utf-8"
    )
)

print("Claims =", len(bm25))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    padding_side="left"
)

print("Loading Qwen3-Embedding-4B...")

model = AutoModel.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).cuda().eval()

print("MODEL_DEVICE =", model.device)


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
        "/",
        len(bm25)
    )
else:
    results = {}


for n, (cid, row) in enumerate(
    bm25.items(),
    1
):

    if cid in results:
        print(
            f"[CACHE] {n}/{len(bm25)} {cid}"
        )
        continue

    candidate_rows = (
        row["top10000"][:TOP_URLS]
    )

    candidate_urls = {
        canon(x["url"]): x["url"]
        for x in candidate_rows
    }

    query_terms = row[
        "query_terms"
    ]

    idx = str(
        int(cid.rsplit("_", 1)[1])
    )

    store_path = (
        STORE / f"{idx}.json"
    )

    raw_by_url = defaultdict(list)

    for x in read_jsonl(
        store_path
    ):

        cu = canon(
            x.get("url", "")
        )

        if cu not in candidate_urls:
            continue

        vals = x.get(
            "url2text",
            []
        )

        if isinstance(vals, str):
            vals = [vals]

        for text in vals:
            text = str(text).strip()

            if text:
                raw_by_url[cu].append(
                    text
                )


    passages = []
    passage_urls = []

    for cu in candidate_urls:

        selected = select_chunks_for_url(
            tokenizer,
            raw_by_url.get(
                cu,
                []
            ),
            query_terms
        )

        for passage in selected:

            passages.append(
                passage
            )

            passage_urls.append(
                cu
            )


    print(
        f"[CLAIM] {n}/{len(bm25)} "
        f"{cid} "
        f"urls={len(candidate_urls)} "
        f"qwen_passages={len(passages)}",
        flush=True
    )

    q = encode(
        model,
        tokenizer,
        [row["claim"]],
        is_query=True
    )[0]

    emb = encode(
        model,
        tokenizer,
        passages,
        is_query=False
    )

    scores = emb @ q

    best_by_url = {}

    best_passage = {}

    for cu, passage, score in zip(
        passage_urls,
        passages,
        scores
    ):

        score = float(score)

        if (
            cu not in best_by_url
            or score > best_by_url[cu]
        ):
            best_by_url[cu] = score
            best_passage[cu] = passage


    ranked = sorted(
        best_by_url.items(),
        key=lambda x: x[1],
        reverse=True
    )

    rank_map = {
        cu: rank
        for rank, (cu, _)
        in enumerate(
            ranked,
            1
        )
    }

    teacher_urls = row[
        "teacher_urls"
    ]

    teacher_ranks = [
        rank_map.get(
            canon(url)
        )
        for url in teacher_urls
    ]


    print(
        f"[DONE] {cid} "
        f"teacher_ranks={teacher_ranks}",
        flush=True
    )


    results[cid] = {
        "claim":
            row["claim"],

        "teacher_urls":
            teacher_urls,

        "bm25_teacher_ranks":
            row["teacher_ranks"],

        "qwen_teacher_ranks":
            teacher_ranks,

        "num_candidate_urls":
            len(candidate_urls),

        "num_qwen_passages":
            len(passages),

        "top250_qwen": [
            {
                "rank": rank,
                "url":
                    candidate_urls[cu],
                "score":
                    score,
                "best_passage":
                    best_passage[cu]
            }

            for rank, (cu, score)
            in enumerate(
                ranked,
                1
            )
        ]
    }


    CHECKPOINT.write_text(
        json.dumps(
            {
                "status":
                    "RUNNING",

                "top_urls":
                    TOP_URLS,

                "per_url":
                    PER_URL,

                "results":
                    results
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    del emb
    del scores

    gc.collect()

    torch.cuda.empty_cache()


all_ranks = [
    r
    for row in results.values()
    for r in row[
        "qwen_teacher_ranks"
    ]
]

den = len(all_ranks)

lines = [
    "=== STAGE 9C QWEN SEMANTIC RERANK ===",
    f"teacher_claims = {len(results)}",
    f"teacher_evidences = {den}",
    f"BM25_TOP_URLS = {TOP_URLS}",
    f"MAX_PASSAGES_PER_URL = {PER_URL}",
]

for k in [
    1,
    3,
    5,
    10,
    20,
    30,
    50,
    100,
    250
]:

    found = sum(
        r is not None
        and r <= k
        for r in all_ranks
    )

    lines.append(
        f"Recall@{k} = "
        f"{found}/{den} = "
        f"{found/den:.6f}"
    )


# Compare the one claim for which exact brute-force Qwen exists.
if "averi_train_0668" in results:

    lines += [
        "",
        "Bruteforce reference for averi_train_0668 = [2, 3]",
        "Optimized ranks for averi_train_0668 = "
        + str(
            results[
                "averi_train_0668"
            ]["qwen_teacher_ranks"]
        )
    ]


summary = "\n".join(lines)

SUMMARY.write_text(
    summary + "\n",
    encoding="utf-8"
)

FINAL.write_text(
    json.dumps(
        {
            "status":
                "STAGE9C_COMPLETE",

            "model":
                MODEL,

            "top_urls":
                TOP_URLS,

            "max_passages_per_url":
                PER_URL,

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
print("STAGE9C_COMPLETE")
print("SUMMARY =", SUMMARY)
print("FINAL =", FINAL)
