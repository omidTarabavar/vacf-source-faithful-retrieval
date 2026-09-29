import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL = "Qwen/Qwen3-Embedding-4B"

BM25_FILE = Path(
    "/workspace/STAGE10A_RESULTS/CAL64_BM25_TOP250.json"
)

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

OUTDIR = Path("/workspace/STAGE10B_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTDIR / "checkpoint.json"
FINAL = OUTDIR / "CAL64_QWEN_SEMANTIC.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

TOP_URLS = 250
PER_URL = 16
WINDOW = 448
OVERLAP = 64

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
    batch_size,
    max_length,
    is_query=False
):
    all_emb = []

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
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
            max_length=max_length,
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

        del inp
        del out
        del emb

    return torch.cat(
        all_emb,
        dim=0
    )


def chunk_text(
    tokenizer,
    text
):
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
            start:start + WINDOW
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

    # Same recipe as Stage 9C:
    # top lexical windows.
    for _, i, x in scored[:10]:

        if x not in seen:
            chosen.append((i, x))
            seen.add(x)

    # title / lead context.
    for i in range(
        min(3, len(chunks))
    ):

        x = chunks[i]

        if x not in seen:
            chosen.append((i, x))
            seen.add(x)

    # long-document coverage.
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


def save_checkpoint(results):
    CHECKPOINT.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "model": MODEL,
                "top_urls": TOP_URLS,
                "max_passages_per_url": PER_URL,
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
        default=64
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512
    )

    args = parser.parse_args()


    src = json.loads(
        BM25_FILE.read_text(
            encoding="utf-8"
        )
    )

    bm25 = src["results"]

    assert len(bm25) == 64


    print("Claims =", len(bm25))
    print("Batch size =", args.batch_size)
    print("Max length =", args.max_length)


    print(
        "Loading tokenizer:",
        MODEL
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        padding_side="left"
    )


    print(
        "Loading model:",
        MODEL
    )

    model = AutoModel.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).cuda().eval()


    print(
        "MODEL_DEVICE =",
        model.device
    )


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


    total_passages = 0


    for n, (cid, row) in enumerate(
        bm25.items(),
        1
    ):

        if cid in results:

            print(
                f"[CACHE] {n}/64 {cid}",
                flush=True
            )

            continue


        candidate_rows = (
            row["top250"][:TOP_URLS]
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


        if not passages:
            raise RuntimeError(
                f"No passages for {cid}"
            )


        print(
            f"[CLAIM] {n}/64 "
            f"{cid} "
            f"urls={len(candidate_urls)} "
            f"qwen_passages={len(passages)}",
            flush=True
        )


        q = encode(
            model,
            tokenizer,
            [row["claim_text"]],
            args.batch_size,
            args.max_length,
            is_query=True
        )[0]


        emb = encode(
            model,
            tokenizer,
            passages,
            args.batch_size,
            args.max_length,
            is_query=False
        )


        scores = (
            emb
            @ q.unsqueeze(1)
        ).squeeze(1)


        best_by_url = {}
        best_passage = {}


        for cu, passage, score in zip(
            passage_urls,
            passages,
            scores.tolist()
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


        top_qwen = []

        for rank, (
            cu,
            score
        ) in enumerate(
            ranked,
            1
        ):

            top_qwen.append({
                "rank": rank,
                "url":
                    candidate_urls[cu],
                "score":
                    float(score),
                "best_passage":
                    best_passage[cu]
            })


        results[cid] = {
            "claim_id":
                cid,

            "claim_text":
                row["claim_text"],

            "num_candidate_urls":
                len(candidate_urls),

            "num_qwen_passages":
                len(passages),

            "top250_qwen":
                top_qwen
        }


        total_passages += len(
            passages
        )


        print(
            f"[DONE] {cid} "
            f"top1={top_qwen[0]['url'][:100]}",
            flush=True
        )


        save_checkpoint(
            results
        )


        del q
        del emb
        del scores

        gc.collect()
        torch.cuda.empty_cache()


    assert len(results) == 64


    # Preserve Stage10A / Cal64 order.
    ordered = {
        cid: results[cid]
        for cid in bm25
    }


    num_candidates = [
        row["num_candidate_urls"]
        for row in ordered.values()
    ]

    num_passages = [
        row["num_qwen_passages"]
        for row in ordered.values()
    ]


    def median(xs):
        ys = sorted(xs)
        n = len(ys)

        if n % 2:
            return ys[n // 2]

        return (
            ys[n//2 - 1]
            + ys[n//2]
        ) / 2


    FINAL.write_text(
        json.dumps(
            {
                "status":
                    "STAGE10B_COMPLETE",

                "cohort":
                    "Cal64",

                "n":
                    64,

                "qrels_used":
                    False,

                "model":
                    MODEL,

                "top_urls":
                    TOP_URLS,

                "max_passages_per_url":
                    PER_URL,

                "results":
                    ordered
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


    lines = [
        "=== STAGE 10B CAL64 QWEN SEMANTIC ===",
        "status = COMPLETE",
        "claims = 64",
        "qrels_used = False",
        f"model = {MODEL}",
        f"Top URLs = {TOP_URLS}",
        f"Max passages per URL = {PER_URL}",
        "",
        f"candidate URLs min = {min(num_candidates)}",
        f"candidate URLs median = {median(num_candidates)}",
        f"candidate URLs max = {max(num_candidates)}",
        "",
        f"Qwen passages min = {min(num_passages)}",
        f"Qwen passages median = {median(num_passages)}",
        f"Qwen passages max = {max(num_passages)}",
        f"Qwen passages total = {sum(num_passages)}",
        "",
        "STAGE10B_CAL64_QWEN_OK"
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

                "top_urls":
                    TOP_URLS,

                "max_passages_per_url":
                    PER_URL,

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
    print(
        "FINAL =",
        FINAL
    )
    print(
        "SUMMARY =",
        SUMMARY
    )


if __name__ == "__main__":
    main()
