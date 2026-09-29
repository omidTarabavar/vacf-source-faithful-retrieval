import argparse
import gc
import json
import re
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlsplit, parse_qsl, urlencode

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


MODEL = "Qwen/Qwen3-Embedding-4B"

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

AUDIT = Path(
    "/workspace/STAGE9_XXP_TEACHER_AUDIT.md"
)

FROZEN = Path(
    "/workspace/STAGE8C_CANDIDATE_Y_64.json"
)

OUTDIR = Path(
    "/workspace/STAGE9A_RESULTS"
)

CHECKPOINT = OUTDIR / "checkpoint.json"
FINAL = OUTDIR / "STAGE9A_QWEN_TEXT_RESCUE.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

TASK = (
    "Given a fact-checking claim, retrieve web evidence passages "
    "that help verify or refute the claim. Prefer passages that "
    "identify the relevant person, object, event, source, date, "
    "location, context, correction, or factual relationship."
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

    query = urlencode(sorted(
        (k, v)
        for k, v in parse_qsl(
            p.query,
            keep_blank_values=True
        )
        if not k.lower().startswith("utm_")
    ))

    key = host + path

    if query:
        key += "?" + query

    return key


def parse_teachers(text):
    cur = None
    teachers = defaultdict(list)

    for line in text.splitlines():

        m = re.match(
            r"^###\s+(averi_train_\d+)",
            line
        )

        if m:
            cur = m.group(1)
            continue

        m = re.match(
            r"^\s*-\s+URL:\s+(\S+)",
            line
        )

        if m and cur:
            teachers[cur].append(
                m.group(1)
            )

    return dict(teachers)


def read_jsonl(path):
    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                yield json.loads(line)


def atomic_json(obj, path):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    tmp.replace(path)


def last_token_pool(
    last_hidden_states,
    attention_mask
):
    left_padding = (
        attention_mask[:, -1].sum()
        == attention_mask.shape[0]
    )

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = (
        attention_mask.sum(dim=1) - 1
    )

    batch_size = (
        last_hidden_states.shape[0]
    )

    return last_hidden_states[
        torch.arange(
            batch_size,
            device=last_hidden_states.device
        ),
        sequence_lengths
    ]


@torch.inference_mode()
def encode_texts(
    model,
    tokenizer,
    texts,
    batch_size,
    max_length,
    is_query=False
):
    all_embeddings = []

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
                f"Instruct: {TASK}\n"
                f"Query: {x}"
                for x in batch
            ]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        outputs = model(**inputs)

        embeddings = last_token_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )

        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=1
        )

        all_embeddings.append(
            embeddings.cpu()
        )

        del inputs
        del outputs
        del embeddings

    return torch.cat(
        all_embeddings,
        dim=0
    ).numpy()


def chunk_piece(
    tokenizer,
    text,
    window=448,
    overlap=64
):
    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not ids:
        return []

    if len(ids) <= window:
        return [text.strip()]

    step = window - overlap

    chunks = []

    for start in range(
        0,
        len(ids),
        step
    ):

        part = ids[
            start:start + window
        ]

        if not part:
            break

        text_part = tokenizer.decode(
            part,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        if text_part:
            chunks.append(
                text_part
            )

        if start + window >= len(ids):
            break

    return chunks


def build_passages(
    store_path,
    tokenizer
):
    passages = []
    urls = []

    for row in read_jsonl(
        store_path
    ):

        url = row.get(
            "url",
            ""
        )

        values = row.get(
            "url2text",
            []
        )

        if isinstance(
            values,
            str
        ):
            values = [values]

        for value in values:

            value = str(
                value
            ).strip()

            if not value:
                continue

            for chunk in chunk_piece(
                tokenizer,
                value
            ):

                passages.append(
                    chunk
                )

                urls.append(
                    url
                )

    return passages, urls


def check_teacher_presence(
    teachers
):
    missing = []

    for cid, target_urls in teachers.items():

        idx = str(
            int(
                cid.rsplit(
                    "_",
                    1
                )[1]
            )
        )

        store_path = STORE / f"{idx}.json"

        present = set()

        for row in read_jsonl(
            store_path
        ):
            present.add(
                canon(
                    row.get(
                        "url",
                        ""
                    )
                )
            )

        for url in target_urls:

            if canon(url) not in present:

                missing.append(
                    (cid, url)
                )

    return missing


def summarize(results):
    all_ranks = []

    for row in results.values():

        all_ranks.extend(
            row["teacher_ranks"]
        )

    denominator = len(
        all_ranks
    )

    lines = []

    lines.append(
        "=== STAGE 9A TEACHER COVERAGE ==="
    )

    lines.append(
        f"teacher_evidences = {denominator}"
    )

    for k in [
        1,
        5,
        10,
        20,
        30,
        50,
        100
    ]:

        found = sum(
            rank is not None
            and rank <= k
            for rank in all_ranks
        )

        ratio = (
            found / denominator
            if denominator
            else 0.0
        )

        lines.append(
            f"Recall@{k} = "
            f"{found}/{denominator} "
            f"= {ratio:.6f}"
        )

    missing100 = sum(
        rank is None
        or rank > 100
        for rank in all_ranks
    )

    lines.append(
        f"missing@100 = {missing100}"
    )

    return "\n".join(lines)


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

    parser.add_argument(
        "--preflight-only",
        action="store_true"
    )

    args = parser.parse_args()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    teachers = parse_teachers(
        AUDIT.read_text(
            encoding="utf-8"
        )
    )

    frozen = json.loads(
        FROZEN.read_text(
            encoding="utf-8"
        )
    )

    claim_text = {
        row["claim_id"]:
            row["claim_text"]

        for row in frozen[
            "methods"
        ]["CANDIDATE_X"]
    }

    print(
        "teacher claims =",
        len(teachers)
    )

    print(
        "teacher URLs =",
        sum(
            len(v)
            for v in teachers.values()
        )
    )

    assert len(teachers) == 12

    total_teacher = sum(
        len(v)
        for v in teachers.values()
    )

    assert total_teacher == 15

    missing = check_teacher_presence(
        teachers
    )

    print(
        "teacher URLs missing "
        "from raw text store =",
        len(missing)
    )

    if missing:
        for x in missing:
            print(
                "MISSING:",
                x
            )

        raise RuntimeError(
            "Teacher URL/store mismatch."
        )

    print(
        "TEACHER_STORE_PREFLIGHT_OK"
    )

    if args.preflight_only:
        return

    print(
        "Loading tokenizer:",
        MODEL
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL,
            padding_side="left"
        )
    )

    print(
        "Loading model:",
        MODEL
    )

    model = (
        AutoModel
        .from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )

    print(
        "MODEL_DEVICE =",
        model.device
    )

    if CHECKPOINT.exists():

        checkpoint = json.loads(
            CHECKPOINT.read_text(
                encoding="utf-8"
            )
        )

        results = checkpoint.get(
            "results",
            {}
        )

        print(
            "Checkpoint loaded:",
            len(results),
            "/",
            len(teachers)
        )

    else:
        results = {}

    items = list(
        teachers.items()
    )

    for n, (
        cid,
        teacher_urls
    ) in enumerate(
        items,
        1
    ):

        if cid in results:

            print(
                f"[CACHE] "
                f"{n}/{len(items)} "
                f"{cid}"
            )

            continue

        idx = str(
            int(
                cid.rsplit(
                    "_",
                    1
                )[1]
            )
        )

        store_path = (
            STORE / f"{idx}.json"
        )

        passages, urls = (
            build_passages(
                store_path,
                tokenizer
            )
        )

        if not passages:
            raise RuntimeError(
                f"No passages: {cid}"
            )

        print(
            f"[CLAIM] "
            f"{n}/{len(items)} "
            f"{cid} "
            f"chunks={len(passages)}",
            flush=True
        )

        query_embedding = (
            encode_texts(
                model,
                tokenizer,
                [claim_text[cid]],
                args.batch_size,
                args.max_length,
                is_query=True
            )[0]
        )

        best_by_url = {}

        for start in range(
            0,
            len(passages),
            args.batch_size
        ):

            batch_passages = passages[
                start:
                start + args.batch_size
            ]

            batch_urls = urls[
                start:
                start + args.batch_size
            ]

            embeddings = encode_texts(
                model,
                tokenizer,
                batch_passages,
                args.batch_size,
                args.max_length,
                is_query=False
            )

            scores = (
                embeddings
                @ query_embedding
            )

            for local_i, (
                url,
                score,
                chunk
            ) in enumerate(
                zip(
                    batch_urls,
                    scores,
                    batch_passages
                )
            ):

                cu = canon(
                    url
                )

                score = float(
                    score
                )

                old = (
                    best_by_url.get(
                        cu
                    )
                )

                if (
                    old is None
                    or score > old["score"]
                ):

                    best_by_url[cu] = {
                        "score":
                            score,

                        "url":
                            url,

                        "best_chunk":
                            chunk
                    }

            del embeddings
            del scores

        ranked = sorted(
            best_by_url.values(),
            key=lambda x:
                x["score"],
            reverse=True
        )

        rank_map = {
            canon(row["url"]):
                rank

            for rank, row
            in enumerate(
                ranked,
                1
            )
        }

        teacher_ranks = [
            rank_map.get(
                canon(url)
            )
            for url in teacher_urls
        ]

        teacher_details = []

        by_canon = {
            canon(row["url"]):
                row
            for row in ranked
        }

        for url, rank in zip(
            teacher_urls,
            teacher_ranks
        ):

            found = by_canon.get(
                canon(url)
            )

            teacher_details.append({
                "url":
                    url,

                "rank":
                    rank,

                "score":
                    (
                        found["score"]
                        if found
                        else None
                    ),

                "best_chunk":
                    (
                        found["best_chunk"]
                        if found
                        else None
                    )
            })

        results[cid] = {
            "claim":
                claim_text[cid],

            "num_chunks":
                len(passages),

            "num_urls":
                len(ranked),

            "teacher_urls":
                teacher_urls,

            "teacher_ranks":
                teacher_ranks,

            "teacher_details":
                teacher_details,

            "top100": [
                {
                    "rank":
                        rank,

                    "url":
                        row["url"],

                    "score":
                        row["score"],

                    "best_chunk":
                        row["best_chunk"]
                }

                for rank, row
                in enumerate(
                    ranked[:100],
                    1
                )
            ]
        }

        atomic_json(
            {
                "model":
                    MODEL,

                "task":
                    TASK,

                "results":
                    results
            },
            CHECKPOINT
        )

        print(
            f"[DONE] {cid} "
            f"teacher_ranks="
            f"{teacher_ranks}",
            flush=True
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_obj = {
        "status":
            "STAGE9A_COMPLETE",

        "model":
            MODEL,

        "task":
            TASK,

        "batch_size":
            args.batch_size,

        "max_length":
            args.max_length,

        "results":
            results
    }

    atomic_json(
        final_obj,
        FINAL
    )

    summary = summarize(
        results
    )

    SUMMARY.write_text(
        summary + "\n",
        encoding="utf-8"
    )

    print()
    print(summary)
    print()
    print(
        "STAGE9A_COMPLETE"
    )
    print(
        "RESULT =",
        FINAL
    )
    print(
        "SUMMARY =",
        SUMMARY
    )


if __name__ == "__main__":
    main()
