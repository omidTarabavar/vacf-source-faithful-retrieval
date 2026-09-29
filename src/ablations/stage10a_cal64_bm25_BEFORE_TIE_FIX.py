import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode


CANDIDATE = Path(
    "/workspace/V5B_FREEZE/CANDIDATE_Y_64_FROZEN.json"
)

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

OUTDIR = Path("/workspace/STAGE10A_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTDIR / "checkpoint.json"
FINAL = OUTDIR / "CAL64_BM25_TOP250.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

TOPK = 250


STOP = {
    "a","an","the","is","are","was","were","be","been","being",
    "of","to","in","on","at","for","from","by","with","and","or",
    "as","that","this","these","those","it","its","their","his",
    "her","has","have","had","shows","show","showing",
    "image","photo","photograph","picture","claim","says","said"
}


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        (text or "").lower()
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


def iter_pieces(path):
    with open(path, encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            url = row.get("url", "")

            vals = row.get(
                "url2text",
                []
            )

            if isinstance(vals, str):
                vals = [vals]

            for text in vals:

                text = str(text).strip()

                if text:
                    yield url, text


def process_claim(job):
    cid = job["claim_id"]
    claim = job["claim_text"]

    qtokens_all = tokenize(claim)

    qtokens = [
        x for x in qtokens_all
        if x not in STOP
        and len(x) > 1
    ]

    if len(qtokens) < 2:
        qtokens = qtokens_all

    qtf = Counter(qtokens)
    qset = set(qtokens)

    idx = str(
        int(cid.rsplit("_", 1)[1])
    )

    path = STORE / f"{idx}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    # ------------------------------
    # PASS 1: BM25 corpus statistics
    # ------------------------------

    N = 0
    total_len = 0
    df = Counter()

    for _, text in iter_pieces(path):

        toks = tokenize(text)

        if not toks:
            continue

        N += 1
        total_len += len(toks)

        seen = {
            t for t in toks
            if t in qset
        }

        for t in seen:
            df[t] += 1

    avgdl = (
        total_len / N
        if N
        else 1.0
    )

    idf = {}

    for t in qset:

        n = df.get(t, 0)

        idf[t] = math.log(
            1.0
            +
            (N - n + 0.5)
            / (n + 0.5)
        )

    # ------------------------------
    # PASS 2: max BM25 score per URL
    # ------------------------------

    k1 = 1.5
    b = 0.75

    best_by_url = {}

    for url, text in iter_pieces(path):

        toks = tokenize(text)

        if not toks:
            continue

        dl = len(toks)

        counts = Counter(
            t for t in toks
            if t in qset
        )

        score = 0.0

        for term, qcount in qtf.items():

            tf = counts.get(term, 0)

            if tf == 0:
                continue

            denom = (
                tf
                + k1 * (
                    1.0
                    - b
                    + b * dl / avgdl
                )
            )

            score += (
                idf[term]
                * (
                    tf * (k1 + 1.0)
                    / denom
                )
                * min(qcount, 2)
            )

        useful = [
            t for t in qtokens
            if len(t) >= 4
        ]

        low = text.lower()

        for a, bb in zip(
            useful,
            useful[1:]
        ):
            if f"{a} {bb}" in low:
                score += 0.35

        cu = canon(url)

        if not cu:
            continue

        old = best_by_url.get(cu)

        if (
            old is None
            or score > old["score"]
        ):
            best_by_url[cu] = {
                "url": url,
                "score": float(score),
                "best_piece": text
            }

    ranked = sorted(
        best_by_url.values(),
        key=lambda x: (
            x["score"],
            canon(x["url"])
        ),
        reverse=True
    )

    top = []

    for rank, row in enumerate(
        ranked[:TOPK],
        1
    ):
        top.append({
            "rank": rank,
            **row
        })

    return {
        "claim_id": cid,
        "claim_text": claim,
        "query_terms": qtokens,
        "num_pieces": N,
        "num_urls": len(ranked),
        "top250": top
    }


data = json.loads(
    CANDIDATE.read_text(
        encoding="utf-8"
    )
)

rows = data[
    "methods"
]["CANDIDATE_X"]

assert len(rows) == 64
assert len({
    x["claim_id"]
    for x in rows
}) == 64


if CHECKPOINT.exists():

    cp = json.loads(
        CHECKPOINT.read_text(
            encoding="utf-8"
        )
    )

    results = cp.get(
        "results",
        {}
    )

else:
    results = {}


jobs = [
    {
        "claim_id": row["claim_id"],
        "claim_text": row["claim_text"]
    }
    for row in rows
    if row["claim_id"] not in results
]


print(
    "TOTAL CLAIMS =",
    len(rows)
)

print(
    "CACHED =",
    len(results)
)

print(
    "REMAINING =",
    len(jobs)
)

workers = min(
    12,
    os.cpu_count() or 4
)

print(
    "WORKERS =",
    workers
)


completed_now = 0


with ProcessPoolExecutor(
    max_workers=workers
) as ex:

    futures = {
        ex.submit(
            process_claim,
            job
        ): job["claim_id"]

        for job in jobs
    }


    for fut in as_completed(futures):

        cid = futures[fut]

        try:
            row = fut.result()

        except Exception as e:

            print(
                f"[ERROR] {cid}: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            raise


        results[cid] = row
        completed_now += 1


        print(
            f"[DONE] "
            f"{len(results)}/64 "
            f"{cid} "
            f"pieces={row['num_pieces']} "
            f"urls={row['num_urls']} "
            f"top={len(row['top250'])}",
            flush=True
        )


        CHECKPOINT.write_text(
            json.dumps(
                {
                    "status":
                        "RUNNING",

                    "topk":
                        TOPK,

                    "results":
                        results
                },
                ensure_ascii=False,
                indent=2
            ) + "\n",
            encoding="utf-8"
        )


# Restore Candidate-Y order.
ordered = {
    row["claim_id"]:
        results[row["claim_id"]]
    for row in rows
}


assert len(ordered) == 64


for cid, row in ordered.items():

    assert len(
        row["top250"]
    ) > 0, cid


FINAL.write_text(
    json.dumps(
        {
            "status":
                "STAGE10A_COMPLETE",

            "cohort":
                "Cal64",

            "n":
                64,

            "topk":
                TOPK,

            "qrels_used":
                False,

            "results":
                ordered
        },
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


num_pieces = [
    row["num_pieces"]
    for row in ordered.values()
]

num_urls = [
    row["num_urls"]
    for row in ordered.values()
]

top_sizes = [
    len(row["top250"])
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


lines = [
    "=== STAGE 10A CAL64 BM25 ===",
    "status = COMPLETE",
    "claims = 64",
    "qrels_used = False",
    f"TOPK = {TOPK}",
    "",
    f"pieces min = {min(num_pieces)}",
    f"pieces median = {median(num_pieces)}",
    f"pieces max = {max(num_pieces)}",
    "",
    f"URLs min = {min(num_urls)}",
    f"URLs median = {median(num_urls)}",
    f"URLs max = {max(num_urls)}",
    "",
    f"TopK min = {min(top_sizes)}",
    f"TopK max = {max(top_sizes)}",
    "",
    f"completed_this_run = {completed_now}",
    "STAGE10A_CAL64_BM25_OK"
]


summary = "\n".join(lines)

SUMMARY.write_text(
    summary + "\n",
    encoding="utf-8"
)


CHECKPOINT.write_text(
    json.dumps(
        {
            "status":
                "COMPLETE",

            "topk":
                TOPK,

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
