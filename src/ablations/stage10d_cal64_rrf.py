import json
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode


BM25_FILE = Path(
    "/workspace/STAGE10A_RESULTS/CAL64_BM25_TOP250.json"
)

SEM_FILE = Path(
    "/workspace/STAGE10B_RESULTS/CAL64_QWEN_SEMANTIC.json"
)

RR_FILE = Path(
    "/workspace/STAGE10C_RESULTS/CAL64_QWEN_RERANKER.json"
)

OUTDIR = Path("/workspace/STAGE10D_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

FINAL = OUTDIR / "CAL64_RRF_RESCUE.json"
SUMMARY = OUTDIR / "SUMMARY.txt"


# ==========================================
# FROZEN from Stage 9E
# ==========================================

K = 10

W_BM25 = 0.25
W_SEM = 1.00
W_RR = 0.50


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


bm25_obj = json.loads(
    BM25_FILE.read_text(
        encoding="utf-8"
    )
)

sem_obj = json.loads(
    SEM_FILE.read_text(
        encoding="utf-8"
    )
)

rr_obj = json.loads(
    RR_FILE.read_text(
        encoding="utf-8"
    )
)


bm25 = bm25_obj["results"]
sem = sem_obj["results"]
rr = rr_obj["results"]


assert len(bm25) == 64
assert len(sem) == 64
assert len(rr) == 64

assert list(bm25.keys()) == list(sem.keys())
assert list(bm25.keys()) == list(rr.keys())


results = {}


for n, cid in enumerate(
    bm25.keys(),
    1
):

    brow = bm25[cid]
    srow = sem[cid]
    rrow = rr[cid]


    # -----------------------------------------
    # Candidate universe = frozen BM25 Top250
    # -----------------------------------------

    candidates = brow[
        "top250"
    ][:250]


    bm25_rank = {
        canon(x["url"]):
            x["rank"]
        for x in candidates
    }


    semantic_rank = {
        canon(x["url"]):
            x["rank"]
        for x in srow[
            "top250_qwen"
        ]
    }


    reranker_rank = {
        canon(x["url"]):
            x["rank"]
        for x in rrow[
            "reranked"
        ]
    }


    semantic_rows = {
        canon(x["url"]):
            x
        for x in srow[
            "top250_qwen"
        ]
    }


    reranker_rows = {
        canon(x["url"]):
            x
        for x in rrow[
            "reranked"
        ]
    }


    fused = []


    # IMPORTANT:
    # preserve BM25 candidate order so tie behavior
    # exactly follows Stage 9E.
    for x in candidates:

        url = x["url"]
        cu = canon(url)

        rb = bm25_rank.get(cu)
        rs = semantic_rank.get(cu)
        rrk = reranker_rank.get(cu)

        score = 0.0


        if rb is not None:

            score += (
                W_BM25
                / (K + rb)
            )


        if rs is not None:

            score += (
                W_SEM
                / (K + rs)
            )


        if rrk is not None:

            score += (
                W_RR
                / (K + rrk)
            )


        sem_row = semantic_rows.get(
            cu,
            {}
        )

        rr_row = reranker_rows.get(
            cu,
            {}
        )


        fused.append({
            "url":
                url,

            "fusion_score":
                float(score),

            "bm25_rank":
                rb,

            "semantic_rank":
                rs,

            "reranker_rank":
                rrk,

            "bm25_score":
                float(x["score"]),

            "semantic_score":
                (
                    float(
                        sem_row["score"]
                    )
                    if "score" in sem_row
                    else None
                ),

            "reranker_score":
                (
                    float(
                        rr_row["reranker_score"]
                    )
                    if "reranker_score"
                    in rr_row
                    else None
                ),

            "best_passage":
                sem_row.get(
                    "best_passage"
                )
        })


    # Stable sort, same logic as Stage9E.
    fused.sort(
        key=lambda x:
            x["fusion_score"],
        reverse=True
    )


    for rank, x in enumerate(
        fused,
        1
    ):
        x["rank"] = rank


    results[cid] = {
        "claim_id":
            cid,

        "claim_text":
            brow["claim_text"],

        "num_candidates":
            len(fused),

        "top250_rrf":
            fused
    }


    top = fused[:3]

    print(
        f"[DONE] {n}/64 "
        f"{cid} "
        f"top1={top[0]['url'][:100]}",
        flush=True
    )


# -----------------------------------------
# Write final
# -----------------------------------------

FINAL.write_text(
    json.dumps(
        {
            "status":
                "STAGE10D_COMPLETE",

            "cohort":
                "Cal64",

            "n":
                64,

            "qrels_used":
                False,

            "frozen_design": {
                "K": K,
                "BM25_weight": W_BM25,
                "semantic_weight": W_SEM,
                "reranker_weight": W_RR
            },

            "results":
                results
        },
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


sizes = [
    x["num_candidates"]
    for x in results.values()
]


lines = [
    "=== STAGE 10D CAL64 FROZEN RRF ===",
    "status = COMPLETE",
    "claims = 64",
    "qrels_used = False",
    "",
    f"K = {K}",
    f"BM25 weight = {W_BM25}",
    f"Semantic weight = {W_SEM}",
    f"Reranker weight = {W_RR}",
    "",
    f"candidate min = {min(sizes)}",
    f"candidate max = {max(sizes)}",
    "",
    "STAGE10D_CAL64_RRF_OK"
]


summary = "\n".join(lines)

SUMMARY.write_text(
    summary + "\n",
    encoding="utf-8"
)


print()
print(summary)
print()
print("FINAL =", FINAL)
print("SUMMARY =", SUMMARY)
