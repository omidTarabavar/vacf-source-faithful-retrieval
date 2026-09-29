import json
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode
from itertools import product

BM25 = Path(
    "/workspace/STAGE9B_RESULTS/BM25_PREFILTER_RESULTS.json"
)

SEM = Path(
    "/workspace/STAGE9C_RESULTS/STAGE9C_QWEN_RERANK.json"
)

RR = Path(
    "/workspace/STAGE9D_RESULTS/STAGE9D_RERANK.json"
)

OUT = Path(
    "/workspace/STAGE9E_RRF_SEARCH.txt"
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


bm25 = json.loads(
    BM25.read_text()
)

sem = json.loads(
    SEM.read_text()
)["results"]

rr = json.loads(
    RR.read_text()
)["results"]


def evaluate(K, wb, ws, wr):

    all_teacher_ranks = []
    per_claim = {}

    for cid in sem:

        b_row = bm25[cid]
        s_row = sem[cid]
        r_row = rr[cid]

        # Candidate universe = BM25 Top250,
        # which already contains 15/15 teacher evidences.
        candidates = b_row["top10000"][:250]

        bm25_rank = {
            canon(x["url"]): x["rank"]
            for x in candidates
        }

        semantic_rank = {
            canon(x["url"]): x["rank"]
            for x in s_row["top250_qwen"]
        }

        reranker_rank = {
            canon(x["url"]): i + 1
            for i, x in enumerate(
                r_row["reranked"]
            )
        }

        scored = []

        for x in candidates:

            cu = canon(x["url"])

            rb = bm25_rank.get(cu)
            rs = semantic_rank.get(cu)
            rrk = reranker_rank.get(cu)

            score = 0.0

            if rb is not None and wb > 0:
                score += wb / (K + rb)

            if rs is not None and ws > 0:
                score += ws / (K + rs)

            if rrk is not None and wr > 0:
                score += wr / (K + rrk)

            scored.append(
                (score, x["url"])
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        fmap = {
            canon(url): rank
            for rank, (_, url)
            in enumerate(
                scored,
                1
            )
        }

        teacher_ranks = [
            fmap.get(
                canon(url)
            )
            for url in s_row[
                "teacher_urls"
            ]
        ]

        per_claim[cid] = teacher_ranks
        all_teacher_ranks.extend(
            teacher_ranks
        )

    n = len(all_teacher_ranks)

    def recall(k):
        return sum(
            r is not None and r <= k
            for r in all_teacher_ranks
        )

    mrr = sum(
        0.0 if r is None else 1.0 / r
        for r in all_teacher_ranks
    ) / n

    return {
        "K": K,
        "wb": wb,
        "ws": ws,
        "wr": wr,
        "r1": recall(1),
        "r3": recall(3),
        "r5": recall(5),
        "r10": recall(10),
        "r20": recall(20),
        "r50": recall(50),
        "mrr": mrr,
        "per_claim": per_claim,
    }


Ks = [5, 10, 20, 40, 60]

weights = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]


runs = []

for K, wb, ws, wr in product(
    Ks,
    weights,
    weights,
    weights
):
    if wb == 0 and ws == 0 and wr == 0:
        continue

    # Require semantic branch to remain present.
    if ws == 0:
        continue

    runs.append(
        evaluate(
            K,
            wb,
            ws,
            wr
        )
    )


# Primary goal: Recall@10.
# Tie breakers: Recall@5, Recall@20, MRR.
runs.sort(
    key=lambda x: (
        x["r10"],
        x["r5"],
        x["r20"],
        x["mrr"]
    ),
    reverse=True
)


best = runs[0]

lines = []

lines.append(
    "=== STAGE 9E THREE-WAY RRF SEARCH ==="
)

lines.append("")
lines.append(
    "Objective order: Recall@10 > Recall@5 > Recall@20 > MRR"
)

lines.append("")
lines.append("TOP 20 CONFIGURATIONS")
lines.append("")


for i, x in enumerate(
    runs[:20],
    1
):

    lines.append(
        f"{i:02d}. "
        f"K={x['K']} "
        f"BM25={x['wb']} "
        f"SEM={x['ws']} "
        f"RR={x['wr']} | "
        f"R@5={x['r5']}/15 "
        f"R@10={x['r10']}/15 "
        f"R@20={x['r20']}/15 "
        f"MRR={x['mrr']:.6f}"
    )


lines += [
    "",
    "=== BEST ===",
    f"K = {best['K']}",
    f"BM25 weight = {best['wb']}",
    f"Semantic weight = {best['ws']}",
    f"Reranker weight = {best['wr']}",
    "",
    f"Recall@1  = {best['r1']}/15",
    f"Recall@3  = {best['r3']}/15",
    f"Recall@5  = {best['r5']}/15",
    f"Recall@10 = {best['r10']}/15",
    f"Recall@20 = {best['r20']}/15",
    f"Recall@50 = {best['r50']}/15",
    f"MRR = {best['mrr']:.6f}",
    "",
    "PER CLAIM TEACHER RANKS"
]


for cid, ranks in best[
    "per_claim"
].items():

    lines.append(
        f"{cid}: {ranks}"
    )


text = "\n".join(lines)

OUT.write_text(
    text + "\n",
    encoding="utf-8"
)

print(text)
print()
print("SAVED:", OUT)
