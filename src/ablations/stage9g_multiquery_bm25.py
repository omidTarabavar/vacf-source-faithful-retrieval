import json
import math
import re
from collections import Counter
from itertools import product
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode


BASE = Path(
    "/workspace/STAGE9B_RESULTS/"
    "BM25_PREFILTER_RESULTS.json"
)

VLM = Path(
    "/workspace/STAGE9F_RESULTS/"
    "MULTIMODAL_QUERIES.json"
)

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

OUTDIR = Path("/workspace/STAGE9G_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTJSON = OUTDIR / "MULTIQUERY_BM25.json"
SUMMARY = OUTDIR / "SUMMARY.txt"


STOP = {
    "a","an","the","is","are","was","were","be","been","being",
    "of","to","in","on","at","for","from","by","with","and","or",
    "as","that","this","these","those","it","its","their","his",
    "her","has","have","had","shows","show","showing",
    "image","photo","photograph","picture","claim","says","said"
}


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


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        (text or "").lower()
    )


def clean_tokens(text):
    toks = tokenize(text)

    x = [
        t for t in toks
        if t not in STOP
        and len(t) > 1
    ]

    return x if len(x) >= 2 else toks


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_pieces(path):
    for row in jsonl(path):

        url = row.get("url", "")

        vals = row.get("url2text", [])

        if isinstance(vals, str):
            vals = [vals]

        for text in vals:
            text = str(text).strip()

            if text:
                yield url, text


base = json.loads(
    BASE.read_text(encoding="utf-8")
)

vlm = json.loads(
    VLM.read_text(encoding="utf-8")
)


claim_data = {}


for num, (cid, brow) in enumerate(
    base.items(),
    1
):

    vrow = vlm[cid]

    parsed = vrow.get("parsed") or {}

    generated = parsed.get("queries") or []

    generated = [
        str(x).strip()
        for x in generated
        if str(x).strip()
    ][:5]


    # Original claim = channel 0
    queries = [
        brow["claim"]
    ] + generated


    # Explicit visual/OCR cue channel.
    cue_parts = []

    cue_parts.extend(
        parsed.get("entities") or []
    )

    cue_parts.extend(
        parsed.get("ocr_text") or []
    )

    cue_parts.extend(
        parsed.get("visual_cues") or []
    )

    cue_query = " ".join(
        str(x).strip()
        for x in cue_parts
        if str(x).strip()
    )

    if cue_query:
        queries.append(cue_query)
        cue_idx = len(queries) - 1
    else:
        cue_idx = None


    qtoks = [
        clean_tokens(q)
        for q in queries
    ]

    qtfs = [
        Counter(x)
        for x in qtoks
    ]

    union_terms = set()

    for x in qtoks:
        union_terms.update(x)


    idx = str(
        int(cid.rsplit("_", 1)[1])
    )

    path = STORE / f"{idx}.json"


    # ---------------------------------------
    # PASS 1: corpus stats
    # ---------------------------------------

    N = 0
    total_len = 0
    df = Counter()


    for _, text in iter_pieces(path):

        toks = tokenize(text)

        if not toks:
            continue

        N += 1
        total_len += len(toks)

        seen = (
            set(toks)
            & union_terms
        )

        for term in seen:
            df[term] += 1


    avgdl = (
        total_len / N
        if N
        else 1.0
    )


    idf = {}

    for term in union_terms:

        n = df.get(term, 0)

        idf[term] = math.log(
            1.0
            +
            (
                N - n + 0.5
            )
            /
            (
                n + 0.5
            )
        )


    # ---------------------------------------
    # PASS 2
    # ---------------------------------------

    best = [
        {}
        for _ in queries
    ]

    all_urls = {}

    k1 = 1.5
    b = 0.75


    for url, text in iter_pieces(path):

        cu = canon(url)

        if not cu:
            continue

        all_urls[cu] = url

        toks = tokenize(text)

        if not toks:
            continue

        dl = len(toks)

        counts = Counter(
            t for t in toks
            if t in union_terms
        )


        for qi, qtf in enumerate(qtfs):

            score = 0.0

            for term, qcount in qtf.items():

                tf = counts.get(term, 0)

                if tf == 0:
                    continue

                denom = (
                    tf
                    + k1
                    * (
                        1.0
                        - b
                        + b * dl / avgdl
                    )
                )

                score += (
                    idf.get(term, 0.0)
                    * (
                        tf * (k1 + 1)
                        / denom
                    )
                    * min(qcount, 2)
                )


            # modest adjacent phrase bonus
            useful = [
                t for t in qtoks[qi]
                if len(t) >= 4
            ]

            low = text.lower()

            for a, bb in zip(
                useful,
                useful[1:]
            ):
                if f"{a} {bb}" in low:
                    score += 0.35


            old = best[qi].get(cu)

            if old is None or score > old:
                best[qi][cu] = float(score)


    rank_maps = []


    for qi in range(len(queries)):

        ranked = sorted(
            all_urls.keys(),
            key=lambda cu: (
                best[qi].get(cu, 0.0),
                cu
            ),
            reverse=True
        )

        rank_maps.append({
            cu: rank
            for rank, cu
            in enumerate(ranked, 1)
        })


    teacher_urls = brow[
        "teacher_urls"
    ]


    channel_teacher_ranks = []

    for rm in rank_maps:

        channel_teacher_ranks.append([
            rm.get(canon(u))
            for u in teacher_urls
        ])


    print(
        f"[{num}/{len(base)}] "
        f"{cid} "
        f"pieces={N} "
        f"urls={len(all_urls)} "
        f"channels={len(queries)} "
        f"orig={channel_teacher_ranks[0]}",
        flush=True
    )


    claim_data[cid] = {
        "claim": brow["claim"],
        "teacher_urls": teacher_urls,
        "queries": queries,
        "cue_idx": cue_idx,
        "urls": all_urls,
        "rank_maps": rank_maps,
        "channel_teacher_ranks":
            channel_teacher_ranks
    }


# ==================================================
# Multi-query RRF search
# ==================================================

Ks = [5, 10, 20, 40, 60]

orig_weights = [
    0.5,
    1.0,
    2.0,
    4.0
]

expanded_weights = [
    0.25,
    0.5,
    1.0,
    2.0
]

cue_weights = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0
]


def run_config(
    K,
    w_orig,
    w_exp,
    w_cue
):

    all_teacher_ranks = []
    per_claim = {}


    for cid, row in claim_data.items():

        rank_maps = row["rank_maps"]
        cue_idx = row["cue_idx"]

        # channels 1..5 = generated queries,
        # cue channel is excluded from these
        exp_indices = [
            i
            for i in range(
                1,
                len(rank_maps)
            )
            if i != cue_idx
        ]


        scored = []


        for cu, raw_url in row[
            "urls"
        ].items():

            score = 0.0

            # Original claim
            r = rank_maps[0].get(cu)

            if r is not None:
                score += (
                    w_orig
                    / (K + r)
                )


            # Average over generated queries.
            if exp_indices:

                tmp = []

                for qi in exp_indices:

                    rr = rank_maps[
                        qi
                    ].get(cu)

                    if rr is not None:
                        tmp.append(
                            1.0 / (K + rr)
                        )

                if tmp:
                    score += (
                        w_exp
                        * sum(tmp)
                        / len(tmp)
                    )


            # VLM OCR/entity/cue channel
            if (
                cue_idx is not None
                and w_cue > 0
            ):

                rr = rank_maps[
                    cue_idx
                ].get(cu)

                if rr is not None:
                    score += (
                        w_cue
                        / (K + rr)
                    )


            scored.append(
                (score, cu)
            )


        scored.sort(
            key=lambda x:
                x[0],
            reverse=True
        )


        fmap = {
            cu: rank
            for rank, (_, cu)
            in enumerate(
                scored,
                1
            )
        }


        ranks = [
            fmap.get(
                canon(u)
            )
            for u in row[
                "teacher_urls"
            ]
        ]


        per_claim[cid] = ranks

        all_teacher_ranks.extend(
            ranks
        )


    n = len(
        all_teacher_ranks
    )


    def recall(k):
        return sum(
            r is not None
            and r <= k
            for r in all_teacher_ranks
        )


    mrr = sum(
        0.0
        if r is None
        else 1.0 / r
        for r in all_teacher_ranks
    ) / n


    return {
        "K": K,
        "wo": w_orig,
        "we": w_exp,
        "wc": w_cue,
        "r1": recall(1),
        "r3": recall(3),
        "r5": recall(5),
        "r10": recall(10),
        "r20": recall(20),
        "r50": recall(50),
        "mrr": mrr,
        "per_claim": per_claim
    }


runs = []


for K, wo, we, wc in product(
    Ks,
    orig_weights,
    expanded_weights,
    cue_weights
):

    runs.append(
        run_config(
            K,
            wo,
            we,
            wc
        )
    )


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


lines = [
    "=== STAGE 9G MULTIQUERY BM25 ===",
    "",
    "Baseline Stage9E final RRF:",
    "Recall@10 = 11/15",
    "MRR = 0.309686",
    "",
    "TOP 20 CONFIGURATIONS",
    ""
]


for i, x in enumerate(
    runs[:20],
    1
):

    lines.append(
        f"{i:02d}. "
        f"K={x['K']} "
        f"ORIG={x['wo']} "
        f"EXP={x['we']} "
        f"CUE={x['wc']} | "
        f"R@5={x['r5']}/15 "
        f"R@10={x['r10']}/15 "
        f"R@20={x['r20']}/15 "
        f"MRR={x['mrr']:.6f}"
    )


lines += [
    "",
    "=== BEST ===",
    f"K = {best['K']}",
    f"Original weight = {best['wo']}",
    f"Expanded-query weight = {best['we']}",
    f"Cue weight = {best['wc']}",
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


summary = "\n".join(lines)

SUMMARY.write_text(
    summary + "\n",
    encoding="utf-8"
)


# Save compact diagnostic output.
serializable = {}

for cid, row in claim_data.items():

    serializable[cid] = {
        "claim":
            row["claim"],
        "teacher_urls":
            row["teacher_urls"],
        "queries":
            row["queries"],
        "cue_idx":
            row["cue_idx"],
        "channel_teacher_ranks":
            row["channel_teacher_ranks"],
        "best_fused_teacher_ranks":
            best["per_claim"][cid]
    }


OUTJSON.write_text(
    json.dumps(
        {
            "best":
                best,
            "claims":
                serializable
        },
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


print()
print(summary)
print()
print("STAGE9G_COMPLETE")
print("SUMMARY =", SUMMARY)
print("JSON =", OUTJSON)
