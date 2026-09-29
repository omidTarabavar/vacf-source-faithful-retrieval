import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode


AUDIT = Path("/workspace/STAGE9_XXP_TEACHER_AUDIT.md")

FROZEN = Path(
    "/workspace/STAGE8C_CANDIDATE_Y_64.json"
)

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

OUTDIR = Path("/workspace/STAGE9B_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "BM25_PREFILTER_RESULTS.json"
SUMMARY = OUTDIR / "SUMMARY.txt"


STOP = {
    "a","an","the","is","are","was","were","be","been","being",
    "of","to","in","on","at","for","from","by","with","and","or",
    "as","that","this","these","those","it","its","their","his",
    "her","has","have","had","shows","show","showing","image",
    "photo","photograph","picture","claim","says","said"
}


def tokenize(s):
    return re.findall(
        r"[a-z0-9]+",
        (s or "").lower()
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


def parse_teachers(text):
    cur = None
    out = defaultdict(list)

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
            out[cur].append(m.group(1))

    return dict(out)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_pieces(path):
    for row in jsonl(path):

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

claims = {
    r["claim_id"]: r["claim_text"]
    for r in frozen[
        "methods"
    ]["CANDIDATE_X"]
}


print("teacher claims =", len(teachers))
print(
    "teacher URLs =",
    sum(len(v) for v in teachers.values())
)


results = {}


for num, (cid, target_urls) in enumerate(
    teachers.items(),
    1
):

    claim = claims[cid]

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

    # ----------------------------------------------------
    # PASS 1: document frequencies only for query terms
    # ----------------------------------------------------

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
            1.0 +
            (
                N - n + 0.5
            ) /
            (
                n + 0.5
            )
        )


    # ----------------------------------------------------
    # PASS 2: BM25 score per source piece
    # then MAX aggregation per URL
    # ----------------------------------------------------

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

            tf = counts.get(
                term,
                0
            )

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


        # Lightweight phrase/entity bonus
        low = text.lower()

        useful = [
            t for t in qtokens
            if len(t) >= 4
        ]

        bonus = 0.0

        for a, bb in zip(
            useful,
            useful[1:]
        ):
            phrase = f"{a} {bb}"

            if phrase in low:
                bonus += 0.35

        score += bonus


        cu = canon(url)

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
        key=lambda x: x["score"],
        reverse=True
    )


    rankmap = {
        canon(row["url"]): rank
        for rank, row
        in enumerate(
            ranked,
            1
        )
    }


    teacher_ranks = [
        rankmap.get(
            canon(url)
        )
        for url in target_urls
    ]


    print(
        f"[{num}/{len(teachers)}] "
        f"{cid} "
        f"pieces={N} "
        f"urls={len(ranked)} "
        f"teacher_ranks={teacher_ranks}",
        flush=True
    )


    results[cid] = {
        "claim": claim,
        "query_terms": qtokens,
        "num_pieces": N,
        "num_urls": len(ranked),
        "teacher_urls": target_urls,
        "teacher_ranks": teacher_ranks,
        "top10000": [
            {
                "rank": rank,
                "url": row["url"],
                "score": row["score"],
                "best_piece": row["best_piece"]
            }
            for rank, row in enumerate(
                ranked[:10000],
                1
            )
        ]
    }


OUT.write_text(
    json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


all_ranks = [
    rank
    for row in results.values()
    for rank in row["teacher_ranks"]
]

den = len(all_ranks)

lines = [
    "=== STAGE 9B BM25 PREFILTER ===",
    f"teacher_claims = {len(results)}",
    f"teacher_evidences = {den}",
]


for k in [
    10,
    50,
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000
]:

    found = sum(
        r is not None
        and r <= k
        for r in all_ranks
    )

    ratio = (
        found / den
        if den
        else 0
    )

    lines.append(
        f"Recall@{k} = "
        f"{found}/{den} "
        f"= {ratio:.6f}"
    )


missing = [
    r for r in all_ranks
    if r is None
    or r > 10000
]

lines.append(
    f"missing@10000 = {len(missing)}"
)

summary = "\n".join(lines)

SUMMARY.write_text(
    summary + "\n",
    encoding="utf-8"
)

print()
print(summary)
print()
print("STAGE9B_COMPLETE")
print("RESULT =", OUT)
print("SUMMARY =", SUMMARY)
