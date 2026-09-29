import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode


BASE_FILE = Path(
    "/workspace/V5B_FREEZE/CANDIDATE_Y_64_FROZEN.json"
)

RRF_FILE = Path(
    "/workspace/STAGE10D_RESULTS/CAL64_RRF_RESCUE.json"
)

STORE = Path(
    "/workspace/stage8c_store/"
    "text_related_store_text_train"
)

OUTDIR = Path("/workspace/STAGE10E_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

FINAL = OUTDIR / "CAL64_V5B_PROTECTED_FUSION.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

MAX_EVIDENCE_TOKENS = 1200


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


def words(text):
    return re.findall(
        r"[a-z0-9]+",
        (text or "").lower()
    )


def iter_store(path):
    with open(path, encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            url = row.get("url", "")

            vals = row.get("url2text", [])

            if isinstance(vals, str):
                vals = [vals]

            yield url, [
                str(x).strip()
                for x in vals
                if str(x).strip()
            ]


def truncate_words(text, n=MAX_EVIDENCE_TOKENS):
    toks = text.split()

    if len(toks) <= n:
        return text.strip()

    return " ".join(toks[:n]).strip()


def overlap_score(a, b):
    aa = words(a)
    bb = words(b)

    if not aa or not bb:
        return 0.0

    ca = Counter(aa)
    cb = Counter(bb)

    common = sum(
        min(ca[k], cb[k])
        for k in ca.keys() & cb.keys()
    )

    return common / max(
        1,
        min(len(aa), len(bb))
    )


def hydrate_rescue_url(
    cid,
    url,
    best_passage
):
    idx = str(
        int(cid.rsplit("_", 1)[1])
    )

    path = STORE / f"{idx}.json"

    target = canon(url)

    raw_pieces = []

    for raw_url, vals in iter_store(path):

        if canon(raw_url) == target:
            raw_pieces.extend(vals)

    if not raw_pieces:
        return truncate_words(
            best_passage or ""
        )

    # Rank source-faithful raw pieces by similarity
    # to the semantic passage.
    scored = []

    for i, text in enumerate(raw_pieces):

        score = overlap_score(
            best_passage or "",
            text
        )

        scored.append(
            (score, i, text)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    chosen_indices = []

    # Always preserve lead/title context.
    for i in range(
        min(2, len(raw_pieces))
    ):
        chosen_indices.append(i)

    # Preserve local relevant source pieces.
    for _, i, _ in scored[:6]:
        if i not in chosen_indices:
            chosen_indices.append(i)

    chosen_indices = sorted(
        chosen_indices
    )

    merged = []

    seen = set()

    for i in chosen_indices:

        text = raw_pieces[i].strip()

        if text and text not in seen:
            merged.append(text)
            seen.add(text)

    hydrated = "\n\n".join(merged)

    return truncate_words(
        hydrated
    )


base_obj = json.loads(
    BASE_FILE.read_text(
        encoding="utf-8"
    )
)

rrf_obj = json.loads(
    RRF_FILE.read_text(
        encoding="utf-8"
    )
)


base_rows = base_obj[
    "methods"
]["CANDIDATE_X"]

rrf = rrf_obj["results"]


assert len(base_rows) == 64
assert len(rrf) == 64


results = []

stats = {
    "base_image_kept": [],
    "rrf_added": [],
    "fallback_added": [],
    "overlap_base_rrf_top10": []
}


for n, base in enumerate(
    base_rows,
    1
):

    cid = base["claim_id"]
    claim = base["claim_text"]

    assert cid in rrf

    selected = base["selected"]
    old_evidence = base["evidence"]

    # Existing source-faithful Candidate-Y evidence.
    old_ev_map = {
        canon(x["url"]): x
        for x in old_evidence
    }

    # ----------------------------
    # Candidate-Y anchor groups
    # ----------------------------

    visual_anchors = [
        x for x in selected
        if x.get("expert") == "BASE_IMAGE"
    ]

    fallback = [
        x for x in selected
        if x.get("expert") != "BASE_IMAGE"
    ]


    rescue = rrf[cid][
        "top250_rrf"
    ]

    rescue_top10 = rescue[:10]

    base_top10_set = {
        canon(x["url"])
        for x in selected[:10]
    }

    rescue_top10_set = {
        canon(x["url"])
        for x in rescue_top10
    }

    overlap = len(
        base_top10_set
        & rescue_top10_set
    )

    # ----------------------------
    # Protected fusion
    # ----------------------------

    final_selected = []
    seen = set()

    base_image_count = 0
    rescue_count = 0
    fallback_count = 0


    # 1. Preserve ALL BASE_IMAGE anchors.
    for x in visual_anchors:

        cu = canon(x["url"])

        if not cu or cu in seen:
            continue

        final_selected.append({
            "url": x["url"],
            "expert": "BASE_IMAGE_PROTECTED",
            "source_score": x.get("source_score")
        })

        seen.add(cu)
        base_image_count += 1

        if len(final_selected) == 10:
            break


    # 2. Fill residual slots from frozen RRF ranking.
    if len(final_selected) < 10:

        for x in rescue:

            cu = canon(x["url"])

            if not cu or cu in seen:
                continue

            final_selected.append({
                "url": x["url"],
                "expert": "TEXT_SEMANTIC_RESCUE",
                "source_score": x.get("fusion_score"),
                "bm25_rank": x.get("bm25_rank"),
                "semantic_rank": x.get("semantic_rank"),
                "reranker_rank": x.get("reranker_rank"),
                "best_passage": x.get("best_passage")
            })

            seen.add(cu)
            rescue_count += 1

            if len(final_selected) == 10:
                break


    # 3. Fallback to original S2_TEXT if necessary.
    if len(final_selected) < 10:

        for x in fallback:

            cu = canon(x["url"])

            if not cu or cu in seen:
                continue

            final_selected.append({
                "url": x["url"],
                "expert": "S2_TEXT_FALLBACK",
                "source_score": x.get("source_score")
            })

            seen.add(cu)
            fallback_count += 1

            if len(final_selected) == 10:
                break


    assert len(final_selected) <= 10

    if not final_selected:
        raise RuntimeError(
            f"No final evidence for {cid}"
        )


    # ----------------------------
    # Source-faithful hydration
    # ----------------------------

    final_evidence = []

    for x in final_selected:

        cu = canon(x["url"])

        # Reuse Stage8C localization whenever possible.
        if cu in old_ev_map:

            ev = old_ev_map[cu]

            text = ev.get(
                "text",
                ""
            )

            hydration = "STAGE8C_REUSED"

        else:

            text = hydrate_rescue_url(
                cid=cid,
                url=x["url"],
                best_passage=x.get(
                    "best_passage",
                    ""
                )
            )

            hydration = "RAW_TEXT_SOURCE_FAITHFUL"


        if not text.strip():
            text = (
                x.get("best_passage")
                or ""
            )

            hydration += "_FALLBACK_BEST_PASSAGE"


        final_evidence.append({
            "url": x["url"],
            "text": truncate_words(text),
            "expert": x["expert"],
            "hydration": hydration
        })


    results.append({
        "claim_id": cid,
        "claim_text": claim,
        "method": "V5B_PROTECTED_FUSION",
        "selected": final_selected,
        "evidence": final_evidence
    })


    stats[
        "base_image_kept"
    ].append(base_image_count)

    stats[
        "rrf_added"
    ].append(rescue_count)

    stats[
        "fallback_added"
    ].append(fallback_count)

    stats[
        "overlap_base_rrf_top10"
    ].append(overlap)


    print(
        f"[DONE] {n}/64 {cid} "
        f"BASE_IMAGE={base_image_count} "
        f"RESCUE={rescue_count} "
        f"FALLBACK={fallback_count} "
        f"overlap10={overlap}",
        flush=True
    )


def med(xs):
    ys = sorted(xs)
    n = len(ys)

    if n % 2:
        return ys[n//2]

    return (
        ys[n//2 - 1]
        + ys[n//2]
    ) / 2


final_obj = {
    "status": "STAGE10E_COMPLETE",
    "cohort": "Cal64",
    "n": 64,
    "qrels_used": False,
    "candidate_method": "V5B_PROTECTED_FUSION",
    "fusion_rule": {
        "1": "preserve all Candidate-Y BASE_IMAGE URLs",
        "2": "fill remaining slots from frozen Stage10D RRF ranking",
        "3": "fallback to Candidate-Y non-BASE_IMAGE URLs if needed"
    },
    "methods": {
        "V5B_PROTECTED_FUSION": results
    }
}


FINAL.write_text(
    json.dumps(
        final_obj,
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


lines = [
    "=== STAGE 10E V5-B PROTECTED FUSION ===",
    "status = COMPLETE",
    "claims = 64",
    "qrels_used = False",
    "",
    "Fusion rule:",
    "1. Preserve all BASE_IMAGE anchors",
    "2. Fill from frozen Stage10D RRF",
    "3. Fallback to original S2_TEXT only if needed",
    "",
    f"BASE_IMAGE kept min = {min(stats['base_image_kept'])}",
    f"BASE_IMAGE kept median = {med(stats['base_image_kept'])}",
    f"BASE_IMAGE kept max = {max(stats['base_image_kept'])}",
    "",
    f"RRF rescue added min = {min(stats['rrf_added'])}",
    f"RRF rescue added median = {med(stats['rrf_added'])}",
    f"RRF rescue added max = {max(stats['rrf_added'])}",
    "",
    f"Fallback added total = {sum(stats['fallback_added'])}",
    "",
    f"CandidateY/RRF Top10 overlap min = {min(stats['overlap_base_rrf_top10'])}",
    f"CandidateY/RRF Top10 overlap median = {med(stats['overlap_base_rrf_top10'])}",
    f"CandidateY/RRF Top10 overlap max = {max(stats['overlap_base_rrf_top10'])}",
    "",
    "STAGE10E_PROTECTED_FUSION_OK"
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
