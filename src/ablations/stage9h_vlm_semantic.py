import json
import re
import gc
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL = "Qwen/Qwen3-Embedding-4B"

BM25_FILE = Path(
    "/workspace/STAGE9B_RESULTS/BM25_PREFILTER_RESULTS.json"
)

SEM_FILE = Path(
    "/workspace/STAGE9C_RESULTS/STAGE9C_QWEN_RERANK.json"
)

RR_FILE = Path(
    "/workspace/STAGE9D_RESULTS/STAGE9D_RERANK.json"
)

VLM_FILE = Path(
    "/workspace/STAGE9F_RESULTS/MULTIMODAL_QUERIES.json"
)

OUTDIR = Path("/workspace/STAGE9H_RESULTS")
OUTDIR.mkdir(parents=True, exist_ok=True)

FINAL = OUTDIR / "STAGE9H_VLM_SEMANTIC.json"
SUMMARY = OUTDIR / "SUMMARY.txt"

BATCH = 64
MAXLEN = 512

TASK = (
    "Given a fact-checking claim or search query, retrieve web evidence "
    "passages that help verify or refute it. Prefer passages identifying "
    "the relevant person, object, event, source, date, location, context, "
    "correction, or factual relationship."
)


def canon(u):
    u = (u or "").strip()

    if not u:
        return ""

    p = urlsplit(u)

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

    out = host + path

    if q:
        out += "?" + q

    return out


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
def encode(model, tok, texts, is_query=False):
    outs = []

    for start in range(0, len(texts), BATCH):

        batch = texts[start:start+BATCH]

        if is_query:
            batch = [
                f"Instruct: {TASK}\nQuery: {x}"
                for x in batch
            ]

        inp = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=MAXLEN,
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

        outs.append(
            emb.cpu()
        )

    return torch.cat(
        outs,
        dim=0
    ).numpy()


bm25 = json.loads(
    BM25_FILE.read_text(
        encoding="utf-8"
    )
)

sem = json.loads(
    SEM_FILE.read_text(
        encoding="utf-8"
    )
)["results"]

rr = json.loads(
    RR_FILE.read_text(
        encoding="utf-8"
    )
)["results"]

vlm = json.loads(
    VLM_FILE.read_text(
        encoding="utf-8"
    )
)


print("Loading tokenizer...")

tok = AutoTokenizer.from_pretrained(
    MODEL,
    padding_side="left"
)

print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).cuda().eval()

print("MODEL_DEVICE =", model.device)


claim_results = {}


for n, cid in enumerate(sem, 1):

    srow = sem[cid]
    brow = bm25[cid]
    rrow = rr[cid]

    candidates = srow[
        "top250_qwen"
    ][:250]

    urls = [
        x["url"]
        for x in candidates
    ]

    passages = [
        x["best_passage"]
        for x in candidates
    ]

    parsed = (
        vlm[cid].get("parsed")
        or {}
    )

    exp_queries = [
        str(x).strip()
        for x in (
            parsed.get("queries")
            or []
        )
        if str(x).strip()
    ][:5]


    cue_parts = []

    cue_parts.extend(
        parsed.get("entities")
        or []
    )

    cue_parts.extend(
        parsed.get("ocr_text")
        or []
    )

    cue_parts.extend(
        parsed.get("visual_cues")
        or []
    )

    cue_query = " ".join(
        str(x).strip()
        for x in cue_parts
        if str(x).strip()
    )


    queries = [
        srow["claim"]
    ] + exp_queries

    cue_idx = None

    if cue_query:
        cue_idx = len(queries)
        queries.append(cue_query)


    print(
        f"[CLAIM] {n}/{len(sem)} "
        f"{cid} "
        f"docs={len(passages)} "
        f"queries={len(queries)}",
        flush=True
    )


    doc_emb = encode(
        model,
        tok,
        passages,
        is_query=False
    )

    query_emb = encode(
        model,
        tok,
        queries,
        is_query=True
    )


    scores = doc_emb @ query_emb.T


    rank_maps = []

    for qi in range(
        scores.shape[1]
    ):

        order = np.argsort(
            -scores[:, qi]
        )

        rank_maps.append({
            canon(urls[idx]):
                rank
            for rank, idx
            in enumerate(
                order,
                1
            )
        })


    original_map = rank_maps[0]

    exp_maps = rank_maps[
        1:1+len(exp_queries)
    ]

    cue_map = (
        rank_maps[cue_idx]
        if cue_idx is not None
        else {}
    )


    bm25_map = {
        canon(x["url"]):
            x["rank"]
        for x in brow[
            "top10000"
        ][:250]
    }


    rr_map = {
        canon(x["url"]):
            rank
        for rank, x
        in enumerate(
            rrow["reranked"],
            1
        )
    }


    teacher_urls = srow[
        "teacher_urls"
    ]


    stored_orig = srow[
        "qwen_teacher_ranks"
    ]

    reproduced_orig = [
        original_map.get(
            canon(u)
        )
        for u in teacher_urls
    ]


    vlm_oracle_ranks = []

    for u in teacher_urls:

        cu = canon(u)

        vals = [
            m.get(cu)
            for m in exp_maps
            if m.get(cu) is not None
        ]

        vlm_oracle_ranks.append(
            min(vals)
            if vals
            else None
        )


    print(
        f"[DONE] {cid} "
        f"stored={stored_orig} "
        f"reproduced={reproduced_orig} "
        f"vlm_sem_oracle={vlm_oracle_ranks}",
        flush=True
    )


    claim_results[cid] = {
        "claim":
            srow["claim"],

        "teacher_urls":
            teacher_urls,

        "urls":
            urls,

        "bm25_rank":
            bm25_map,

        "original_semantic_rank":
            original_map,

        "expanded_semantic_ranks":
            exp_maps,

        "cue_semantic_rank":
            cue_map,

        "reranker_rank":
            rr_map,

        "stored_original_teacher_ranks":
            stored_orig,

        "reproduced_original_teacher_ranks":
            reproduced_orig,

        "vlm_semantic_oracle_teacher_ranks":
            vlm_oracle_ranks
    }


    del doc_emb
    del query_emb
    del scores

    gc.collect()
    torch.cuda.empty_cache()


# =====================================================
# Freeze Stage9E settings.
#
# K=10
# BM25=0.25
# ORIGINAL SEM=1.0
# RR=0.5
#
# Only test the additional VLM semantic contribution.
# =====================================================

K = 10

BASE_W_BM25 = 0.25
BASE_W_SEM = 1.0
BASE_W_RR = 0.5

VLM_WEIGHTS = [
    0.0,
    0.10,
    0.25,
    0.50,
    1.0,
    2.0
]

CUE_WEIGHTS = [
    0.0,
    0.10,
    0.25,
    0.50
]


def evaluate(w_vlm, w_cue):

    all_ranks = []
    per_claim = {}


    for cid, row in claim_results.items():

        scored = []

        for url in row["urls"]:

            cu = canon(url)

            score = 0.0


            rb = row[
                "bm25_rank"
            ].get(cu)

            if rb is not None:
                score += (
                    BASE_W_BM25
                    / (K + rb)
                )


            rs = row[
                "original_semantic_rank"
            ].get(cu)

            if rs is not None:
                score += (
                    BASE_W_SEM
                    / (K + rs)
                )


            rrk = row[
                "reranker_rank"
            ].get(cu)

            if rrk is not None:
                score += (
                    BASE_W_RR
                    / (K + rrk)
                )


            # VLM semantic branch:
            # retain strongest expanded-query signal,
            # rather than averaging noisy queries.
            vlm_rr = []

            for rm in row[
                "expanded_semantic_ranks"
            ]:

                r = rm.get(cu)

                if r is not None:
                    vlm_rr.append(
                        1.0 / (K + r)
                    )

            if vlm_rr and w_vlm > 0:

                score += (
                    w_vlm
                    * max(vlm_rr)
                )


            if w_cue > 0:

                rc = row[
                    "cue_semantic_rank"
                ].get(cu)

                if rc is not None:
                    score += (
                        w_cue
                        / (K + rc)
                    )


            scored.append(
                (score, url)
            )


        scored.sort(
            key=lambda x:
                x[0],
            reverse=True
        )


        fmap = {
            canon(url):
                rank
            for rank, (_, url)
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
        all_ranks.extend(ranks)


    n = len(all_ranks)


    def recall(k):
        return sum(
            r is not None
            and r <= k
            for r in all_ranks
        )


    mrr = sum(
        0.0
        if r is None
        else 1.0 / r
        for r in all_ranks
    ) / n


    return {
        "w_vlm":
            w_vlm,

        "w_cue":
            w_cue,

        "r1":
            recall(1),

        "r3":
            recall(3),

        "r5":
            recall(5),

        "r10":
            recall(10),

        "r20":
            recall(20),

        "r50":
            recall(50),

        "mrr":
            mrr,

        "per_claim":
            per_claim
    }


runs = []

for wv in VLM_WEIGHTS:

    for wc in CUE_WEIGHTS:

        runs.append(
            evaluate(
                wv,
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

baseline = next(
    x for x in runs
    if x["w_vlm"] == 0.0
    and x["w_cue"] == 0.0
)


lines = [
    "=== STAGE 9H VLM -> QWEN SEMANTIC ===",
    "",
    "Frozen Stage9E weights:",
    "K=10 BM25=0.25 SEM=1.0 RR=0.5",
    "",
    "RECOMPUTED BASELINE",
    f"Recall@5  = {baseline['r5']}/15",
    f"Recall@10 = {baseline['r10']}/15",
    f"Recall@20 = {baseline['r20']}/15",
    f"MRR = {baseline['mrr']:.6f}",
    "",
    "TOP CONFIGURATIONS",
]


for i, x in enumerate(
    runs[:12],
    1
):

    lines.append(
        f"{i:02d}. "
        f"VLM={x['w_vlm']} "
        f"CUE={x['w_cue']} | "
        f"R@5={x['r5']}/15 "
        f"R@10={x['r10']}/15 "
        f"R@20={x['r20']}/15 "
        f"MRR={x['mrr']:.6f}"
    )


lines += [
    "",
    "=== BEST ===",
    f"VLM weight = {best['w_vlm']}",
    f"CUE weight = {best['w_cue']}",
    f"Recall@1  = {best['r1']}/15",
    f"Recall@3  = {best['r3']}/15",
    f"Recall@5  = {best['r5']}/15",
    f"Recall@10 = {best['r10']}/15",
    f"Recall@20 = {best['r20']}/15",
    f"Recall@50 = {best['r50']}/15",
    f"MRR = {best['mrr']:.6f}",
    "",
    "PER CLAIM BEST RANKS"
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

FINAL.write_text(
    json.dumps(
        {
            "baseline":
                baseline,

            "best":
                best,

            "claims":
                claim_results
        },
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)


print()
print(summary)
print()
print("STAGE9H_COMPLETE")
print("SUMMARY =", SUMMARY)
print("FINAL =", FINAL)
