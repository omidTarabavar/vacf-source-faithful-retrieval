import json, re, gc
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


STAGE7 = Path("/workspace/VAL152_STAGE7/predictions")
FROZEN = Path("/workspace/VAL152_STAGE8A/STAGE8A_FROZEN_EVIDENCE.json")
FULL = Path("/workspace/VAL152_CANDIDATE_Y.json")
STORE = Path("/workspace/Knowledge_Store/val_extracted/converted_datastore/text_related")
OUT = Path("/workspace/VAL152_ABLATIONS.json")

BASE_TOKENIZER = "google/siglip-base-patch16-384"
S2_MODEL = "google/siglip2-large-patch16-512"
RESTORE_TOKENIZER = "google/gemma-3-27b-it"

CHUNK_TOKENS = 60
BUDGET = 1200


def canon(u):
    u = (u or "").strip()
    if not u:
        return ""
    try:
        p = urlsplit(u)
    except Exception:
        return u.rstrip("/")
    net = p.netloc.lower()
    if net.startswith("www."):
        net = net[4:]
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if path != "/":
        path = path.rstrip("/")
    q = urlencode(sorted(
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ))
    return urlunsplit(((p.scheme or "https").lower(), net, path, q, ""))


def canon_raw(u):
    u = (u or "").strip()
    if not u:
        return ""
    try:
        p = urlsplit(u)
    except Exception:
        return u.rstrip("/").lower()
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if path != "/":
        path = path.rstrip("/")
    q = urlencode(sorted(
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ))
    key = host + path
    if q:
        key += "?" + q
    return key


def unique_top10(ranking):
    out, seen = [], set()
    for x in ranking:
        u = canon(x.get("url", ""))
        if not u or u in seen:
            continue
        seen.add(u)
        out.append({"url": u, "score": x.get("score")})
        if len(out) == 10:
            break
    return out


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunks_for_urls(store_path, urls, chunk_tok):
    targets = {canon(u) for u in urls}
    by_url = defaultdict(list)

    for e in jsonl(store_path):
        u = canon(e.get("url", ""))
        if u not in targets:
            continue
        pieces = e.get("url2text", [])
        if isinstance(pieces, str):
            pieces = [pieces]
        for piece in pieces:
            ids = chunk_tok.encode(str(piece), add_special_tokens=False)
            for i in range(0, len(ids), CHUNK_TOKENS):
                s = chunk_tok.decode(
                    ids[i:i+CHUNK_TOKENS],
                    clean_up_tokenization_spaces=True
                ).strip()
                if s:
                    by_url[u].append(s)
    return by_url


def tensor_feat(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
        return x.last_hidden_state[:, 0]
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    raise TypeError(type(x))


class Expert:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(S2_MODEL)
        self.processor = AutoProcessor.from_pretrained(S2_MODEL)
        self.model = AutoModel.from_pretrained(
            S2_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

        ml = getattr(self.tokenizer, "model_max_length", 64)
        try:
            ml = int(ml)
        except Exception:
            ml = 64
        if ml <= 0 or ml > 4096:
            ml = 64
        self.max_len = min(ml, 64)

    @torch.inference_mode()
    def text_embeddings(self, texts, batch=64):
        outs = []
        for i in range(0, len(texts), batch):
            inp = self.tokenizer(
                texts[i:i+batch],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            ).to(self.device)
            feat = tensor_feat(self.model.get_text_features(**inp)).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outs.append(feat.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outs, axis=0)

    @torch.inference_mode()
    def image_embedding(self, path):
        im = Image.open(path).convert("RGB")
        inp = self.processor(images=im, return_tensors="pt")
        inp = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inp.items()
        }
        feat = tensor_feat(self.model.get_image_features(**inp)).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat[0].cpu().numpy().astype(np.float32, copy=False)

    def top3(self, chunks, image_paths):
        if not chunks:
            return []
        te = self.text_embeddings(chunks)

        best_mean = -1e30
        best_idx = None

        for image_path in image_paths:
            q = self.image_embedding(image_path)
            scores = te @ q
            idx = np.argsort(scores)[::-1][:3]
            mean = float(np.mean(scores[idx]))
            if mean > best_mean:
                best_mean = mean
                best_idx = idx

        return [chunks[int(i)] for i in best_idx] if best_idx is not None else []


def raw_pieces(path, url):
    target = canon_raw(url)
    out = []
    for x in jsonl(path):
        if canon_raw(x.get("url", "")) != target:
            continue
        vals = x.get("url2text", [])
        if isinstance(vals, str):
            vals = [vals]
        out.extend(str(v).strip() for v in vals if str(v).strip())
    return out


def norm_tokens(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def piece_score(chunk, piece):
    a = set(norm_tokens(chunk))
    b = set(norm_tokens(piece))
    if not a:
        return 0.0
    return len(a & b) / len(a)


restore_tok = AutoTokenizer.from_pretrained(RESTORE_TOKENIZER)


def restore(pieces, old_text):
    anchors = set()

    old_chunks = [
        x.strip() for x in (old_text or "").splitlines()
        if x.strip()
    ]

    for chunk in old_chunks:
        scores = [piece_score(chunk, p) for p in pieces]
        if scores:
            anchors.add(max(range(len(scores)), key=scores.__getitem__))

    priority = {}

    for i in anchors:
        priority[i] = 0
        for d in (-1, 1):
            j = i + d
            if 0 <= j < len(pieces):
                priority[j] = min(priority.get(j, 99), 1)
        for d in (-2, 2):
            j = i + d
            if 0 <= j < len(pieces):
                priority[j] = min(priority.get(j, 99), 2)

    for i in range(min(3, len(pieces))):
        priority[i] = min(priority.get(i, 99), 2)

    chosen = []
    used = 0

    for i in sorted(priority, key=lambda x: (priority[x], x)):
        text = pieces[i]
        ids = restore_tok.encode(text, add_special_tokens=False)
        remaining = BUDGET - used

        if remaining <= 0:
            break

        if len(ids) <= remaining:
            chosen.append((i, text))
            used += len(ids)

        elif priority[i] == 0 and remaining >= 64:
            clipped = restore_tok.decode(
                ids[:remaining],
                skip_special_tokens=True
            ).strip()
            if clipped:
                chosen.append((i, clipped))
            used = BUDGET
            break

    chosen.sort(key=lambda x: x[0])
    return "\n\n".join(x[1] for x in chosen).strip()


frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
full = json.loads(FULL.read_text(encoding="utf-8"))

full_rows = full["methods"]["CANDIDATE_X"]
frozen_rows = frozen["methods"]["CANDIDATE_X"]

assert len(full_rows) == len(frozen_rows) == 152


# ------------------------------------------------------------
# 1) Visual-only + identical source-faithful restoration
# ------------------------------------------------------------
visual_rows = []

for row in full_rows:
    evid = [
        dict(e)
        for e in row["evidence"]
        if e.get("expert") == "BASE_IMAGE"
    ]

    visual_rows.append({
        **row,
        "method": "VISUAL_ONLY_RESTORED",
        "evidence": evid,
    })


# ------------------------------------------------------------
# 2) Contextual-only top-10 from independent S2_TEXT ranking
#    then identical hydration + restoration
# ------------------------------------------------------------
print("Loading contextual expert:", S2_MODEL, flush=True)

chunk_tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
chunk_tok.model_max_length = int(1e6)

expert = Expert()
context_rows = []

for pos, base_row in enumerate(full_rows, 1):
    cid = base_row["claim_id"]
    idx = str(int(cid.rsplit("_", 1)[1]))

    pred_path = STAGE7 / "siglip2" / f"{cid}.json"
    pred = json.loads(pred_path.read_text(encoding="utf-8"))

    ranking = pred["stores"]["TEXT_STORE"]["ranking"]
    selected = unique_top10(ranking)

    image_paths = [
        p for p in pred.get("images", [])
        if Path(p).exists()
    ]
    if not image_paths:
        raise RuntimeError(f"No claim image: {cid}")

    store_path = STORE / "text_related_store_text_val" / f"{idx}.json"

    by_url = chunks_for_urls(
        store_path,
        [x["url"] for x in selected],
        chunk_tok
    )

    evidence = []

    for item in selected:
        u = item["url"]
        chunks = by_url.get(u, [])

        best = expert.top3(chunks, image_paths)

        if not best:
            raise RuntimeError(f"No hydrated chunks: {cid} {u}")

        old_text = "\n".join(best).strip()

        pieces = raw_pieces(store_path, u)
        if not pieces:
            raise RuntimeError(f"No raw pieces: {cid} {u}")

        restored_text = restore(pieces, old_text)
        if not restored_text:
            raise RuntimeError(f"Empty restored text: {cid} {u}")

        evidence.append({
            "url": u,
            "text": restored_text,
            "images": [],
            "expert": "S2_TEXT",
            "hydration_status": "OK",
            "localizer": "source_faithful_context_restoration",
            "budget_tokens": BUDGET,
        })

    context_rows.append({
        "claim_id": cid,
        "claim_text": base_row["claim_text"],
        "method": "CONTEXTUAL_ONLY_RESTORED",
        "evidence": evidence,
    })

    print(
        f"CONTEXT {pos}/152 {cid} evidence={len(evidence)}",
        flush=True
    )


del expert
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


out = {
    "status": "VAL152_ABLATIONS_READY",
    "cohort": "Official AVerImaTeC Validation 152",
    "n": 152,
    "qrels_used_for_selection_or_hydration": False,
    "methods": {
        "XXP_FULL": full["methods"]["XXP_FULL"],
        "VISUAL_ONLY": visual_rows,
        "CONTEXTUAL_ONLY": context_rows,
        "PVCF_WO_RESTORATION": frozen_rows,
        "PVCF_FULL": full_rows,
    },
}

OUT.write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("\nREADY:", OUT)

for method, rows in out["methods"].items():
    lens = [len(r["evidence"]) for r in rows]
    print(
        method,
        "claims=", len(rows),
        "min=", min(lens),
        "max=", max(lens),
        "avg=", sum(lens)/len(lens),
        "zero=", sum(x == 0 for x in lens),
    )
