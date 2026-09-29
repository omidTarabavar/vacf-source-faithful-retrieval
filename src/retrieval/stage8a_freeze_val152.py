#!/usr/bin/env python3
"""
Stage 8A FIXED — freeze Candidate-X directly from Stage-7 prediction files.

Fix:
The earlier script expected `top10_urls` inside STAGE7.json, but the Stage-7
result file on the user's machine does not store that field.  This version
reconstructs the frozen Candidate-X from the source prediction files that
actually produced Stage 7:

  BASE_IMAGE  = stage7_model_store_matrix/predictions/base/<claim>.json
                 -> stores.IMAGE_STORE.ranking
  S2_TEXT     = stage7_model_store_matrix/predictions/siglip2/<claim>.json
                 -> stores.TEXT_STORE.ranking

Candidate-X = first unique BASE_IMAGE URLs, then unique S2_TEXT URLs until 10.

No qrels/gold are read.
"""
import argparse, hashlib, json, os, re, gc, tarfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

BASE_MODEL = "google/siglip-base-patch16-384"
S2_MODEL = "google/siglip2-large-patch16-512"
CHUNK_TOKENIZER = BASE_MODEL
CHUNK_TOKENS = 60
METHOD = "BASE_IMAGE_THEN_S2_TEXT"

def atomic_json(path, obj):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

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
    path = re.sub("/{2,}", "/", p.path or "/")
    if path != "/":
        path = path.rstrip("/")
    q = urlencode(sorted(
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ))
    return urlunsplit(((p.scheme or "https").lower(), net, path, q, ""))

def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def unique_ranked(items, k=10):
    out = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            url = item
            score = None
        else:
            url = item.get("url", "")
            score = item.get("score")
        u = canon(url)
        if not u or u in seen:
            continue
        seen.add(u)
        out.append({"url": u, "score": score})
        if len(out) >= k:
            break
    return out

def load_stage7_ranking(pred_root, model_key, cid, store):
    p = Path(pred_root) / model_key / f"{cid}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    z = json.loads(p.read_text(encoding="utf-8"))
    if z.get("status") != "OK":
        raise RuntimeError(f"Bad Stage7 prediction: {p}")
    ranking = z["stores"][store]["ranking"]
    return unique_ranked(ranking, k=100)

def build_candidate_selection(pred_root, cid):
    base_img = load_stage7_ranking(pred_root, "base", cid, "IMAGE_STORE")
    s2_text = load_stage7_ranking(pred_root, "siglip2", cid, "TEXT_STORE")

    selected = []
    seen = set()

    for x in base_img:
        u = x["url"]
        if u in seen:
            continue
        seen.add(u)
        selected.append({
            "url": u,
            "expert": "BASE_IMAGE",
            "source_score": x.get("score"),
        })
        if len(selected) >= 10:
            return selected

    for x in s2_text:
        u = x["url"]
        if u in seen:
            continue
        seen.add(u)
        selected.append({
            "url": u,
            "expert": "S2_TEXT",
            "source_score": x.get("score"),
        })
        if len(selected) >= 10:
            break

    return selected

def sane_max_len(tok):
    x = getattr(tok, "model_max_length", 64)
    try:
        x = int(x)
    except Exception:
        x = 64
    if x <= 0 or x > 4096:
        x = 64
    return min(x, 64)

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
    def __init__(self, name, batch_size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(
            name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.max_len = sane_max_len(self.tokenizer)

    @torch.inference_mode()
    def text_embeddings(self, texts):
        outs = []
        for i in range(0, len(texts), self.batch_size):
            inp = self.tokenizer(
                texts[i:i+self.batch_size],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            feat = tensor_feat(self.model.get_text_features(**inp)).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outs.append(feat.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1), dtype=np.float32)

    @torch.inference_mode()
    def image_embedding(self, path):
        im = Image.open(path).convert("RGB")
        inp = self.processor(images=im, return_tensors="pt")
        inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        feat = tensor_feat(self.model.get_image_features(**inp)).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat[0].cpu().numpy().astype(np.float32, copy=False)

def common_chunks_for_urls(store_path, target_urls, chunk_tok):
    targets = {canon(u) for u in target_urls}
    by_url = defaultdict(list)

    for e in jsonl(store_path):
        u = canon(e.get("url", ""))
        if u not in targets:
            continue
        pieces = e.get("url2text", [])
        if isinstance(pieces, str):
            pieces = [pieces]
        for piece in pieces:
            ids = chunk_tok.encode(piece, add_special_tokens=False)
            for i in range(0, len(ids), CHUNK_TOKENS):
                s = chunk_tok.decode(
                    ids[i:i+CHUNK_TOKENS],
                    clean_up_tokenization_spaces=True,
                ).strip()
                if s:
                    by_url[u].append(s)
    return by_url

def resolve_images(xp, image_root):
    paths = []
    for ref in xp.get("claim_image_refs") or []:
        p = Path(image_root) / ref
        if p.exists():
            paths.append(str(p))
    if not paths:
        for block in xp.get("image_channel", []):
            p = block.get("image_path")
            if p and Path(p).exists() and p not in paths:
                paths.append(p)
    if not paths:
        raise RuntimeError(f"No claim image for {xp.get('claim_id')}")
    return paths

def top3_chunks(expert, chunks, image_paths):
    if not chunks:
        return []
    te = expert.text_embeddings(chunks)
    best_mean = -1e30
    best_idx = None
    for image_path in image_paths:
        q = expert.image_embedding(image_path)
        scores = te @ q
        idx = np.argsort(scores)[::-1][:3]
        mean = float(np.mean(scores[idx])) if len(idx) else -1e30
        if mean > best_mean:
            best_mean = mean
            best_idx = idx
    if best_idx is None:
        return chunks[:3]
    return [chunks[int(i)] for i in best_idx]

def freeze_xxp(xp):
    out = []
    seen = set()
    for x in xp.get("merged_deduplicated", []):
        u = canon(x.get("url", ""))
        if not u or u in seen:
            continue
        seen.add(u)
        out.append({
            "url": u,
            "text": (x.get("text", "") or "").strip(),
            "images": [],
            "source": x.get("source", ""),
        })
        if len(out) >= 10:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="/workspace/to_v4_retrieval/STATE.json")
    ap.add_argument("--stage7-pred", default="/workspace/stage7_model_store_matrix/predictions")
    ap.add_argument("--xxp", default="/workspace/xxp_stage4_cal64/predictions")
    ap.add_argument("--image-root", default="/workspace/xxp_stage4_cal64/claim_images")
    ap.add_argument("--image-store", default="/workspace/xxp_cal64_official_store/image_related_store_text_train")
    ap.add_argument("--text-store", default="/workspace/xxp_cal64_official_store/text_related_store_text_train")
    ap.add_argument("--out", default="/workspace/stage8_official_eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    rows = list(state["cohorts"]["fresh"])
    if len(rows) != 152:
        raise RuntimeError(f"Expected 152 validation claims, got {len(rows)}")

    # 1) Reconstruct and freeze the exact Candidate-X URL selection from Stage7 source predictions.
    selections = []
    for row in rows:
        cid = row["claim_id"]
        sel = build_candidate_selection(args.stage7_pred, cid)
        selections.append({
            "claim_id": cid,
            "claim_text": row["claim_text"],
            "selected": sel,
        })

    selection_path = out / "CANDIDATE_X_SELECTION.json"
    atomic_json(selection_path, {
        "status": "CANDIDATE_X_URLS_FROZEN",
        "method": METHOD,
        "source": "direct Stage7 model prediction files",
        "qrels_used": False,
        "claims": selections,
    })

    # 2) Hydrate selected URLs to evidence text, expert-consistently.
    chunk_tok = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)
    chunk_tok.model_max_length = int(1e6)

    cand = {}
    for x in selections:
        cand[x["claim_id"]] = {
            "claim_id": x["claim_id"],
            "claim_text": x["claim_text"],
            "method": METHOD,
            "selected": x["selected"],
            "evidence": [None] * len(x["selected"]),
        }

    for expert_label, model_name, store_dir, batch in [
        ("BASE_IMAGE", BASE_MODEL, args.image_store, 128),
        ("S2_TEXT", S2_MODEL, args.text_store, 64),
    ]:
        print(f"Loading {expert_label}: {model_name}", flush=True)
        expert = Expert(model_name, batch)

        for n, x in enumerate(selections, 1):
            target = [z["url"] for z in x["selected"] if z["expert"] == expert_label]
            if not target:
                continue

            cid = x["claim_id"]
            xp_path = Path(args.xxp) / f"{cid}.json"
            xp = json.loads(xp_path.read_text(encoding="utf-8"))
            idx = str(xp["official_store_id"])
            images = resolve_images(xp, args.image_root)

            store_path = Path(store_dir) / f"{idx}.json"
            by_url = common_chunks_for_urls(store_path, target, chunk_tok)

            for pos, z in enumerate(x["selected"]):
                if z["expert"] != expert_label:
                    continue
                u = z["url"]
                chunks = by_url.get(u, [])
                best = top3_chunks(expert, chunks, images)

                if not best:
                    # Fallback is explicit and recorded; do not fabricate evidence.
                    text = ""
                    hydration_status = "NO_TEXT_FOUND"
                else:
                    text = "\n".join(best).strip()
                    hydration_status = "OK"

                cand[cid]["evidence"][pos] = {
                    "url": u,
                    "text": text,
                    "images": [],
                    "expert": expert_label,
                    "hydration_status": hydration_status,
                }

            print(f"{expert_label} {n}/64 {cid}", flush=True)

        del expert
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) Freeze exact saved XxP-full evidence for the same claims.
    cand_out = []
    xxp_out = []
    missing_hydration = []

    for row in rows:
        cid = row["claim_id"]
        if any(e is None for e in cand[cid]["evidence"]):
            raise RuntimeError(f"Internal hydration bug: {cid}")

        for e in cand[cid]["evidence"]:
            if e["hydration_status"] != "OK":
                missing_hydration.append({"claim_id": cid, "url": e["url"], "expert": e["expert"]})

        cand_out.append(cand[cid])

        xp = json.loads((Path(args.xxp) / f"{cid}.json").read_text(encoding="utf-8"))
        xxp_out.append({
            "claim_id": cid,
            "claim_text": xp["claim_text"],
            "method": "OFFICIAL_XXP_FULL_RETRIEVAL",
            "evidence": freeze_xxp(xp),
        })

    frozen = {
        "status": "STAGE8A_FROZEN",
        "cohort": "OfficialVal152 development",
        "n": 64,
        "qrels_used_for_selection_or_hydration": False,
        "candidate_method": METHOD,
        "methods": {
            "XXP_FULL": xxp_out,
            "CANDIDATE_X": cand_out,
        },
        "hydration_warnings": missing_hydration,
    }

    frozen_path = out / "STAGE8A_FROZEN_EVIDENCE.json"
    atomic_json(frozen_path, frozen)

    manifest = {
        "status": "STAGE8A_FROZEN_OK",
        "candidate_method": METHOD,
        "n": 64,
        "hydration_warning_count": len(missing_hydration),
        "selection_sha256": sha256(selection_path),
        "frozen_evidence_sha256": sha256(frozen_path),
        "qrels_used": False,
        "next": "Official Gemma-3-27B evidence evaluation. Do not change Candidate-X.",
    }
    manifest_path = out / "STAGE8A_MANIFEST.json"
    atomic_json(manifest_path, manifest)

    bundle = out / "STAGE8_TRANSFER.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(selection_path, arcname=selection_path.name)
        tar.add(frozen_path, arcname=frozen_path.name)
        tar.add(manifest_path, arcname=manifest_path.name)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("TRANSFER_BUNDLE:", bundle)

if __name__ == "__main__":
    main()
