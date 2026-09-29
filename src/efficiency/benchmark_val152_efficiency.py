import argparse
import json
import os
import re
import sys
import time
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import torch
from PIL import Image

VAL = Path(
    "/workspace/AVerImaTec_Shared_Task/"
    "data/data_clean/split_data/val.json"
)

IMAGE_ROOT = Path(
    "/workspace/AVerImaTec_Shared_Task/data/data_clean/images"
)

STORE_ROOT = Path(
    "/workspace/Knowledge_Store/val_extracted/"
    "converted_datastore/text_related"
)

IMAGE_STORE = STORE_ROOT / "image_related_store_text_val"
TEXT_STORE  = STORE_ROOT / "text_related_store_text_val"

BASE_MODEL = "/workspace/FINAL_HF_CACHE/hub/models--google--siglip-base-patch16-384/snapshots/41aec1c83b32e0a6fca20ad88ba058aa5b5ea394"
S2_MODEL   = "/workspace/FINAL_HF_CACHE/hub/models--google--siglip2-large-patch16-512/snapshots/49488218e80259885f3be61d7a9455faf833b7a8"

GEMMA_TOK = (
    "/workspace/FINAL_HF_CACHE/hub/"
    "models--google--gemma-3-27b-it/"
    "snapshots/005ad3404e59d6023443cb575daa05336842228a"
)

CHUNK_N = 60
BUDGET = 1200

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True, choices=["xxp","vacf"])
parser.add_argument("--count", type=int, default=10)
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()

rows = json.loads(VAL.read_text(encoding="utf-8"))
assert len(rows) == 152

# One warm-up claim, then exactly `count` measured claims.
if args.count >= len(rows):
    measure_ids = list(range(len(rows)))
else:
    measure_ids = list(range(1, 1 + args.count))

assert measure_ids
assert max(measure_ids) < len(rows)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak():
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
        }

    sync()

    return {
        "allocated_gb":
            torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gb":
            torch.cuda.max_memory_reserved() / 1024**3,
    }


def summarize(times):
    a=np.asarray(times,dtype=float)
    return {
        "n": int(len(a)),
        "mean_sec": float(np.mean(a)),
        "median_sec": float(np.median(a)),
        "p95_sec": float(np.percentile(a,95)),
        "min_sec": float(np.min(a)),
        "max_sec": float(np.max(a)),
        "total_sec": float(np.sum(a)),
    }


print("METHOD:",args.method)
print("warmup claim:",args.warmup)
print("measured claims:",measure_ids)
print("CUDA:",torch.cuda.is_available())

for idx in [args.warmup] + measure_ids:
    assert (IMAGE_ROOT / Path(rows[idx]["claim_images"][0]).name).exists()

if args.check_only:
    print("CHECK_OK")
    raise SystemExit(0)


# =========================================================
# XxP
# =========================================================

if args.method == "xxp":

    sys.path.insert(0,"/workspace/XXP_OFFICIAL")

    from GenerationLLMHandler import GenerationLLMHandler
    from EvidenceRetriever import EvidenceRetriever
    from run_pipeline import parse_questions
    from sentence_transformers import SentenceTransformer

    load_t0=time.perf_counter()

    gen=GenerationLLMHandler(
        image_root=str(IMAGE_ROOT),
        model_name="/workspace/.hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    )

    emb=SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    retriever=EvidenceRetriever(
        text_store_dir=str(TEXT_STORE),
        image_store_dir=str(IMAGE_STORE),
        emb_model=emb,
        siglip_model_name=BASE_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    sync()
    model_load_sec=time.perf_counter()-load_t0

    print("MODELS READY sec=",model_load_sec,flush=True)

    def run_claim(idx):

        ex=rows[idx]

        claim=ex["claim_text"]

        claim_imgs=[
            os.path.basename(str(x))
            for x in (ex.get("claim_images") or [])
        ]

        img_paths=[
            str(IMAGE_ROOT / x)
            for x in claim_imgs
        ]

        # --------------------
        # Qwen question generation
        # --------------------

        sync()
        t0=time.perf_counter()

        raw_q=gen.generate_question(
            claim_text=claim,
            image_refs=claim_imgs,
            max_tokens=512,
            temperature=0.2,
        )

        sync()
        q_sec=time.perf_counter()-t0

        (
            all_questions,
            text_questions,
            image_questions,
            image_q_with_indices,
        )=parse_questions(raw_q)

        # --------------------
        # Actual retrieval
        # --------------------

        sync()
        t0=time.perf_counter()

        n_hits=0

        if text_questions:

            text_query=claim+" "+" ".join(text_questions)

            ev,urls,scores=retriever.retrieve_evidences(
                claim_text=claim,
                claim_image=None,
                claim_id=str(idx),
                question=text_query,
                strategy="Text-search",
            )

            n_hits += len(ev)

        if image_q_with_indices and img_paths:

            for iq in image_q_with_indices:

                ip=None

                for k in iq["indices"]:
                    pos=k-1
                    if 0 <= pos < len(img_paths):
                        ip=img_paths[pos]
                        break

                if ip is None:
                    continue

                img_query=claim+" "+iq["text"]

                hits=retriever.retrieve_evidences(
                    claim_text=claim,
                    claim_image=ip,
                    claim_id=str(idx),
                    question=img_query,
                    strategy="Image-search",
                )

                n_hits += len(hits)

        sync()
        retrieval_sec=time.perf_counter()-t0

        return {
            "question_generation_sec":q_sec,
            "retrieval_sec":retrieval_sec,
            "total_sec":q_sec+retrieval_sec,
            "questions":len(all_questions),
            "retrieved_hits":n_hits,
            "claim_images":len(claim_imgs),
        }

    print("WARMUP...",flush=True)
    _=run_claim(args.warmup)
    sync()
    print("WARMUP DONE",flush=True)

    results=[]

    for pos,idx in enumerate(measure_ids,1):

        reset_peak()

        r=run_claim(idx)
        r["claim_index"]=idx
        r["claim_id"]=f"averi_val_{idx:04d}"
        r["peak_cuda"]=peak()

        results.append(r)

        print(
            f"[XXP] {pos}/{len(measure_ids)} "
            f"id={idx} total={r['total_sec']:.3f}s "
            f"qgen={r['question_generation_sec']:.3f}s "
            f"retr={r['retrieval_sec']:.3f}s "
            f"peak={r['peak_cuda']['reserved_gb']:.2f}GB",
            flush=True,
        )

    out={
        "method":"XxP",
        "benchmark_type":"warm evidence-acquisition latency",
        "model_load_sec":model_load_sec,
        "warmup_claim":args.warmup,
        "measured_claims":measure_ids,
        "per_claim":results,
        "summary":{
            "total":summarize([x["total_sec"] for x in results]),
            "question_generation":summarize([
                x["question_generation_sec"] for x in results
            ]),
            "retrieval":summarize([
                x["retrieval_sec"] for x in results
            ]),
            "peak_reserved_gb":max(
                x["peak_cuda"]["reserved_gb"] for x in results
            ),
            "peak_allocated_gb":max(
                x["peak_cuda"]["allocated_gb"] for x in results
            ),
        },
    }


# =========================================================
# VACF
# =========================================================

else:

    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    def jl(path):
        with open(path,encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def canon(u):
        u=(u or "").strip()
        if not u:
            return ""

        try:
            p=urlsplit(u)
        except Exception:
            return u.rstrip("/")

        net=p.netloc.lower()

        if net.startswith("www."):
            net=net[4:]

        path=re.sub("/{2,}","/",p.path or "/")

        if path != "/":
            path=path.rstrip("/")

        q=urlencode(sorted(
            (k,v)
            for k,v in parse_qsl(
                p.query,
                keep_blank_values=True
            )
            if not k.lower().startswith("utm_")
        ))

        return urlunsplit((
            (p.scheme or "https").lower(),
            net,
            path,
            q,
            "",
        ))

    def canon_raw(u):
        u=(u or "").strip()
        if not u:
            return ""

        p=urlsplit(u)

        host=p.netloc.lower()

        if host.startswith("www."):
            host=host[4:]

        path=re.sub(r"/{2,}","/",p.path or "/")

        if path != "/":
            path=path.rstrip("/")

        query=urlencode(sorted(
            (k,v)
            for k,v in parse_qsl(
                p.query,
                keep_blank_values=True
            )
            if not k.lower().startswith("utm_")
        ))

        key=host+path

        if query:
            key+="?"+query

        return key

    def feature(x):
        if torch.is_tensor(x):
            return x
        if getattr(x,"pooler_output",None) is not None:
            return x.pooler_output
        if getattr(x,"last_hidden_state",None) is not None:
            return x.last_hidden_state[:,0]
        if isinstance(x,(tuple,list)) and x and torch.is_tensor(x[0]):
            return x[0]
        raise TypeError(type(x))

    class Expert:
        def __init__(self,name,batch):
            self.dev="cuda" if torch.cuda.is_available() else "cpu"
            self.batch=batch

            self.tok=AutoTokenizer.from_pretrained(name)
            self.proc=AutoProcessor.from_pretrained(name)

            self.model=AutoModel.from_pretrained(
                name,
                torch_dtype=(
                    torch.float16
                    if self.dev=="cuda"
                    else torch.float32
                ),
            ).to(self.dev).eval()

            ml=getattr(self.tok,"model_max_length",64)

            try:
                ml=int(ml)
            except Exception:
                ml=64

            self.ml=(
                64
                if ml <= 0 or ml > 4096
                else min(ml,64)
            )

        @torch.inference_mode()
        def txt(self,texts):

            out=[]

            for i in range(0,len(texts),self.batch):

                x=self.tok(
                    texts[i:i+self.batch],
                    padding="max_length",
                    truncation=True,
                    max_length=self.ml,
                    return_tensors="pt",
                ).to(self.dev)

                z=feature(
                    self.model.get_text_features(**x)
                ).float()

                z=z/z.norm(
                    dim=-1,
                    keepdim=True
                ).clamp_min(1e-12)

                out.append(
                    z.cpu().numpy().astype("float32")
                )

            if not out:
                return np.zeros((0,1),"float32")

            return np.concatenate(out)

        @torch.inference_mode()
        def img(self,path):

            im=Image.open(path).convert("RGB")

            x=self.proc(
                images=im,
                return_tensors="pt"
            )

            x={
                k:(v.to(self.dev) if torch.is_tensor(v) else v)
                for k,v in x.items()
            }

            z=feature(
                self.model.get_image_features(**x)
            ).float()

            z=z/z.norm(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-12)

            return z[0].cpu().numpy().astype("float32")

    chunk_tok=None
    gemma_tok=None

    def common_chunks(path):

        texts=[]
        urls=[]

        for e in jl(path):

            u=e.get("url","")
            parts=e.get("url2text",[])

            if isinstance(parts,str):
                parts=[parts]

            for part in parts:

                if not part:
                    continue

                ids=chunk_tok.encode(
                    part,
                    add_special_tokens=False
                )

                for j in range(0,len(ids),CHUNK_N):

                    s=chunk_tok.decode(
                        ids[j:j+CHUNK_N],
                        clean_up_tokenization_spaces=True
                    ).strip()

                    if s:
                        texts.append(s)
                        urls.append(u)

        return texts,urls

    def rank(expert,texts,urls,images):

        if not texts:
            return []

        emb=expert.txt(texts)

        best={}

        for ip in images:

            q=expert.img(ip)
            sc=emb@q

            by=defaultdict(list)

            for u,s in zip(urls,sc):
                by[u].append(float(s))

            for u,ss in by.items():

                score=float(
                    np.mean(
                        sorted(ss,reverse=True)[:3]
                    )
                )

                if u not in best or score > best[u]:
                    best[u]=score

        return sorted(
            best.items(),
            key=lambda x:x[1],
            reverse=True
        )

    def select_urls(base_rank,s2_rank):

        out=[]
        seen=set()

        for expert,rr in [
            ("BASE_IMAGE",base_rank[:100]),
            ("S2_TEXT",s2_rank[:100]),
        ]:

            for u,s in rr:

                cu=canon(u)

                if not cu or cu in seen:
                    continue

                seen.add(cu)

                out.append({
                    "url":cu,
                    "expert":expert,
                    "source_score":float(s),
                })

                if len(out)>=10:
                    return out

        return out

    def chunks_for_urls(path,urls):

        targets={canon(u) for u in urls}
        by=defaultdict(list)

        for e in jl(path):

            u=canon(e.get("url",""))

            if u not in targets:
                continue

            parts=e.get("url2text",[])

            if isinstance(parts,str):
                parts=[parts]

            for part in parts:

                ids=chunk_tok.encode(
                    part,
                    add_special_tokens=False
                )

                for j in range(0,len(ids),CHUNK_N):

                    s=chunk_tok.decode(
                        ids[j:j+CHUNK_N],
                        clean_up_tokenization_spaces=True
                    ).strip()

                    if s:
                        by[u].append(s)

        return by

    def top3(expert,chunks,images):

        if not chunks:
            return []

        te=expert.txt(chunks)

        best_mean=-1e30
        best_idx=None

        for ip in images:

            q=expert.img(ip)
            scores=te@q
            idx=np.argsort(scores)[::-1][:3]

            m=float(np.mean(scores[idx]))

            if m > best_mean:
                best_mean=m
                best_idx=idx

        return [
            chunks[int(i)]
            for i in best_idx
        ]

    def norm_tokens(s):
        return re.findall(
            r"[a-z0-9]+",
            (s or "").lower()
        )

    def piece_score(chunk,piece):

        a=set(norm_tokens(chunk))
        b=set(norm_tokens(piece))

        if not a:
            return 0.0

        return len(a & b)/len(a)

    def raw_pieces(path,url):

        target=canon_raw(url)
        pieces=[]

        for x in jl(path):

            if canon_raw(x.get("url","")) != target:
                continue

            vals=x.get("url2text",[])

            if isinstance(vals,str):
                vals=[vals]

            pieces.extend([
                str(v).strip()
                for v in vals
                if str(v).strip()
            ])

        return pieces

    def pack_context(pieces,anchors):

        priority={}

        for i in anchors:

            priority[i]=0

            for d in (-1,1):
                j=i+d
                if 0 <= j < len(pieces):
                    priority[j]=min(
                        priority.get(j,99),1
                    )

            for d in (-2,2):
                j=i+d
                if 0 <= j < len(pieces):
                    priority[j]=min(
                        priority.get(j,99),2
                    )

        for i in range(min(3,len(pieces))):
            priority[i]=min(
                priority.get(i,99),2
            )

        chosen=[]
        used=0

        for i in sorted(
            priority,
            key=lambda x:(priority[x],x)
        ):

            text=pieces[i]

            ids=gemma_tok.encode(
                text,
                add_special_tokens=False
            )

            remaining=BUDGET-used

            if remaining <= 0:
                break

            if len(ids) <= remaining:

                chosen.append((i,text))
                used += len(ids)

            elif priority[i] == 0 and remaining >= 64:

                clipped=gemma_tok.decode(
                    ids[:remaining],
                    skip_special_tokens=True
                ).strip()

                if clipped:
                    chosen.append((i,clipped))

                break

        chosen.sort(key=lambda x:x[0])

        return "\n\n".join(
            x[1] for x in chosen
        ).strip()

    load_t0=time.perf_counter()

    chunk_tok=AutoTokenizer.from_pretrained(BASE_MODEL)
    chunk_tok.model_max_length=int(1e6)

    gemma_tok=AutoTokenizer.from_pretrained(
        GEMMA_TOK,
        local_files_only=True,
    )

    base=Expert(BASE_MODEL,128)
    s2=Expert(S2_MODEL,64)

    sync()
    model_load_sec=time.perf_counter()-load_t0

    print("MODELS READY sec=",model_load_sec,flush=True)

    def run_claim(idx):

        ex=rows[idx]

        images=[
            str(
                IMAGE_ROOT /
                os.path.basename(str(x))
            )
            for x in (ex.get("claim_images") or [])
        ]

        ipath=IMAGE_STORE/f"{idx}.json"
        tpath=TEXT_STORE/f"{idx}.json"

        # -------------------
        # BASE_IMAGE ranking
        # -------------------

        sync()
        t0=time.perf_counter()

        texts,urls=common_chunks(ipath)

        sync()
        base_prepare_sec=time.perf_counter()-t0

        sync()
        t0=time.perf_counter()

        br=rank(base,texts,urls,images)

        sync()
        base_rank_sec=time.perf_counter()-t0
        base_sec=base_prepare_sec+base_rank_sec

        del texts,urls

        # -------------------
        # S2_TEXT ranking
        # -------------------

        sync()
        t0=time.perf_counter()

        texts,urls=common_chunks(tpath)

        sync()
        s2_prepare_sec=time.perf_counter()-t0

        sync()
        t0=time.perf_counter()

        sr=rank(s2,texts,urls,images)

        sync()
        s2_rank_sec=time.perf_counter()-t0
        s2_sec=s2_prepare_sec+s2_rank_sec

        del texts,urls

        # -------------------
        # Fusion + localization +
        # source-faithful restoration
        # -------------------

        sync()
        t0=time.perf_counter()

        selected=select_urls(br,sr)

        evid=[]

        for expert_name,expert,store in [
            ("BASE_IMAGE",base,ipath),
            ("S2_TEXT",s2,tpath),
        ]:

            targets=[
                x["url"]
                for x in selected
                if x["expert"]==expert_name
            ]

            if not targets:
                continue

            by=chunks_for_urls(store,targets)

            for z in selected:

                if z["expert"] != expert_name:
                    continue

                old=top3(
                    expert,
                    by.get(z["url"],[]),
                    images
                )

                pieces=raw_pieces(
                    store,
                    z["url"]
                )

                anchors=set()

                for chunk in old:

                    scores=[
                        piece_score(chunk,p)
                        for p in pieces
                    ]

                    if scores:
                        anchors.add(
                            max(
                                range(len(scores)),
                                key=scores.__getitem__
                            )
                        )

                restored=pack_context(
                    pieces,
                    anchors
                )

                evid.append({
                    "url":z["url"],
                    "expert":expert_name,
                    "text":restored,
                })

        # restore final original selected ordering
        evmap={
            (x["url"],x["expert"]):x
            for x in evid
        }

        evid=[
            evmap[(z["url"],z["expert"])]
            for z in selected
        ]

        sync()
        post_sec=time.perf_counter()-t0

        return {
            "base_image_prepare_sec":base_prepare_sec,
            "base_image_rank_sec":base_rank_sec,
            "base_image_sec":base_sec,
            "s2_text_prepare_sec":s2_prepare_sec,
            "s2_text_rank_sec":s2_rank_sec,
            "s2_text_sec":s2_sec,
            "fusion_hydration_restoration_sec":post_sec,
            "total_sec":base_sec+s2_sec+post_sec,
            "selected_urls":len(selected),
            "evidence":len(evid),
            "claim_images":len(images),
        }

    print("WARMUP...",flush=True)
    _=run_claim(args.warmup)
    sync()
    print("WARMUP DONE",flush=True)

    results=[]

    for pos,idx in enumerate(measure_ids,1):

        reset_peak()

        r=run_claim(idx)

        r["claim_index"]=idx
        r["claim_id"]=f"averi_val_{idx:04d}"
        r["peak_cuda"]=peak()

        results.append(r)

        print(
            f"[VACF] {pos}/{len(measure_ids)} "
            f"id={idx} total={r['total_sec']:.3f}s "
            f"base={r['base_image_sec']:.3f}s "
            f"s2={r['s2_text_sec']:.3f}s "
            f"post={r['fusion_hydration_restoration_sec']:.3f}s "
            f"peak={r['peak_cuda']['reserved_gb']:.2f}GB",
            flush=True,
        )

    out={
        "method":"VACF",
        "benchmark_type":"warm evidence-acquisition latency",
        "model_load_sec":model_load_sec,
        "warmup_claim":args.warmup,
        "measured_claims":measure_ids,
        "per_claim":results,
        "summary":{
            "total":summarize([
                x["total_sec"] for x in results
            ]),
            "base_image":summarize([
                x["base_image_sec"] for x in results
            ]),
            "base_image_prepare":summarize([
                x["base_image_prepare_sec"] for x in results
            ]),
            "base_image_rank":summarize([
                x["base_image_rank_sec"] for x in results
            ]),
            "s2_text":summarize([
                x["s2_text_sec"] for x in results
            ]),
            "s2_text_prepare":summarize([
                x["s2_text_prepare_sec"] for x in results
            ]),
            "s2_text_rank":summarize([
                x["s2_text_rank_sec"] for x in results
            ]),
            "fusion_hydration_restoration":summarize([
                x["fusion_hydration_restoration_sec"]
                for x in results
            ]),
            "peak_reserved_gb":max(
                x["peak_cuda"]["reserved_gb"]
                for x in results
            ),
            "peak_allocated_gb":max(
                x["peak_cuda"]["allocated_gb"]
                for x in results
            ),
        },
    }


Path(args.out).write_text(
    json.dumps(
        out,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8",
)

print("\n========== SUMMARY ==========")
print(json.dumps(out["summary"],indent=2))
print("MODEL LOAD SEC:",model_load_sec)
print("WROTE:",args.out)
