#!/usr/bin/env python3
import argparse,json,os,re,time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel,AutoProcessor,AutoTokenizer

MODELS={'base':'google/siglip-base-patch16-384','siglip2':'google/siglip2-large-patch16-512'}
CHUNK_TOK='google/siglip-base-patch16-384'; CHUNK_N=60

def jl(p):
  with open(p,encoding='utf-8') as f:
    for x in f:
      if x.strip(): yield json.loads(x)

def atomic(p,o):
  p=Path(p); t=Path(str(p)+'.tmp'); t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(t,p)

def feature(x):
  if torch.is_tensor(x): return x
  if getattr(x,'pooler_output',None) is not None: return x.pooler_output
  if getattr(x,'last_hidden_state',None) is not None: return x.last_hidden_state[:,0]
  if isinstance(x,(tuple,list)) and x and torch.is_tensor(x[0]): return x[0]
  raise TypeError(type(x))

def common_chunks(path,tok):
  texts=[]; urls=[]
  for e in jl(path):
    u=e.get('url',''); parts=e.get('url2text',[])
    if isinstance(parts,str): parts=[parts]
    for part in parts:
      if not part: continue
      ids=tok.encode(part,add_special_tokens=False)
      for i in range(0,len(ids),CHUNK_N):
        s=tok.decode(ids[i:i+CHUNK_N],clean_up_tokenization_spaces=True).strip()
        if s: texts.append(s); urls.append(u)
  return texts,urls

class E:
  def __init__(self,name,batch):
    self.dev='cuda' if torch.cuda.is_available() else 'cpu'; self.batch=batch
    self.tok=AutoTokenizer.from_pretrained(name); self.proc=AutoProcessor.from_pretrained(name)
    self.model=AutoModel.from_pretrained(name,torch_dtype=torch.float16 if self.dev=='cuda' else torch.float32).to(self.dev).eval()
    ml=getattr(self.tok,'model_max_length',64)
    try: ml=int(ml)
    except: ml=64
    self.ml=64 if ml<=0 or ml>4096 else min(ml,64)
  @torch.inference_mode()
  def txt(self,texts):
    out=[]
    for i in range(0,len(texts),self.batch):
      x=self.tok(texts[i:i+self.batch],padding='max_length',truncation=True,max_length=self.ml,return_tensors='pt').to(self.dev)
      z=feature(self.model.get_text_features(**x)).float(); z=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-12); out.append(z.cpu().numpy().astype('float32'))
    return np.concatenate(out) if out else np.zeros((0,1),'float32')
  @torch.inference_mode()
  def img(self,p):
    im=Image.open(p).convert('RGB'); x=self.proc(images=im,return_tensors='pt'); x={k:(v.to(self.dev) if torch.is_tensor(v) else v) for k,v in x.items()}
    z=feature(self.model.get_image_features(**x)).float(); z=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-12); return z[0].cpu().numpy().astype('float32')

def rank(expert,texts,urls,images):
  if not texts:return []
  emb=expert.txt(texts); best={}
  for ip in images:
    q=expert.img(ip)
    if emb.shape[1]!=q.shape[0]: raise RuntimeError(f'dim mismatch {emb.shape} {q.shape}')
    sc=emb@q; by=defaultdict(list)
    for u,s in zip(urls,sc): by[u].append(float(s))
    for u,ss in by.items():
      s=float(np.mean(sorted(ss,reverse=True)[:3]))
      if u not in best or s>best[u]: best[u]=s
  return sorted(best.items(),key=lambda x:x[1],reverse=True)

def images_for(xp,root):
  out=[]
  for r in xp.get('claim_image_refs') or []:
    p=Path(root)/r
    if p.exists(): out.append(str(p))
  if not out:
    for b in xp.get('image_channel',[]):
      p=b.get('image_path')
      if p and Path(p).exists() and p not in out: out.append(p)
  if not out: raise RuntimeError('no claim image '+xp.get('claim_id',''))
  return out

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model-key',choices=MODELS,required=True); ap.add_argument('--batch',type=int,default=64)
  ap.add_argument('--v4',default='/workspace/to_v4_retrieval'); ap.add_argument('--xxp',default='/workspace/xxp_stage4_cal64/predictions'); ap.add_argument('--image-root',default='/workspace/xxp_stage4_cal64/claim_images')
  ap.add_argument('--istore',default='/workspace/xxp_cal64_official_store/image_related_store_text_train'); ap.add_argument('--tstore',default='/workspace/xxp_cal64_official_store/text_related_store_text_train'); ap.add_argument('--out',default='/workspace/stage7_model_store_matrix'); a=ap.parse_args()
  model=MODELS[a.model_key]; out=Path(a.out)/'predictions'/a.model_key; out.mkdir(parents=True,exist_ok=True)
  st=json.loads((Path(a.v4)/'STATE.json').read_text()); rows=st['cohorts']['fresh']; ctok=AutoTokenizer.from_pretrained(CHUNK_TOK); ctok.model_max_length=int(1e6); e=E(model,a.batch)
  print(json.dumps({'loaded':model,'device':e.dev,'batch':a.batch}),flush=True)
  for n,r in enumerate(rows,1):
    cid=r['claim_id']; dst=out/f'{cid}.json'
    if dst.exists():
      try:
        z=json.loads(dst.read_text());
        if z.get('status')=='OK' and z.get('model')==model: print('cached',n,'/64',cid,flush=True); continue
      except: pass
    xp=json.loads((Path(a.xxp)/f'{cid}.json').read_text()); idx=str(xp['official_store_id']); ims=images_for(xp,a.image_root); t0=time.time(); stores={}
    for sn,sd in [('IMAGE_STORE',a.istore),('TEXT_STORE',a.tstore)]:
      texts,urls=common_chunks(Path(sd)/f'{idx}.json',ctok); rr=rank(e,texts,urls,ims); stores[sn]={'n_chunks':len(texts),'n_urls':len(set(urls)),'ranking':[{'url':u,'score':s} for u,s in rr[:100]]}; del texts,urls,rr; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    atomic(dst,{'status':'OK','qrels_read':False,'claim_id':cid,'model_key':a.model_key,'model':model,'common_chunk_tokenizer':CHUNK_TOK,'common_chunk_tokens':CHUNK_N,'images':ims,'stores':stores,'seconds':time.time()-t0})
    print('done',n,'/64',cid,'sec',round(time.time()-t0,1),flush=True)
if __name__=='__main__': main()
