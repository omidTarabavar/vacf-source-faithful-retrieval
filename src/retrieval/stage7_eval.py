#!/usr/bin/env python3
import json,re
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode
import numpy as np

def jl(p):
  with open(p,encoding='utf-8') as f:
    for x in f:
      if x.strip(): yield json.loads(x)
def c(u):
  u=(u or '').strip()
  if not u:return ''
  try:p=urlsplit(u)
  except:return u.rstrip('/')
  net=p.netloc.lower(); net=net[4:] if net.startswith('www.') else net; path=re.sub('/{2,}','/',p.path or '/'); path=path if path=='/' else path.rstrip('/'); q=urlencode(sorted((k,v) for k,v in parse_qsl(p.query,keep_blank_values=True) if not k.lower().startswith('utm_'))); return urlunsplit(((p.scheme or 'https').lower(),net,path,q,''))
def uq(xs,k=10):
  o=[];s=set()
  for u in xs:
    u=c(u)
    if u and u not in s:s.add(u);o.append(u)
    if k and len(o)>=k:break
  return o
def qrels(project):
  o={}
  for r in jl(Path(project)/'data/qrels.jsonl'):
    v=r.get('gold_urls') or r.get('urls') or r.get('relevant_urls') or r.get('qrels'); v=list(v) if isinstance(v,dict) else v; v=[r['url']] if v is None and r.get('url') else v; v=[v] if isinstance(v,str) else v
    if r.get('claim_id') and v:o[r['claim_id']]=v
  return o
def m(urls,g):
  u=uq(urls,10);gs={c(x) for x in g if c(x)};h=[i+1 for i,x in enumerate(u) if x in gs];return {'rr':1/h[0] if h else 0.,'hit':1. if h else 0.,'recall':len(h)/len(gs) if gs else 0.,'n':len(u)}
def boot(a,b,seed=707,n=10000):
  d=np.asarray(b)-np.asarray(a);rng=np.random.default_rng(seed);v=np.empty(n)
  for i in range(n):ix=rng.integers(0,len(d),len(d));v[i]=d[ix].mean()
  return float(d.mean()),float(np.quantile(v,.025)),float(np.quantile(v,.975))
def rank(pred,key,cid,store):
  z=json.loads((Path(pred)/key/f'{cid}.json').read_text());return [x['url'] for x in z['stores'][store]['ranking']]
def xi(xp):
  u=[]
  for b in xp.get('image_channel',[]):u += [h.get('url','') for h in b.get('hits',[])]
  return uq(u,10)

def main():
  root=Path('/workspace/stage7_model_store_matrix'); pred=root/'predictions'; st=json.loads(Path('/workspace/to_v4_retrieval/STATE.json').read_text()); rows=st['cohorts']['fresh']
  for r in rows:
    for k in ['base','siglip2']:
      if not (pred/k/f"{r['claim_id']}.json").exists(): raise FileNotFoundError((pred/k/f"{r['claim_id']}.json"))
  g=qrels(st['project']); names=['CACHED_XI','BASE_IMAGE','S2_IMAGE','BASE_TEXT','S2_TEXT','XI_THEN_BASE_TEXT','XI_THEN_S2_TEXT','BASE_IMAGE_THEN_S2_TEXT','S2_IMAGE_THEN_S2_TEXT']; by={k:[] for k in names}; per=[]
  for r in rows:
    cid=r['claim_id']; bi=rank(pred,'base',cid,'IMAGE_STORE'); bt=rank(pred,'base',cid,'TEXT_STORE'); si=rank(pred,'siglip2',cid,'IMAGE_STORE'); stx=rank(pred,'siglip2',cid,'TEXT_STORE'); xp=json.loads(Path('/workspace/xxp_stage4_cal64/predictions',f'{cid}.json').read_text()); x=xi(xp)
    vals={'CACHED_XI':x,'BASE_IMAGE':uq(bi,10),'S2_IMAGE':uq(si,10),'BASE_TEXT':uq(bt,10),'S2_TEXT':uq(stx,10),'XI_THEN_BASE_TEXT':uq(x+bt,10),'XI_THEN_S2_TEXT':uq(x+stx,10),'BASE_IMAGE_THEN_S2_TEXT':uq(bi+stx,10),'S2_IMAGE_THEN_S2_TEXT':uq(si+stx,10)}; mm={k:m(v,g[cid]) for k,v in vals.items()}; [by[k].append(mm[k]) for k in names]; per.append({'claim_id':cid,'metrics':mm})
  sm={k:{'mrr10':float(np.mean([x['rr'] for x in v])),'hit10':float(np.mean([x['hit'] for x in v])),'recall10':float(np.mean([x['recall'] for x in v])),'unique10':float(np.mean([x['n'] for x in v]))} for k,v in by.items()}
  head={}
  for k,model,store in [('BASE_IMAGE','base','IMAGE_STORE'),('S2_IMAGE','siglip2','IMAGE_STORE'),('BASE_TEXT','base','TEXT_STORE'),('S2_TEXT','siglip2','TEXT_STORE')]:
    n100=0
    for r in rows:
      cid=r['claim_id']; urls=rank(pred,model,cid,store);gs={c(x) for x in g[cid] if c(x)};n100+=bool(set(uq(urls,100))&gs)
    head[k]={'hit10':int(sum(x['hit'] for x in by[k])),'hit100':int(n100)}
  comp={}
  base=[x['rr'] for x in by['CACHED_XI']]
  for k in names:
    if k!='CACHED_XI':comp[k]=boot(base,[x['rr'] for x in by[k]])
  (root/'STAGE7.json').write_text(json.dumps({'summary':sm,'headroom':head,'bootstrap_vs_XI':comp,'per_claim':per},ensure_ascii=False,indent=2)+'\n')
  lines=['# Stage 7 — Controlled model × store matrix','','|method|MRR@10|Hit@10|Recall@10|unique@10|','|---|---:|---:|---:|---:|']
  for k in names:
    x=sm[k];lines.append(f"|{k}|{x['mrr10']:.6f}|{x['hit10']:.6f}|{x['recall10']:.6f}|{x['unique10']:.2f}|")
  lines+=['','## Candidate headroom @100']+[f"- {k}: hit@10 claims={v['hit10']}, hit@100 claims={v['hit100']}" for k,v in head.items()]+['','## ΔMRR@10 vs cached XI']
  for k in ['S2_IMAGE','S2_TEXT','XI_THEN_BASE_TEXT','XI_THEN_S2_TEXT','S2_IMAGE_THEN_S2_TEXT']:
    d,lo,hi=comp[k];lines.append(f'- {k}: {d:+.6f} [{lo:+.6f}, {hi:+.6f}]')
  (root/'STAGE7.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
