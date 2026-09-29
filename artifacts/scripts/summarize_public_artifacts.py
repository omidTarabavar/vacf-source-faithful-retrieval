#!/usr/bin/env python3
import csv, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
v = root / "val152"

def read_csv(name):
    with open(v / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def mean(xs):
    return sum(xs) / len(xs)

def report(rows, a, b):
    av = [float(r[a]) for r in rows]
    bv = [float(r[b]) for r in rows]
    d = [y-x for x,y in zip(av,bv)]
    print(f"n={len(rows)}")
    print(f"{a}: {mean(av):.12f}")
    print(f"{b}: {mean(bv):.12f}")
    print(f"delta: {mean(d):.12f}")
    print(f"wins/losses/ties: {sum(x>0 for x in d)}/{sum(x<0 for x in d)}/{sum(x==0 for x in d)}")
    print()

print("XxP vs full VACF")
report(read_csv("VAL152_XXP_VACF_PER_CLAIM_EVIDENCE.csv"),
       "xxp_evidence_score", "vacf_evidence_score")

print("Visual-only vs contextual-only")
report(read_csv("VAL152_VISUAL_CONTEXTUAL_PER_CLAIM_EVIDENCE.csv"),
       "visual_only_evidence_score", "contextual_only_evidence_score")

print("VACF without restoration vs full VACF")
report(read_csv("VAL152_RESTORATION_PER_CLAIM_EVIDENCE.csv"),
       "vacf_without_restoration_evidence_score", "vacf_full_evidence_score")

with open(v / "VAL152_ALL_METRICS_PAIRED_STATS_FINAL.json", encoding="utf-8") as f:
    stats = json.load(f)
print("Exact manuscript bootstrap output:")
print(json.dumps(stats["comparisons"]["Evidence"]["VACF_minus_XXP"], indent=2))
