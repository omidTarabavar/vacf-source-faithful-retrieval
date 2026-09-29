import json
from pathlib import Path

BASE = Path(
    "/workspace/V5B_FREEZE/"
    "CANDIDATE_Y_64_FROZEN.json"
)

V5 = Path(
    "/workspace/STAGE10E_RESULTS/"
    "CAL64_V5B_PROTECTED_FUSION.json"
)

OUT = Path(
    "/workspace/STAGE10F_RESULTS/"
    "CAL64_ADDONLY_FUSION.json"
)

OUT.parent.mkdir(parents=True, exist_ok=True)


base = json.loads(BASE.read_text())
v5 = json.loads(V5.read_text())


base_rows = {
    x["claim_id"]: x
    for x in base["methods"]["CANDIDATE_X"]
}

v5_rows = {
    x["claim_id"]: x
    for x in v5["methods"]["V5B_PROTECTED_FUSION"]
}


final = []

stats = {
    "claims":0,
    "added":0,
    "total_added_urls":0
}


for cid, row in base_rows.items():

    selected = list(row["selected"])
    evidence = list(row["evidence"])

    existing = {
        e["url"]
        for e in evidence
    }

    added = 0

    # rescue candidates from V5B
    v5e = v5_rows[cid]["evidence"]

    for e in v5e:

        if e["url"] in existing:
            continue

        # only strong rescue
        if (
            e.get("expert")
            == "TEXT_SEMANTIC_RESCUE"
        ):

            # add at most 3
            if added >= 3:
                break

            evidence.append(e)
            selected.append({
                "url": e["url"],
                "expert": e.get("expert"),
            })

            existing.add(e["url"])
            added += 1


    final.append(
        {
            "claim_id": cid,
            "claim_text": row["claim_text"],
            "method": "CANDIDATE_Y_ADDONLY",
            "selected": selected[:10],
            "evidence": evidence[:10],
            "added_rescue": added
        }
    )

    stats["claims"] += 1
    stats["added"] += int(added>0)
    stats["total_added_urls"] += added


out = {
    "status":"READY",
    "methods":{
        "CANDIDATE_Y_ADDONLY": final
    },
    "stats":stats
}


OUT.write_text(
    json.dumps(
        out,
        indent=2,
        ensure_ascii=False
    )
)

print(json.dumps(stats,indent=2))
print("DONE:", OUT)
