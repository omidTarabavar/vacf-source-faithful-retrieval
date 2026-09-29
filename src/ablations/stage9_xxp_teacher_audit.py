import json
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode

RESULTS = Path(
    "/workspace/STAGE8C_FULL_RESULTS/STAGE8B_RESULTS.json"
)

FROZEN = Path(
    "/workspace/STAGE8C_CANDIDATE_Y_64.json"
)

LOG = Path(
    "/workspace/stage8c_full64.log"
)

STORE = Path(
    "/workspace/stage8c_store"
)

OUT = Path(
    "/workspace/STAGE9_XXP_TEACHER_AUDIT.md"
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

    query = urlencode(sorted(
        (k, v)
        for k, v in parse_qsl(
            p.query,
            keep_blank_values=True
        )
        if not k.lower().startswith("utm_")
    ))

    key = host + path

    if query:
        key += "?" + query

    return key


def url_in_store(cid, url, branch):
    idx = str(
        int(cid.rsplit("_", 1)[1])
    )

    p = STORE / branch / f"{idx}.json"

    if not p.exists():
        return False

    target = canon(url)

    with p.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            if canon(
                x.get("url", "")
            ) == target:
                return True

    return False


res = json.loads(
    RESULTS.read_text(
        encoding="utf-8"
    )
)

fr = json.loads(
    FROZEN.read_text(
        encoding="utf-8"
    )
)

xmap = {
    x["claim_id"]: x
    for x in fr["methods"]["XXP_FULL"]
}

ymap = {
    x["claim_id"]: x
    for x in fr["methods"]["CANDIDATE_X"]
}


# ---------------------------------------------------------
# Parse successful XxP PRED indices from official evaluator log
# ---------------------------------------------------------

matched_pred = {}

current_method = None
pending = []

for line in LOG.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines():

    if line.startswith("METHOD:"):
        current_method = (
            line.split(
                ":",
                1
            )[1].strip()
        )
        pending = []
        continue

    if (
        current_method == "XXP_FULL"
        and line.startswith(
            "ref in pred:"
        )
    ):
        pending = sorted(set(
            int(x) - 1
            for x in re.findall(
                r"PRED_(\d+)",
                line
            )
        ))

    m = re.search(
        r"\[EVAL\]\s+XXP_FULL\s+"
        r"\d+/64\s+(\S+)\s+"
        r"EvidenceScore="
        r"([0-9.]+)",
        line
    )

    if m:
        cid = m.group(1)

        matched_pred[cid] = list(
            pending
        )

        pending = []


# ---------------------------------------------------------
# Score categories
# ---------------------------------------------------------

ids = list(
    res["methods"]["XXP_FULL"].keys()
)

rows = []

for cid in ids:

    xs = res[
        "methods"
    ]["XXP_FULL"][cid][
        "evidence_score"
    ]

    ys = res[
        "methods"
    ]["CANDIDATE_X"][cid][
        "evidence_score"
    ]

    rows.append(
        (cid, xs, ys)
    )


losses = [
    r for r in rows
    if r[1] > r[2]
]

wins = [
    r for r in rows
    if r[2] > r[1]
]

both_zero = [
    r for r in rows
    if r[1] == 0
    and r[2] == 0
]


oracle = sum(
    max(x, y)
    for _, x, y in rows
) / len(rows)

candidate_mean = sum(
    y
    for _, _, y in rows
) / len(rows)

xxp_mean = sum(
    x
    for _, x, _ in rows
) / len(rows)


lines = []

lines += [
    "# Stage 9 — XxP Teacher / Failure Audit",
    "",
    f"- XxP mean: {xxp_mean:.6f}",
    f"- Candidate-Y mean: {candidate_mean:.6f}",
    f"- Per-claim oracle max(XxP,Y): {oracle:.6f}",
    f"- Oracle headroom over Y: {oracle-candidate_mean:+.6f}",
    f"- Y wins: {len(wins)}",
    f"- Y losses: {len(losses)}",
    f"- Both zero: {len(both_zero)}",
    "",
    "## XxP > Candidate-Y cases",
    ""
]


teacher_missing = 0
teacher_already_in_y = 0
teacher_in_text_store = 0
teacher_in_image_store = 0


for cid, xs, ys in sorted(
    losses,
    key=lambda z: z[1]-z[2],
    reverse=True
):

    xr = xmap[cid]
    yr = ymap[cid]

    y_urls = {
        canon(e.get("url", ""))
        for e in yr["evidence"]
    }

    claim_text = (
        yr.get("claim_text")
        or xr.get("claim_text")
        or ""
    )

    lines += [
        f"### {cid}",
        "",
        f"- XxP: **{xs:.6f}**",
        f"- Candidate-Y: **{ys:.6f}**",
        f"- Delta Y-X: **{ys-xs:+.6f}**",
        "",
        f"Claim: {claim_text}",
        "",
        "Successful XxP evidence:",
        ""
    ]

    inds = matched_pred.get(
        cid,
        []
    )

    if not inds:
        lines += [
            "_No successful PRED indices "
            "available in this log "
            "(possibly cached claim)._",
            ""
        ]

    for pi in inds:

        if pi >= len(
            xr["evidence"]
        ):
            continue

        e = xr[
            "evidence"
        ][pi]

        url = e.get(
            "url",
            ""
        )

        text = (
            e.get(
                "text",
                ""
            )
            .replace(
                "\n",
                " "
            )
            .strip()
        )

        overlap = (
            canon(url)
            in y_urls
        )

        in_img = url_in_store(
            cid,
            url,
            "image_related_store_text_train"
        )

        in_txt = url_in_store(
            cid,
            url,
            "text_related_store_text_train"
        )

        if overlap:
            teacher_already_in_y += 1
        else:
            teacher_missing += 1

        if in_img:
            teacher_in_image_store += 1

        if in_txt:
            teacher_in_text_store += 1

        lines += [
            f"- PRED_{pi+1}",
            f"  - URL: {url}",
            f"  - already in Candidate-Y Top10: {overlap}",
            f"  - present in image-related store: {in_img}",
            f"  - present in text-related store: {in_txt}",
            f"  - text: {text[:900]}",
            ""
        ]

    lines += [
        "Candidate-Y URLs:",
        ""
    ]

    for j, e in enumerate(
        yr["evidence"],
        1
    ):
        lines.append(
            f"{j}. {e.get('url','')}"
        )

    lines.append("")


lines += [
    "## Aggregate teacher coverage",
    "",
    f"- Successful XxP evidence already in Candidate-Y Top10: {teacher_already_in_y}",
    f"- Successful XxP evidence missing from Candidate-Y Top10: {teacher_missing}",
    f"- Successful XxP evidence found in image-related raw store: {teacher_in_image_store}",
    f"- Successful XxP evidence found in text-related raw store: {teacher_in_text_store}",
    "",
    "## Both-zero claims",
    ""
]

for cid, xs, ys in both_zero:
    claim = (
        ymap[cid].get(
            "claim_text",
            ""
        )
    )

    lines.append(
        f"- {cid}: {claim}"
    )


OUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("STAGE9_AUDIT_READY")
print("losses =", len(losses))
print("wins =", len(wins))
print("both_zero =", len(both_zero))
print("oracle =", f"{oracle:.6f}")
print(
    "oracle_headroom =",
    f"{oracle-candidate_mean:+.6f}"
)
print(
    "teacher URLs already in Y =",
    teacher_already_in_y
)
print(
    "teacher URLs missing from Y =",
    teacher_missing
)
print(
    "teacher URLs in image store =",
    teacher_in_image_store
)
print(
    "teacher URLs in text store =",
    teacher_in_text_store
)
print("OUTPUT =", OUT)
