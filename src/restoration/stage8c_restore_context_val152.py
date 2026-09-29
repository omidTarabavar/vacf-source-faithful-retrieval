import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from transformers import AutoTokenizer

FROZEN = Path("/workspace/VAL152_STAGE8A/STAGE8A_FROZEN_EVIDENCE.json")

STORE = Path("/workspace/Knowledge_Store/val_extracted/converted_datastore/text_related")
OUT = Path("/workspace/VAL152_CANDIDATE_Y.json")

LIMIT = 152
BUDGET = 1200


def canon(u):
    from urllib.parse import urlsplit, parse_qsl, urlencode

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

    query = urlencode(sorted(
        (k, v)
        for k, v in parse_qsl(
            p.query,
            keep_blank_values=True
        )
        if not k.lower().startswith("utm_")
    ))

    # Scheme is intentionally ignored for raw-store matching.
    key = host + path

    if query:
        key += "?" + query

    return key
def norm_tokens(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def piece_score(chunk, piece):
    a = set(norm_tokens(chunk))
    b = set(norm_tokens(piece))

    if not a:
        return 0.0

    # We care about how much of the old retrieved chunk
    # is contained in the original source piece.
    return len(a & b) / len(a)


def load_raw_pieces(path, url):
    target = canon(url)
    pieces = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if canon(x.get("url", "")) != target:
                continue

            vals = x.get("url2text", [])

            if isinstance(vals, str):
                vals = [vals]

            pieces.extend([
                str(v).strip()
                for v in vals
                if str(v).strip()
            ])

    return pieces


tok = AutoTokenizer.from_pretrained(
    "google/gemma-3-27b-it"
)


def pack_context(pieces, anchors):
    if not pieces:
        return ""

    priority = {}

    # Exact pieces containing the previously selected chunks.
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

    # Keep title / date / lead metadata when available.
    for i in range(min(3, len(pieces))):
        priority[i] = min(priority.get(i, 99), 2)

    chosen = []
    used = 0

    for i in sorted(priority, key=lambda x: (priority[x], x)):
        text = pieces[i]

        ids = tok.encode(
            text,
            add_special_tokens=False
        )

        remaining = BUDGET - used

        if remaining <= 0:
            break

        if len(ids) <= remaining:
            chosen.append((i, text))
            used += len(ids)

        elif priority[i] == 0 and remaining >= 64:
            clipped = tok.decode(
                ids[:remaining],
                skip_special_tokens=True
            ).strip()

            if clipped:
                chosen.append((i, clipped))

            used = BUDGET
            break

    chosen.sort(key=lambda x: x[0])

    return "\n\n".join(
        x[1] for x in chosen
    ).strip()


frozen = json.loads(
    FROZEN.read_text(encoding="utf-8")
)

rows = frozen["methods"]["CANDIDATE_X"]

new_rows = []

for pos, row in enumerate(rows):

    # Claims after the diagnostic five stay untouched.
    if pos >= LIMIT:
        new_rows.append(row)
        continue

    cid = row["claim_id"]
    idx = str(int(cid.rsplit("_", 1)[1]))

    new_row = {
        **row,
        "method": "CANDIDATE_Y_SOURCE_FAITHFUL_CONTEXT",
        "evidence": [],
    }

    for evid in row["evidence"]:

        expert = evid.get("expert")

        if expert == "BASE_IMAGE":
            branch = "image_related_store_text_val"

        elif expert == "S2_TEXT":
            branch = "text_related_store_text_val"

        else:
            raise RuntimeError(
                f"Unknown expert {cid}: {expert}"
            )

        store_path = STORE/branch/f"{idx}.json"

        pieces = load_raw_pieces(
            store_path,
            evid["url"]
        )

        if not pieces:
            raise RuntimeError(
                f"No raw text found:\n"
                f"{cid}\n{evid['url']}"
            )

        old_chunks = [
            x.strip()
            for x in (evid.get("text") or "").splitlines()
            if x.strip()
        ]

        anchors = set()

        for chunk in old_chunks:
            scores = [
                piece_score(chunk, piece)
                for piece in pieces
            ]

            if scores:
                anchors.add(
                    max(
                        range(len(scores)),
                        key=scores.__getitem__
                    )
                )

        text = pack_context(
            pieces,
            anchors
        )

        if not text:
            raise RuntimeError(
                f"Empty restored evidence {cid}"
            )

        new_row["evidence"].append({
            **evid,
            "text": text,
            "localizer":
                "source_faithful_context_restoration",
            "budget_tokens": BUDGET,
        })

    new_rows.append(new_row)

    print(
        "RESTORED",
        pos + 1,
        "/",
        LIMIT,
        cid
    )

out = {
    **frozen,
    "status": "STAGE8C_CANDIDATE_Y_READY",
    "stage8c_note":
        "URLs/order unchanged; first five claims "
        "use source-faithful contextual evidence.",
    "methods": {
        "XXP_FULL":
            frozen["methods"]["XXP_FULL"],
        "CANDIDATE_X":
            new_rows,
    },
}

OUT.write_text(
    json.dumps(
        out,
        ensure_ascii=False,
        indent=2
    ) + "\n",
    encoding="utf-8"
)

print("STAGE8C_READY:", OUT)
