import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SRC = Path(
    "/workspace/STAGE9C_RESULTS/"
    "STAGE9C_QWEN_RERANK.json"
)

TRAIN = Path(
    "/workspace/AVerImaTec_Shared_Task/"
    "data/data_clean/split_data/train.json"
)

DATA_ROOT = Path(
    "/workspace/AVerImaTec_Shared_Task/"
    "data/data_clean"
)

OUTDIR = Path(
    "/workspace/STAGE9F_RESULTS"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT = OUTDIR / "MULTIMODAL_QUERIES.json"


def find_image(name):
    if not name:
        return None

    name = str(name)

    candidates = [
        DATA_ROOT / "images" / name,
        DATA_ROOT / name,
        TRAIN.parent / name,
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    # Fallback only when direct paths fail.
    matches = list(
        DATA_ROOT.rglob(
            Path(name).name
        )
    )

    if matches:
        return matches[0].resolve()

    return None


def parse_json_response(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"```$",
        "",
        text
    ).strip()

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.S
    )

    if not match:
        return None

    try:
        return json.loads(
            match.group(0)
        )
    except Exception:
        return None


src = json.loads(
    SRC.read_text(
        encoding="utf-8"
    )
)["results"]

train = json.loads(
    TRAIN.read_text(
        encoding="utf-8"
    )
)


print(
    "Loading processor:",
    MODEL
)

processor = AutoProcessor.from_pretrained(
    MODEL
)


print(
    "Loading model:",
    MODEL
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="auto"
).eval()

print(
    "MODEL_DEVICE =",
    model.device
)


if OUT.exists():

    existing = json.loads(
        OUT.read_text(
            encoding="utf-8"
        )
    )

else:
    existing = {}


for n, (cid, row) in enumerate(
    src.items(),
    1
):

    if cid in existing:
        print(
            f"[CACHE] {n}/{len(src)} {cid}"
        )
        continue

    idx = int(
        cid.rsplit(
            "_",
            1
        )[1]
    )

    tr = train[idx]

    claim = row["claim"]

    image_names = (
        tr.get("claim_images", [])
        or []
    )

    if isinstance(
        image_names,
        str
    ):
        image_names = [
            image_names
        ]

    image_paths = []

    for name in image_names[:3]:

        # Handle strings and simple dict forms.
        if isinstance(name, dict):

            name = (
                name.get("path")
                or name.get("filename")
                or name.get("name")
                or name.get("image")
            )

        p = find_image(name)

        if p is not None:
            image_paths.append(p)


    prompt = f"""
You are preparing retrieval queries for multimodal fact checking.

CLAIM:
{claim}

Analyze the claim together with the supplied claim image(s).

Do NOT decide whether the claim is true or false.
Do NOT invent facts that are not visible or stated.

Your only task is to create high-recall web evidence retrieval queries.

Extract:
1. visible text/OCR that could be useful for search
2. identifiable people, organizations, places, dates, objects, logos or events
3. the key factual relationship that needs verification
4. exactly five complementary search queries

The five queries should deliberately differ:
- one entity/event query
- one source/date/location query
- one claim-verification query
- one image-context/origin query
- one fact-check style query

Preserve exact proper names and useful numbers whenever possible.

Return ONLY valid JSON in this exact structure:

{{
  "visual_cues": ["..."],
  "ocr_text": ["..."],
  "entities": ["..."],
  "verification_target": "...",
  "queries": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}}
""".strip()


    content = []

    for p in image_paths:

        content.append({
            "type": "image",
            "path": str(p)
        })

    content.append({
        "type": "text",
        "text": prompt
    })


    messages = [
        {
            "role": "user",
            "content": content
        }
    ]


    print(
        f"[CLAIM] {n}/{len(src)} "
        f"{cid} "
        f"images={len(image_paths)}",
        flush=True
    )


    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(
        model.device
    )


    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )


    generated = outputs[
        0,
        inputs["input_ids"].shape[-1]:
    ]


    raw = processor.decode(
        generated,
        skip_special_tokens=True
    ).strip()


    parsed = parse_json_response(
        raw
    )


    existing[cid] = {
        "claim":
            claim,

        "image_paths": [
            str(x)
            for x in image_paths
        ],

        "raw_output":
            raw,

        "parsed":
            parsed
    }


    OUT.write_text(
        json.dumps(
            existing,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


    print(
        f"[DONE] {cid} "
        f"json_ok={parsed is not None}",
        flush=True
    )


print()
print(
    "STAGE9F_QUERY_GENERATION_COMPLETE"
)
print(
    "OUTPUT =",
    OUT
)
