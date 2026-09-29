import sys
import json
from pathlib import Path

sys.path.insert(0, "/workspace/XXP_OFFICIAL")

import run_pipeline as rp
from datasets import Dataset, DatasetDict


VAL_JSON = Path(
    "/workspace/AVerImaTec_Shared_Task/"
    "data/data_clean/split_data/val.json"
)

# --------------------------------------------------
# Force XxP to use our already verified local val.json.
# This avoids HF trying to infer train+validation+test
# formats together.
# --------------------------------------------------

_orig_load_dataset = rp.load_dataset

def local_averimatec_loader(name, *args, **kwargs):
    if name == "Rui4416/AVerImaTeC":
        rows = json.loads(VAL_JSON.read_text(encoding="utf-8"))
        assert len(rows) == 152, f"Expected 152 val rows, got {len(rows)}"

        val_ds = Dataset.from_list(rows)

        print(
            f"[LOCAL DATASET] AVerImaTeC validation loaded: "
            f"{len(val_ds)} rows"
        )

        return DatasetDict({
            "validation": val_ds
        })

    return _orig_load_dataset(name, *args, **kwargs)

rp.load_dataset = local_averimatec_loader


# --------------------------------------------------
# Evidence-only:
# retrieval is kept unchanged.
# Final veracity + justification generation skipped.
# --------------------------------------------------

rp.GenerationLLMHandler.generate_prediction = \
    lambda self, *args, **kwargs: ""

rp.GenerationLLMHandler.generate_justification = \
    lambda self, *args, **kwargs: ""


def run(n, out):
    rp.run_pipeline(
        image_dir=(
            "/workspace/AVerImaTec_Shared_Task/"
            "data/data_clean/images"
        ),
        text_store_dir=(
            "/workspace/Knowledge_Store/val_extracted/"
            "converted_datastore/text_related/"
            "text_related_store_text_val"
        ),
        image_store_dir=(
            "/workspace/Knowledge_Store/val_extracted/"
            "converted_datastore/text_related/"
            "image_related_store_text_val"
        ),
        output_path=out,
        split="validation",
        num_instances=n,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run(a.n, a.out)
