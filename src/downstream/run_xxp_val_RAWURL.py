import sys
import json
from pathlib import Path

sys.path.insert(0, "/workspace/XXP_OFFICIAL")

import run_pipeline_with_urls as rp

from datasets import Dataset, DatasetDict


VAL_JSON = Path(
    "/workspace/AVerImaTec_Shared_Task/"
    "data/data_clean/split_data/val.json"
)


_orig_load_dataset = rp.load_dataset


def local_loader(name, *args, **kwargs):

    if name == "Rui4416/AVerImaTeC":

        rows = json.loads(
            VAL_JSON.read_text(encoding="utf-8")
        )

        assert len(rows) == 152

        ds = Dataset.from_list(rows)

        print(
            f"[LOCAL DATASET] validation={len(ds)}"
        )

        return DatasetDict({
            "validation": ds
        })

    return _orig_load_dataset(
        name, *args, **kwargs
    )


rp.load_dataset = local_loader


# Keep question generation + retrieval unchanged.
# Skip only stages irrelevant to evidence comparison.

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
            "/workspace/Knowledge_Store/"
            "val_extracted/converted_datastore/"
            "text_related/"
            "text_related_store_text_val"
        ),
        image_store_dir=(
            "/workspace/Knowledge_Store/"
            "val_extracted/converted_datastore/"
            "text_related/"
            "image_related_store_text_val"
        ),
        output_path=out,
        split="validation",
        num_instances=n,
    )


if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n",
        type=int,
        required=True
    )

    ap.add_argument(
        "--out",
        required=True
    )

    a = ap.parse_args()

    run(a.n, a.out)
