# Paths and inputs expected by the paper scripts

The source ZIP is a faithful export of an experimental `/workspace` tree.
Many scripts therefore contain absolute defaults such as `/workspace/...`.
This cleaned repository preserves the exact experiment code rather than
silently changing scientific behavior.

## External inputs

1. AVerImaTeC split JSON files and claim images.
2. Organizer-provided claim-specific knowledge store.
3. Hugging Face checkpoints:
   - `google/siglip-base-patch16-384`
   - `google/siglip2-large-patch16-512`
   - `google/gemma-3-27b-it` for the official semantic evaluator
   - `Qwen/Qwen3-VL-8B-Instruct` for matched downstream verification
4. XxP comparator code for the controlled matched comparison.

## Main workspace names found in the archived scripts

- `/workspace/AVerImaTec_Shared_Task`
- `/workspace/XXP_OFFICIAL`
- `/workspace/Knowledge_Store/...`
- `/workspace/stage7_model_store_matrix/predictions`
- `/workspace/VAL152_STAGE8A/STAGE8A_FROZEN_EVIDENCE.json`
- `/workspace/TEST352_STAGE8A/STAGE8A_FROZEN_EVIDENCE.json`
- `/workspace/VAL152_CANDIDATE_Y.json`
- `/workspace/TEST352_CANDIDATE_Y.json`

## Portability

Some scripts expose CLI arguments for these paths; others define path constants
near the top of the file. Before public execution on a new machine, replace
those constants with local paths or wrap the scripts in your preferred config
system. This release intentionally keeps the exact paper scripts intact.
