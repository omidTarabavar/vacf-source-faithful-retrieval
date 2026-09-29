from pathlib import Path

root = Path(__file__).resolve().parent
required = [
    "README.md",
    "requirements.txt",
    "CITATION.cff",
    "src/retrieval/stage7_run_expert.py",
    "src/retrieval/stage8a_freeze_val152.py",
    "src/restoration/stage8c_restore_context_val152.py",
    "src/evaluation/stage10f_eval_val152_FINAL_seeded.py",
    "src/efficiency/benchmark_val152_efficiency.py",
    "docs/REPRODUCIBILITY.md",
]
missing = [p for p in required if not (root / p).exists()]
if missing:
    raise SystemExit("Missing required release files:\n" + "\n".join(missing))
print("Release structure OK")
