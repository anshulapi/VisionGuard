# VisionGuard research evidence

This directory is a separate audit/evaluation artifact. It is not the course paper and does not change application source. It records one reproducible experiment run on the generated synthetic dataset.

## Reproduce

From repository root with dependencies installed:

```bash
python -m data.generate_dataset --groups 60 --seed 2026 --out-dir data/generated/research_seed2026
python -m models.train --manifest data/generated/research_seed2026/manifest.csv
python -m eval.evaluate --manifest data/generated/research_seed2026/manifest.csv
python -m scripts.run_demo --manifest data/generated/research_seed2026/manifest.csv
python -m pytest -q
python research/build_research.py --manifest data/generated/research_seed2026/manifest.csv
```

`experiment_manifest.csv` is the authoritative sample record for the run. `experiment_results.csv` contains one row per held-out input, ground-truth labels, rule/RF predictions, score changes, correction counts, and aligned-reference PSNR/SSIM. Figures and selected case triptychs are generated from the same run. `evaluation_summary.md` reports measured values and per-split class counts; `implementation_status.md` distinguishes implemented algorithms from unavailable evaluation. No IoU/Dice values are available because no ground-truth segmentation masks exist.

The PDF course handouts could not be extracted in the current environment, and no claim about their precise requirements is made in the audit.
