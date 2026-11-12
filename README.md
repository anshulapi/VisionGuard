# VisionGuard — HSR-AIML-056

Local, CPU-friendly image quality and illustrative e-commerce compliance inspector. It demonstrates classical DIP (sharpness, exposure distributions, flat-region noise, LAB CLAHE, morphology, two-polarity Otsu and confidence-aware component selection) alongside a small Random Forest trained on hand-crafted DIP features. No paid APIs, GPU, downloaded models or external dataset are required.

## Install and run

Use Python 3.11 (the code is also compatible with the supplied Python 3.12 runtime):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m data.generate_dataset --groups 60 --seed 2026 --out-dir data/generated/research_seed2026
python -m models.train --manifest data/generated/research_seed2026/manifest.csv
python -m eval.evaluate --manifest data/generated/research_seed2026/manifest.csv
python -m scripts.run_demo --manifest data/generated/research_seed2026/manifest.csv
python -m pytest -q
python research/build_research.py --manifest data/generated/research_seed2026/manifest.csv
streamlit run app.py
```

Generated inputs, model, plots and reports go under `data/generated/` and `outputs/`; neither is tracked. Dataset generation uses a fixed seed, cleans only the default generated-data root or a marker-owned custom output, refuses unknown non-empty custom outputs, and keeps source groups within one split. Pass the same `--manifest` path to training, evaluation and demo so they consume exactly the same split. The optional `--real-dir PATH` argument ingests local images, resized to 256x256; resulting defect labels still describe only the synthetic transform and must not be treated as labels for arbitrary real imagery. The noisy watch JPEG mentioned in the reliability request was not included in the repository; tests use a reproducible watch-like image with Gaussian noise, blur and exposure variants.

## Outputs

- `outputs/model/visionguard_rf.joblib`, `metrics.json`
- `outputs/evaluation/results.csv`, `metrics.json`, `confusion_matrices.json`, `quality_comparison.png`, `examples/`
- `outputs/demo/` with input, corrected image, mask, conditionally generated CLAHE diagnostic preview, and JSON trace

## Design and limits

See [ARCHITECTURE_IMPLEMENTED.md](ARCHITECTURE_IMPLEMENTED.md), [report/PROJECT_REPORT.md](report/PROJECT_REPORT.md), and [report/PRESENTATION_OUTLINE.md](report/PRESENTATION_OUTLINE.md). Marketplace-style rules are examples, not official certifications. Noise uses robust residuals from low-gradient regions and two filter scales; the estimate is heuristic and should be validated on real examples. Segmentation reports confidence and warnings for ambiguous results. The correction planner only accepts target-metric improvements that also pass score, defect, similarity and compliance guards. Blur is diagnosed but not automatically “fixed”, because lost detail cannot be reliably recovered. Synthetic product shapes are simple and do not establish real-catalog performance.
