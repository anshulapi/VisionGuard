# Repository implementation audit

Audit scope: all tracked-looking Python modules, Markdown documentation, requirements, tests, generated-data implementation, existing outputs, and repository directories. This document distinguishes code presence from measured validation.

| Component | Code location | Audited implementation | Empirically exercised | Status / caveat |
|---|---|---|---|---|
| Histogram / exposure analysis | `dip/quality.py` | Mean, median, shadow/highlight/clipping fractions, local 4x4 intensity distribution, RMS contrast | Yes; exposure / contrast tests and 12-row held-out run | Implemented as summary statistics, not a histogram-equalization module |
| CLAHE | `dip/enhancement.py` | CLAHE applied to LAB L channel; optional intermediate saving; UI diagnostic preview | Available, conditionally shown; low-contrast path exercised in pipeline tests | Implemented; not used unconditionally; planner may reject it |
| Gamma correction | `dip/enhancement.py`, `agent/planner.py` | LUT-based gamma transform and bounded gamma choice for exposure | Yes; under/overexposure tests and eval | Implemented; accepted on a measured synthetic test, but its exposure flag remained |
| Sharpness / blur | `dip/quality.py` | Laplacian variance and Sobel Tenengrad | Yes; Gaussian and motion-blur tests, held-out blur cases | Detection implemented; lost detail is not restored automatically |
| Noise estimation | `dip/quality.py` | Two-scale median residual, flat-region selection, MAD sigma, confidence | Yes; synthetic noise and exposure-independent regression tests | Heuristic; textured real products can confound it |
| Denoising | `dip/enhancement.py` | Median, bilateral and non-local means | Bilateral exercised by accepted auto-fix cases; others callable | Implemented; only bilateral is selected by planner |
| White balance / restoration | `dip/enhancement.py` | Border-median channel balancing; gamma and denoise corrections | White balance cleared the detected cast flag on one held-out synthetic image; gamma improved exposure score on one but left exposure flagged | Limited restoration operations; not a general inverse-image restoration system |
| Segmentation | `dip/segmentation.py` | LAB border-distance Otsu, both polarities, opening/closing, component scoring, optional GrabCut | Yes; dark background, small object, ambiguous background tests | Implemented with confidence / fallback; no mask benchmark is possible |
| Morphology and region analysis | `dip/segmentation.py` | Morphological close/open, connected component stats, selected component union, bbox/centroid/coverage | Yes; tests and app smoke test | Implemented; the selection score and border model remain heuristics |
| Compliance | `dip/compliance.py` | Resolution, centering, coverage, border uniformity/whiteness, clipping and confidence rules | Yes; configurable rule tests | Illustrative settings only, not official marketplace certification |
| Rule baseline | `eval/evaluate.py`, `models/train.py` | Quality flags mapped to five labels | Yes; held-out evaluation | Baseline is deterministic and uses the same threshold flags as the inspector |
| Machine learning | `models/features.py`, `models/train.py` | Multi-output Random Forest on nine handcrafted features | Yes; train and held-out evaluation | Offline only; validation split is not used for tuning; tiny synthetic support |
| Agent / auto-fixer | `agent/planner.py` | Deterministic sequential plan, target-specific checks, score/no-new-defect/SSIM/compliance gates, trace | Yes; tests and held-out test images | Not an LLM/VLM agent; only a few operations can be attempted; blur stays unchanged |
| Streamlit | `app.py` | Multi-upload, measurements, mask/overlay, correction, history, image/JSON downloads | Yes; Streamlit AppTest upload/Auto-Fix/download smoke test | No external model inference in app; interactive manual launch not performed during this audit |
| Classification evaluation | `eval/evaluate.py` | Per-label confusion, precision/recall/F1, macro/micro metrics, FPR/FNR and clean-control false reject for baseline | Yes; 12 held-out samples | Support is 1–3 per positive class; estimates are highly uncertain |
| Restoration evaluation | `eval/evaluate.py` | Aligned clean-reference PSNR/SSIM, quality before/after, selected example triptychs | Yes; 10 defective aligned pairs | Metrics don't apply to segmentation and remain task-specific |
| Segmentation evaluation | None | No ground truth masks are generated or loaded | No | IoU/Dice are not available and must not be invented |
| Dataset | `data/generate_dataset.py` | Fixed-seed geometric shape images; one clean record and one of five defects, grouped split | Yes; regenerated 60 groups | Synthetic only; real-dir option has synthetic labels, not human annotation |

## Test / execution record

- `./.venv-1/bin/python -m pytest -q`: **15 passed**.
- `python -m data.generate_dataset --groups 60 --seed 2026 --out-dir data/generated/research_seed2026`: completed in an isolated, marker-owned folder; 42 train, 6 validation, 12 test.
- `python -m models.train --manifest data/generated/research_seed2026/manifest.csv`: completed and saved `outputs/model/visionguard_rf.joblib` and metrics.
- `MPLCONFIGDIR=/private/tmp/visionguard-mpl python -m eval.evaluate --manifest data/generated/research_seed2026/manifest.csv`: completed; wrote held-out results, metrics, chart, confusion JSON, and examples.
- `python -m scripts.run_demo --manifest data/generated/research_seed2026/manifest.csv`: completed without regenerating or changing the experimental dataset.
- `research/build_research.py`: builds the separate evidence package after these commands.

## Unverified / incomplete evidence

- Exposure scoring now includes the midtone condition that produces the exposure flag; a regression test confirms gamma improves the corresponding score. The quality score also includes a color-balance component; a regression test confirms measured white balance can improve that component and pass the planner guards. Remaining flags are retained in the trace, and the planner reports `partially_improved` when any initially detected flag remains.
- The legacy root `data/generated` tree remains untouched and may contain stale files from earlier runs. The final experiments use only the isolated manifest and its 120 referenced image files.

- The two supplied course PDFs were found but text extraction was not possible in this environment (no PDF CLI or Python reader; local PDFKit attempt failed due to Swift toolchain/module-cache errors). This repository audit therefore does not assert conformance to their detailed instructions.
- No supplied real product data or the referenced watch JPEG was found. Real-world performance is unmeasured.
- No ground truth segmentation annotations, multi-defect test set, threshold calibration, or validation-driven model tuning exist.
- No broad ablation study compares image enhancement operators; the outputs compare only rule vs RF defect classification and auto-fix before/after on the current synthetic test set.
- Pytest checks a representative subset of API behavior, not every image format, threshold configuration, operation, or marketplace rule combination.
