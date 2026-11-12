# VisionGuard repository audit and experiment summary

## Scope and protocol

Audit and measurements were run from this repository at the current working tree. The executed data is recorded in `experiment_manifest.csv`, generated into an isolated output directory from `/Users/anshulchandra/Desktop/visionguard/data/generated/research_seed2026/manifest.csv` with seed 2026. There are 60 labeled input samples, each paired with a clean reference (120 manifest-referenced images): 42 train, 6 validation, and 12 held-out test. There are 10 clean controls and 50 defective inputs. Overall defect counts are {'blur': 10, 'noise': 10, 'exposure': 10, 'colorcast': 10, 'low_contrast': 10}. The optional real-image ingestion path was not used because no real dataset was supplied. The manifest is the authoritative sample record.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| clean | 7 | 1 | 2 | 10 |
| blur | 6 | 1 | 3 | 10 |
| noise | 7 | 1 | 2 | 10 |
| exposure | 8 | 0 | 2 | 10 |
| colorcast | 7 | 2 | 1 | 10 |
| low_contrast | 7 | 1 | 2 | 10 |


The rule baseline maps the quality analyzer's flags directly to defect labels. The Random Forest uses nine hand-crafted DIP features and is trained only on train rows. The six-row validation split is currently counted but not used for tuning or model selection. The RF is an offline experiment; it is not called by the Streamlit inspector or the correction planner.

## Held-out detection results

| Method | Subset accuracy | Label accuracy | Macro precision | Macro recall | Macro F1 | Micro F1 | Label FPR | Label FNR | Clean-image false reject rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP rule baseline | 0.833 | 0.967 | 0.900 | 1.000 | 0.933 | 0.909 | 0.040 | 0.000 | 0.000 |
| Random Forest | 0.917 | 0.983 | 0.900 | 1.000 | 0.933 | 0.952 | 0.020 | 0.000 | 0.500 |

| Defect | Support | Rule precision / recall / F1 | RF precision / recall / F1 |
|---|---:|---:|---:|
| blur | 3 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| noise | 2 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| exposure | 2 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| colorcast | 1 | 1.000 / 1.000 / 1.000 | 0.500 / 1.000 / 0.667 |
| low_contrast | 2 | 0.500 / 1.000 / 0.667 | 1.000 / 1.000 / 1.000 |


The test split is very small; per-class scores are based on support between one and three examples. The RF's image-level false reject rate is 0.500 because it predicts a colour cast for one of the two clean controls. This rate is more interpretable for the two clean images than aggregate label FPR, but also highly uncertain.

## Correction / restoration

On the 10 defective samples with aligned clean references, mean heuristic quality score changed from 89.468 to 90.496 across the full 12-image test set. Restoration-only mean PSNR changed from 19.735 to 20.192 dB; mean SSIM changed from 0.7769 to 0.7965. The planner accepted 5 operations and rejected 3; it modified 0 clean controls. Clean-reference PSNR/SSIM is omitted for unchanged exact reference pairs and clean controls. IoU/Dice are not calculated: the dataset provides no ground-truth segmentation masks.

The accepted denoising cases reduced the measured residual-noise estimate but retained the noise flag. White balance cleared the colour-cast flag for `synthetic_0019`. Gamma was accepted for `synthetic_0028` and increased its exposure score from 0.787 to 0.933, but the exposure flag remained; the final status is partial improvement. Blur was left unchanged. CLAHE was rejected for `synthetic_0035` because its contrast score decreased slightly. These examples distinguish an accepted measurable target change from complete defect removal.

## Implemented algorithms and AI components

Code inspection confirms histogram-derived intensity evidence (mean/median, clipping, shadow/highlight fractions, local grid means), CLAHE on LAB luminance, gamma LUT correction, Laplacian variance and Tenengrad, multi-scale median-residual MAD noise estimation, median/bilateral/NLM denoisers, Otsu thresholding in two polarities, morphology, connected-component scoring, GrabCut refinement, and quality/compliance assessment. Classical operations are implemented in the source; only selected fixes are run automatically. Generic unsharp masking exists but automatic blur restoration is intentionally not applied.

The AI/CV learner is a multi-output scikit-learn Random Forest over nine engineered measurements. The `agent` is deterministic condition selection and guarded sequential correction, not an LLM or autonomous model-driven agent. There is no VLM, pretrained detector, external model, paid API, or GPU dependency. Streamlit AppTest coverage exists and passed as part of the repository suite.

## Audit findings and missing evidence

- Pytest: 15 passed. The model/evaluation held-out test contains 12 source groups.
- Dataset generation now cleans prior split folders only in the default generated-data root or a marker-owned output directory; it refuses to overwrite a non-empty unowned custom directory. The experiments used the isolated seed-2026 output and explicit manifest paths.
- The legacy root `data/generated` split folders may contain stale files from before this generator change. They were not read by these experiments; the isolated manifest is authoritative.
- Validation data exists but is not used for threshold tuning, calibration, or model selection. Thresholds remain hand-selected heuristics.
- The validation split has no positive exposure samples; it cannot support validation-based calibration for every defect category.
- The synthetic images are simple PIL shapes with one controlled corruption each; there is no independent real-product test set, multi-defect test, or annotator review.
- Segmentation has no pixel masks. Therefore IoU/Dice cannot be reported, despite the segmentation algorithm being implemented.
- SSIM/PSNR cover image correction only, not segmentation. Results are synthetic aligned-pair metrics and improvements are small overall.
- `research` metrics do not certify marketplace compliance; thresholds are illustrative.
- The supplied course PDFs exist, but could not be text-extracted in this environment: no `pdftotext`/Poppler or PDF Python parser was installed, and the available local Swift PDFKit attempt failed due to toolchain/module-cache errors. No requirements from those PDFs are asserted in this audit.
- The referenced real watch image is absent from the repository. Existing synthetic watch-like tests are regression fixtures, not real-image evidence.

## Reproduction

From repository root, with Python 3.11 and `requirements.txt` installed:

```bash
python -m data.generate_dataset --groups 60 --seed 2026 --out-dir data/generated/research_seed2026
python -m models.train --manifest data/generated/research_seed2026/manifest.csv
python -m eval.evaluate --manifest data/generated/research_seed2026/manifest.csv
python -m scripts.run_demo --manifest data/generated/research_seed2026/manifest.csv
python -m pytest -q
python research/build_research.py --manifest data/generated/research_seed2026/manifest.csv
```

The current observed run used Python 3.11.18. Raw per-image results, plots, case composites, and this summary are generated by `research/build_research.py` from the first five commands' outputs. No paper text is drafted here.
