# VisionGuard Project Report

**Project ID:** HSR-AIML-056  
**Title:** Automated E-commerce Image Quality and Compliance Inspector  
**Status:** Reproducible local MVP

## Abstract

VisionGuard measures common image defects and product-layout cues with classical digital image processing, then demonstrates deterministic correction and a Random Forest classifier using interpretable DIP features. Its purpose is to make intermediate image operations visible and measurable in a low-dependency local application. Compliance rules are illustrative and do not represent official marketplace requirements.

## Methodology

Inputs are RGB images (the quality API also accepts BGR arrays with an explicit order parameter). The quality stage measures grayscale Laplacian variance, Sobel gradient energy (Tenengrad), mean and median intensity, near-saturated clipping fraction, shadow/highlight fractions, 4x4 local brightness spread, RMS contrast, and resolution. Noise is estimated by the robust MAD scale `1.4826 * median(|r - median(r)|)` on pixels selected from the low-gradient portion of a smoothed guide image. Residuals from 3x3 and 5x5 median filters are combined by their median; confidence reflects flat-region support and agreement between scales. Noise classification is independent of exposure classification. The exposure score includes the same midtone evidence used by its defect flag; a color-balance score component supports cast correction. Component scores are normalized and weighted; contrast contribution is reduced when exposure is flagged to limit correlated penalties. Cutoffs and weights are exposed by `QualityThresholds` and should be tuned on validation data.

Enhancement operations include CLAHE on LAB L, gamma lookup-table correction, median and bilateral filtering, unsharp masking and gray-world white balance. Operations are individually callable and intermediate outputs can be saved. Conservative clipping and fixed strengths limit aggressive changes.

Segmentation estimates a background colour from border pixels in LAB, thresholds distance with Otsu in both polarities, applies gentle closing and small-component cleanup, and scores connected components for area plausibility, border contact and disconnected residue. GrabCut refinement is confidence-gated. The report includes bounding box, normalized centroid, foreground coverage, center score, border colour statistics, confidence, warnings and an explicit fallback marker. Low confidence is also visible as a presentation-rule review failure.

The correction planner denoises first when noise is detected, then considers gamma, white balance and CLAHE only when associated defects are measured. Blur is reported but not auto-sharpened. Each attempted correction records its target metric, aggregate score change, quality flags, SSIM-to-original, guard results and reason. An operation must show a meaningful target improvement, exceed the minimum aggregate score gain, avoid new quality defects, preserve a minimum SSIM, and avoid regressing previously passing presentation compliance. Rejected candidates are discarded, and input arrays are never modified. These are transparent heuristics rather than perceptual guarantees.

## Dataset and split

`python -m data.generate_dataset` creates deterministic PIL-drawn product-like shapes on a light background. Each source group gets a clean image and either no defect or one injected defect: Gaussian blur, Gaussian noise, underexposure, red channel cast, or reduced contrast. A fixed seed and stratified grouped split keep each source in exactly one of train (70%), validation (10%) or test (20%). The default is 60 groups. There are no external or human-subject images. The optional local-image ingest path should be described separately in any submission and its synthetic labels do not constitute real-image annotations.

## Learning and comparison

The learner is a multi-output Random Forest using nine DIP measurements. It is trained on the train split and evaluated on held-out source groups; a deterministic rule baseline is evaluated alongside it. The evaluation writes sample counts, per-defect precision/recall/F1, macro/micro F1, confusion matrices, false-positive/false-negative rates, correction counts, clean-control modification counts and image-level quality changes. PSNR/SSIM are computed only for aligned defective images with a clean reference; clean unchanged controls are excluded from restoration averages.

## Evaluation protocol and outputs

Run generator, training, evaluation and demo commands from the repository root. `outputs/evaluation/results.csv` has per-image scores; `metrics.json` has aggregate values, and `quality_comparison.png` visualizes quality before and after correction. The first five held-out samples are saved as original/fixed/clean triptychs in `outputs/evaluation/examples/`. PSNR and SSIM are calculated only for aligned synthetic pairs. No IoU/Dice result is reported because the generator does not provide true object masks. False reject rate is the fraction of clean test images flagged as defective by the rule baseline.

## Results from the generated run

The latest reproducible run used 60 source groups: 42 train, 6 validation and 12 test. The manifest contains 10 clean controls and 50 defective images; the test split contains 2 clean and 10 defective images. Results below come from `outputs/model/metrics.json` and `outputs/evaluation/metrics.json`.

| Test method | Precision (macro) | Recall (macro) | F1 (macro) | F1 (micro) | False positive rate | False negative rate |
|---|---:|---:|---:|---:|---:|---:|
| DIP rule baseline | 0.900 | 1.000 | 0.933 | 0.909 | 0.040 | 0.000 |
| Random Forest | 0.900 | 1.000 | 0.933 | 0.952 | 0.020 | 0.000 |

Per-defect Random Forest F1 on this small held-out sample was 1.000 for blur (3 samples), noise (2), exposure (2), and low contrast (2); colour cast F1 was 0.667 (1 sample). The rule baseline had F1 0.667 for low contrast (2 samples) and 1.000 for the other classes in this split. These support counts are too small to infer general performance.

| Correction measure | Before | After |
|---|---:|---:|
| Mean quality score | 89.375 | 89.755 |
| Mean PSNR on 10 aligned defective references (dB) | 19.735 | 19.771 |
| Mean SSIM on 10 aligned defective references | 0.7769 | 0.7787 |

The planner accepted 2 corrections and rejected 7 attempted operations; 0 clean test controls were unnecessarily modified. The false-reject rate on the 2 clean controls was 0.000. The quality score and restoration metrics are heuristic/task-specific and only slightly changed in this run. Results are reproducible for the fixed seed but depend on the small synthetic dataset and its particular split.

## Success and failure analysis

Use the generated `largest_quality_gains`, `lowest_final_quality` and `ssim_decreased_cases` fields in `outputs/evaluation/metrics.json`, plus the per-image CSV and example triptychs. These are measured candidate cases, not claims of overall success: a higher heuristic quality score can coexist with a lower SSIM, which is why both are reported. Likely failure modes include blur, since lost detail cannot be reconstructed; underexposure, where gamma correction cannot restore clipped or absent information; and segmentation of light or irregular products against a similarly light background. The synthetic data does not test reflective, transparent, textured, cluttered, or real catalog images. Discuss the identified filenames with their actual per-image measurements; do not infer generalization from this small dataset.

## Reproducibility, limitations and next steps

All generation seeds are fixed and the split is by source group. Thresholds are explicit in `dip/quality.py` and `dip/compliance.py`, not claimed as official or tuned values. Limitations include simple geometric products, one synthetic defect per sample, a small dataset, heuristic feature detection, no segmentation ground-truth masks, and no reliable blur recovery. A larger, documented real-product test set with manual defect and mask labels is required before making real-world performance claims.
