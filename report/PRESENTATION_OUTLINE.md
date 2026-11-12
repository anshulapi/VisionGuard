# VisionGuard — 7-slide presentation outline

1. **Problem and objective** — HSR-AIML-056; inspect product quality and explain illustrative compliance outcomes. State that synthetic data is the reproducible MVP dataset.
2. **Pipeline architecture** — Show `ARCHITECTURE_IMPLEMENTED.md` flow: quality metrics, segmentation/morphology, learner, deterministic fixer and reinspection.
3. **Classical DIP methods** — Laplacian variance, Tenengrad, robust noise residual, LAB CLAHE, gamma and Otsu + morphology. Show generated original and intermediate mask / CLAHE images.
4. **Data and protocol** — Explain PIL shape generation, five injected defects, fixed seed, group-separated 70/10/20 split, and test-only reporting.
5. **Baseline vs learned method** — Populate actual macro/micro F1 and classification report from `outputs/model/metrics.json`; explain weak labels and rule limitations.
6. **Auto-fix and evaluation** — Show an actual before/after triptych, operation log, quality score and aligned PSNR/SSIM from `outputs/evaluation/`.
7. **Failure analysis and conclusion** — Select actual measured successes and failures; highlight blur irreversibility, white-on-white segmentation and synthetic-to-real limits. List next step: manually labeled real catalog sample.
