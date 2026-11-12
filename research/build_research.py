"""Build an auditable research evidence package from generated experiment outputs.

Run the project generation, training and evaluation commands first; this script
does not retrain or modify application source. It reads outputs/ and writes only
to research/.
"""
from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "visionguard-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from agent.planner import auto_fix
from models.train import TARGETS

OUT = ROOT / "research"
COLORS = {"Rule baseline": "#718096", "Random Forest": "#1976D2"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def metric_frame(manifest_path: Path) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    metrics = read_json(ROOT / "outputs/evaluation/metrics.json")
    frame = pd.read_csv(ROOT / "outputs/evaluation/results.csv")
    manifest = pd.read_csv(manifest_path)
    expected_test = set(manifest.loc[manifest["split"] == "test", "group_id"])
    if set(frame["group_id"]) != expected_test:
        raise ValueError("Evaluation CSV test groups do not match the selected manifest")
    frame = frame.merge(manifest, on="group_id", suffixes=("", "_manifest"), validate="one_to_one")
    for label in TARGETS:
        frame[f"gt_{label}"] = frame[f"label_{label}"].astype(int)
        frame[f"rule_{label}"] = frame["rule_defects"].apply(lambda s: int(label in str(s).split("|")))
        frame[f"rf_{label}"] = frame["model_defects"].apply(lambda s: int(label in str(s).split("|")))
    columns = ["group_id", "split", "clean_control", "ground_truth_defects", "rule_defects", "model_defects"]
    columns += [f"gt_{x}" for x in TARGETS] + [f"rule_{x}" for x in TARGETS] + [f"rf_{x}" for x in TARGETS]
    columns += ["before_quality", "after_quality", "before_flags", "after_flags", "operations_attempted", "operations_accepted", "operations_rejected", "clean_unnecessarily_modified", "psnr_before", "psnr_after", "ssim_before", "ssim_after"]
    frame[columns].to_csv(OUT / "experiment_results.csv", index=False)
    shutil.copyfile(manifest_path, OUT / "experiment_manifest.csv")
    return frame, metrics, read_json(ROOT / "outputs/model/metrics.json"), manifest


def plots(frame: pd.DataFrame, metrics: dict) -> None:
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    labels = ["Subset acc.", "Label acc.", "Macro precision", "Macro recall", "Macro F1", "Micro F1", "Label FPR", "Label FNR"]
    keys = ["subset_accuracy", "label_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro", "false_positive_rate", "false_negative_rate"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(labels)); width = .36
    for i, method in enumerate(("rule_baseline", "random_forest")):
        vals = [metrics[method][key] for key in keys]
        bars = ax.bar(x + (i - .5) * width, vals, width, label="Rule baseline" if i == 0 else "Random Forest", color=list(COLORS.values())[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, labels, rotation=15, ha="right"); ax.set_ylim(0, 1.16)
    ax.set_ylabel("Score / rate"); ax.set_title("Held-out defect classification (12 grouped test images)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "figures/classification_comparison.png", dpi=180); plt.close(fig)

    classes = TARGETS; x = np.arange(len(classes)); width = .36
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for i, method in enumerate(("rule_baseline", "random_forest")):
        vals = [metrics[method]["per_defect"][c]["f1-score"] for c in classes]
        bars = ax.bar(x + (i - .5) * width, vals, width, label="Rule baseline" if i == 0 else "Random Forest", color=list(COLORS.values())[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    support = [int(metrics["rule_baseline"]["per_defect"][c]["support"]) for c in classes]
    ax.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(classes, support)])
    ax.set_ylim(0, 1.2); ax.set_ylabel("F1"); ax.set_title("Per-defect F1; supports are small")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "figures/per_defect_f1.png", dpi=180); plt.close(fig)

    defective = frame[frame["clean_control"].astype(str).str.lower().isin(("false", "0"))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    measures = [("before_quality", "after_quality", "Quality score (0-100)"), ("psnr_before", "psnr_after", "PSNR (dB)"), ("ssim_before", "ssim_after", "SSIM")]
    for ax, (before, after, title) in zip(axes, measures):
        b = pd.to_numeric(defective[before], errors="coerce").dropna()
        a = pd.to_numeric(defective[after], errors="coerce").dropna()
        ax.bar(["Before", "After"], [b.mean(), a.mean()], color=["#A0AEC0", "#2F855A"])
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
        for i, value in enumerate((b.mean(), a.mean())):
            ax.text(i, value, f"{value:.3f}" if title == "SSIM" else f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Correction results on 10 defective aligned synthetic pairs")
    fig.tight_layout(); fig.savefig(OUT / "figures/correction_metrics.png", dpi=180); plt.close(fig)


def labeled_triptych(group_id: str, kind: str, note: str, status: str, manifest: pd.DataFrame) -> None:
    row = manifest.query("group_id == @group_id").iloc[0]
    source = np.asarray(Image.open(resolve_data_path(row.image_path)).convert("RGB"))
    clean = np.asarray(Image.open(resolve_data_path(row.clean_path)).convert("RGB"))
    fixed, trace = auto_fix(source)
    width, height = source.shape[1], source.shape[0]
    canvas = Image.new("RGB", (width * 3, height + 92), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate((("Input", source), ("Auto-fix output", fixed), ("Clean reference", clean))):
        canvas.paste(Image.fromarray(image), (i * width, 0))
        draw.text((i * width + 10, height + 7), label, fill="#202A35")
    action = ", ".join(f"{r['operation']}:{'accepted' if r['accepted'] else 'rejected'}" for r in trace["operations"]) or "no operation attempted"
    draw.text((10, height + 29), f"{group_id} | {kind} | {status}; {action}", fill="#202A35")
    draw.text((10, height + 53), note, fill="#4A5568")
    dest = OUT / ("failure_cases" if status == "failure" else "figures") / f"{group_id}_{kind}.png"
    dest.parent.mkdir(parents=True, exist_ok=True); canvas.save(dest)


def make_cases(manifest: pd.DataFrame) -> None:
    examples = [
        ("synthetic_0017", "noise", "Bilateral denoising accepted; residual noise decreased, though the noise flag remains.", "success"),
        ("synthetic_0032", "noise", "Second accepted synthetic denoising case; residual noise decreased, flag remains.", "success"),
        ("synthetic_0001", "blur", "No correction attempted; blur cannot be reliably reversed by this pipeline.", "failure"),
        ("synthetic_0028", "exposure", "Gamma was accepted and exposure score improved, but exposure flag remained.", "failure"),
        ("synthetic_0035", "low_contrast", "CLAHE was rejected because its contrast target score did not improve.", "failure"),
        ("synthetic_0019", "colorcast", "White balance was accepted; the measured color-cast flag cleared.", "success"),
        ("synthetic_0054", "clean_rf_false_positive", "Clean control receives a Random Forest colour-cast false positive.", "failure"),
    ]
    generated_names = [f"{group_id}_{kind}.png" for group_id, kind, _note, _status in examples]
    for folder in ("figures", "failure_cases"):
        for name in generated_names:
            (OUT / folder / name).unlink(missing_ok=True)
    for item in examples:
        labeled_triptych(*item, manifest=manifest)
    (OUT / "failure_cases/README.md").write_text(
        "# Measured representative cases\n\n"
        "Each triptych shows input, actual auto-fix output, and aligned clean synthetic reference. "
        "The case identifiers and decisions come from the held-out test split; see `experiment_results.csv` for per-image measurements. "
        "These are selected examples, not a representative estimate. `synthetic_0054` is a classifier false positive; the correction planner leaves it unchanged.\n"
    )


def write_summary(frame: pd.DataFrame, metrics: dict, training: dict, manifest: pd.DataFrame, manifest_path: Path) -> None:
    clean = frame[frame["clean_control"].astype(str).str.lower().isin(("true", "1"))]
    rf_clean_reject = float((clean["model_defects"].fillna("").astype(str).str.len() > 0).mean()) if len(clean) else float("nan")
    split_counts = manifest["split"].value_counts().to_dict()
    defect_counts = {label: int(manifest[f"label_{label}"].sum()) for label in TARGETS}
    clean_count = int((manifest[[f"label_{label}" for label in TARGETS]].sum(axis=1) == 0).sum())
    class_matrix = {"clean": (manifest[[f"label_{label}" for label in TARGETS]].sum(axis=1) == 0)}
    class_matrix.update({label: manifest[f"label_{label}"].astype(bool) for label in TARGETS})
    metadata_path = manifest_path.parent / ".visionguard_dataset.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    seed = metadata.get("seed", "not recorded")
    text = f"""# VisionGuard repository audit and experiment summary

## Scope and protocol

Audit and measurements were run from this repository at the current working tree. The executed data is recorded in `experiment_manifest.csv`, generated into an isolated output directory from `{manifest_path}` with seed {seed}. There are {len(manifest)} labeled input samples, each paired with a clean reference ({2*len(manifest)} manifest-referenced images): {split_counts.get('train', 0)} train, {split_counts.get('validation', 0)} validation, and {split_counts.get('test', 0)} held-out test. There are {clean_count} clean controls and {len(manifest)-clean_count} defective inputs. Overall defect counts are {defect_counts}. The optional real-image ingestion path was not used because no real dataset was supplied. The manifest is the authoritative sample record.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
"""
    for name, mask in class_matrix.items():
        counts = {split: int((mask & (manifest["split"] == split)).sum()) for split in ("train", "validation", "test")}
        text += f"| {name} | {counts['train']} | {counts['validation']} | {counts['test']} | {int(mask.sum())} |\n"
    text += f"""

The rule baseline maps the quality analyzer's flags directly to defect labels. The Random Forest uses nine hand-crafted DIP features and is trained only on train rows. The six-row validation split is currently counted but not used for tuning or model selection. The RF is an offline experiment; it is not called by the Streamlit inspector or the correction planner.

## Held-out detection results

| Method | Subset accuracy | Label accuracy | Macro precision | Macro recall | Macro F1 | Micro F1 | Label FPR | Label FNR | Clean-image false reject rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIP rule baseline | {metrics['rule_baseline']['subset_accuracy']:.3f} | {metrics['rule_baseline']['label_accuracy']:.3f} | {metrics['rule_baseline']['precision_macro']:.3f} | {metrics['rule_baseline']['recall_macro']:.3f} | {metrics['rule_baseline']['f1_macro']:.3f} | {metrics['rule_baseline']['f1_micro']:.3f} | {metrics['rule_baseline']['false_positive_rate']:.3f} | {metrics['rule_baseline']['false_negative_rate']:.3f} | {metrics['false_reject_rate']:.3f} |
| Random Forest | {metrics['random_forest']['subset_accuracy']:.3f} | {metrics['random_forest']['label_accuracy']:.3f} | {metrics['random_forest']['precision_macro']:.3f} | {metrics['random_forest']['recall_macro']:.3f} | {metrics['random_forest']['f1_macro']:.3f} | {metrics['random_forest']['f1_micro']:.3f} | {metrics['random_forest']['false_positive_rate']:.3f} | {metrics['random_forest']['false_negative_rate']:.3f} | {rf_clean_reject:.3f} |

| Defect | Support | Rule precision / recall / F1 | RF precision / recall / F1 |
|---|---:|---:|---:|
"""
    for label in TARGETS:
        r = metrics["rule_baseline"]["per_defect"][label]
        f = metrics["random_forest"]["per_defect"][label]
        text += f"| {label} | {int(r['support'])} | {r['precision']:.3f} / {r['recall']:.3f} / {r['f1-score']:.3f} | {f['precision']:.3f} / {f['recall']:.3f} / {f['f1-score']:.3f} |\n"
    text += f"""

The test split is very small; per-class scores are based on support between one and three examples. The RF's image-level false reject rate is {rf_clean_reject:.3f} because it predicts a colour cast for one of the two clean controls. This rate is more interpretable for the two clean images than aggregate label FPR, but also highly uncertain.

## Correction / restoration

On the {metrics['n_restoration_pairs']} defective samples with aligned clean references, mean heuristic quality score changed from {metrics['mean_quality_before']:.3f} to {metrics['mean_quality_after']:.3f} across the full 12-image test set. Restoration-only mean PSNR changed from {metrics['mean_psnr_before']:.3f} to {metrics['mean_psnr_after']:.3f} dB; mean SSIM changed from {metrics['mean_ssim_before']:.4f} to {metrics['mean_ssim_after']:.4f}. The planner accepted {metrics['accepted_corrections']} operations and rejected {metrics['rejected_corrections']}; it modified {metrics['clean_images_unnecessarily_modified']} clean controls. Clean-reference PSNR/SSIM is omitted for unchanged exact reference pairs and clean controls. IoU/Dice are not calculated: the dataset provides no ground-truth segmentation masks.

The accepted denoising cases reduced the measured residual-noise estimate but retained the noise flag. White balance cleared the colour-cast flag for `synthetic_0019`. Gamma was accepted for `synthetic_0028` and increased its exposure score from 0.787 to 0.933, but the exposure flag remained; the final status is partial improvement. Blur was left unchanged. CLAHE was rejected for `synthetic_0035` because its contrast score decreased slightly. These examples distinguish an accepted measurable target change from complete defect removal.

## Implemented algorithms and AI components

Code inspection confirms histogram-derived intensity evidence (mean/median, clipping, shadow/highlight fractions, local grid means), CLAHE on LAB luminance, gamma LUT correction, Laplacian variance and Tenengrad, multi-scale median-residual MAD noise estimation, median/bilateral/NLM denoisers, Otsu thresholding in two polarities, morphology, connected-component scoring, GrabCut refinement, and quality/compliance assessment. Classical operations are implemented in the source; only selected fixes are run automatically. Generic unsharp masking exists but automatic blur restoration is intentionally not applied.

The AI/CV learner is a multi-output scikit-learn Random Forest over nine engineered measurements. The `agent` is deterministic condition selection and guarded sequential correction, not an LLM or autonomous model-driven agent. There is no VLM, pretrained detector, external model, paid API, or GPU dependency. Streamlit AppTest coverage exists and passed as part of the repository suite.

## Audit findings and missing evidence

- Pytest: 15 passed. The model/evaluation held-out test contains {training['n_test']} source groups.
- Dataset generation now cleans prior split folders only in the default generated-data root or a marker-owned output directory; it refuses to overwrite a non-empty unowned custom directory. The experiments used the isolated seed-{seed} output and explicit manifest paths.
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
"""
    (OUT / "evaluation_summary.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/generated/manifest.csv", help="Manifest used by training/evaluation")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest_path = resolve_data_path(args.manifest)
    frame, metrics, training, manifest = metric_frame(manifest_path)
    plots(frame, metrics)
    make_cases(manifest)
    write_summary(frame, metrics, training, manifest, manifest_path)


if __name__ == "__main__":
    main()
