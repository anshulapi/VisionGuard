"""Held-out defect detection and correction evaluation with source-group isolation."""
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from agent.planner import auto_fix
from models.features import extract_features
from models.train import TARGETS

ROOT = Path(__file__).resolve().parents[1]


def _safe_psnr(reference: np.ndarray, image: np.ndarray) -> float | None:
    if np.array_equal(reference, image):
        return None
    value = float(peak_signal_noise_ratio(reference, image, data_range=255))
    return value if np.isfinite(value) else None


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, target_names=TARGETS, zero_division=0, output_dict=True)
    matrices = {name: confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).tolist() for i, name in enumerate(TARGETS)}
    tn, fp, fn, tp = (0, 0, 0, 0)
    for matrix in matrices.values():
        tn += matrix[0][0]; fp += matrix[0][1]; fn += matrix[1][0]; tp += matrix[1][1]
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "label_accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "per_defect": {name: {metric: float(report[name][metric]) for metric in ("precision", "recall", "f1-score", "support")} for name in TARGETS},
        "confusion_matrices": matrices,
    }


def evaluate(manifest: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest) if manifest else ROOT / "data/generated/manifest.csv"
    frame = pd.read_csv(manifest_path)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise ValueError("The manifest has no held-out test rows")

    model_path = ROOT / "outputs/model/visionguard_rf.joblib"
    bundle = joblib.load(model_path) if model_path.exists() else None
    rows: list[dict[str, Any]] = []
    y_values: list[np.ndarray] = []
    rule_values: list[np.ndarray] = []
    model_values: list[np.ndarray] = []
    for _, row in test.iterrows():
        image_path = ROOT / row["image_path"]
        clean_path = ROOT / row["clean_path"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        clean = np.asarray(Image.open(clean_path).convert("RGB"))
        fixed, trace = auto_fix(image)
        q_before = trace["before"]["quality"]
        q_after = trace["after"]["quality"]
        truth = np.array([int(row[f"label_{label}"]) for label in TARGETS], dtype=int)
        rule_pred = np.array([int(label in q_before["flags"]) for label in TARGETS], dtype=int)
        y_values.append(truth); rule_values.append(rule_pred)
        model_pred = None
        if bundle:
            features = extract_features(image).reshape(1, -1)
            model_pred = bundle["model"].predict(features)[0].astype(int)
            model_values.append(model_pred)

        operations = trace["operations"]
        accepted = sum(bool(operation["accepted"]) for operation in operations)
        rejected = len(operations) - accepted
        clean_control = bool(truth.sum() == 0)
        restored = bool(truth.sum() > 0)
        if restored:
            psnr_before = _safe_psnr(clean, image)
            psnr_after = _safe_psnr(clean, fixed)
            ssim_before = float(structural_similarity(clean, image, channel_axis=2, data_range=255))
            ssim_after = float(structural_similarity(clean, fixed, channel_axis=2, data_range=255))
        else:
            psnr_before = psnr_after = ssim_before = ssim_after = np.nan
        row_result = {
            "image_path": row["image_path"], "group_id": row["group_id"],
            "ground_truth_defects": "|".join(label for label, value in zip(TARGETS, truth) if value),
            "rule_defects": "|".join(label for label, value in zip(TARGETS, rule_pred) if value),
            "model_defects": "|".join(label for label, value in zip(TARGETS, model_pred if model_pred is not None else []) if value),
            "clean_control": clean_control, "before_quality": q_before["quality_score"],
            "after_quality": q_after["quality_score"], "before_flags": "|".join(q_before["flags"]),
            "after_flags": "|".join(q_after["flags"]), "operations_attempted": len(operations),
            "operations_accepted": accepted, "operations_rejected": rejected,
            "clean_unnecessarily_modified": bool(clean_control and accepted > 0),
            "psnr_before": psnr_before, "psnr_after": psnr_after,
            "ssim_before": ssim_before, "ssim_after": ssim_after,
        }
        rows.append(row_result)
        if len(rows) <= 5:
            example_dir = ROOT / "outputs/evaluation/examples"
            example_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.hstack((image, fixed, clean))).save(example_dir / f"example_{len(rows):02d}.png")

    y = np.asarray(y_values); rule = np.asarray(rule_values)
    rule_metrics = _classification_metrics(y, rule)
    model_metrics = _classification_metrics(y, np.asarray(model_values)) if model_values else None
    result_frame = pd.DataFrame(rows)
    out_dir = ROOT / "outputs/evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(out_dir / "results.csv", index=False)
    clean_rows = result_frame["clean_control"]
    corrupted_rows = ~clean_rows
    report = {
        "dataset_size": int(len(frame)), "test_size": int(len(test)),
        "clean_images": int(frame[[f"label_{label}" for label in TARGETS]].sum(axis=1).eq(0).sum()),
        "defective_images": int(frame[[f"label_{label}" for label in TARGETS]].sum(axis=1).gt(0).sum()),
        "test_clean_images": int(clean_rows.sum()), "test_defective_images": int(corrupted_rows.sum()),
        "n_restoration_pairs": int(result_frame.loc[corrupted_rows, "psnr_before"].notna().sum()),
        "rule_baseline": rule_metrics, "random_forest": model_metrics,
        # Backward-compatible aliases retained from the previous evaluation JSON.
        "baseline_precision_macro": rule_metrics["precision_macro"],
        "baseline_recall_macro": rule_metrics["recall_macro"],
        "baseline_f1_macro": rule_metrics["f1_macro"],
        "baseline_f1_micro": rule_metrics["f1_micro"],
        "false_reject_rate": float(np.mean(rule[clean_rows.to_numpy()].sum(axis=1) > 0)) if clean_rows.any() else None,
        "mean_quality_before": float(result_frame["before_quality"].mean()),
        "mean_quality_after": float(result_frame["after_quality"].mean()),
        "mean_psnr_before": float(result_frame.loc[corrupted_rows, "psnr_before"].mean()) if result_frame.loc[corrupted_rows, "psnr_before"].notna().any() else None,
        "mean_psnr_after": float(result_frame.loc[corrupted_rows, "psnr_after"].mean()) if result_frame.loc[corrupted_rows, "psnr_after"].notna().any() else None,
        "mean_ssim_before": float(result_frame.loc[corrupted_rows, "ssim_before"].mean()) if corrupted_rows.any() else None,
        "mean_ssim_after": float(result_frame.loc[corrupted_rows, "ssim_after"].mean()) if corrupted_rows.any() else None,
        "accepted_corrections": int(result_frame["operations_accepted"].sum()),
        "rejected_corrections": int(result_frame["operations_rejected"].sum()),
        "clean_images_unnecessarily_modified": int(result_frame["clean_unnecessarily_modified"].sum()),
        "success_candidates_by_quality_gain": result_frame.assign(gain=result_frame.after_quality - result_frame.before_quality).nlargest(3, "gain")["image_path"].tolist(),
        "lowest_final_quality_candidates": result_frame.nsmallest(3, "after_quality")["image_path"].tolist(),
        "limitations": "Small synthetic grouped dataset; the watch JPEG mentioned in the issue was not present in the repository. PSNR/SSIM are restricted to aligned corrupted samples with clean references. Compliance thresholds are illustrative.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Before", "After"], [report["mean_quality_before"], report["mean_quality_after"]])
    axes[0].set_title("Mean quality score")
    axes[1].bar(["Before", "After"], [report["mean_ssim_before"] or 0, report["mean_ssim_after"] or 0])
    axes[1].set_title("Mean SSIM on corrupted references")
    fig.tight_layout(); fig.savefig(out_dir / "quality_comparison.png", dpi=150); plt.close(fig)
    (out_dir / "confusion_matrices.json").write_text(json.dumps({"rule_baseline": rule_metrics["confusion_matrices"], "random_forest": model_metrics["confusion_matrices"] if model_metrics else None}, indent=2))
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Dataset manifest path (defaults to data/generated/manifest.csv)")
    evaluate(parser.parse_args().manifest)
