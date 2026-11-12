"""Deterministic correction planner with defect-specific and perceptual guardrails."""
from dataclasses import dataclass
import logging
from typing import Any, Callable

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from dip.compliance import ComplianceRules, inspect_compliance
from dip.enhancement import bilateral_denoise, clahe_lab, gamma_correct, white_balance
from dip.quality import QualityThresholds, analyze_quality
from dip.segmentation import segment_product

logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    min_quality_improvement: float = 0.15
    min_ssim_to_original: float = 0.90
    noise_reduction_fraction: float = 0.08
    min_exposure_score_gain: float = 0.02
    min_contrast_score_gain: float = 0.03
    min_noise_confidence: float = 0.35


def inspect(
    image: np.ndarray,
    thresholds: QualityThresholds | None = None,
    rules: ComplianceRules | None = None,
) -> dict[str, Any]:
    """Run quality, segmentation and separate technical/presentation checks."""
    quality = analyze_quality(image, thresholds)
    segmentation = segment_product(image, refine=True)
    compliance = inspect_compliance(image, quality, segmentation, rules)
    return {"quality": quality, "segmentation": segmentation, "compliance": compliance}


def _target_improved(name: str, before: dict[str, Any], after: dict[str, Any], config: PlannerConfig) -> tuple[bool, str]:
    bq, aq = before["quality"], after["quality"]
    if name == "bilateral_denoise":
        target = bq["noise_sigma"] * (1 - config.noise_reduction_fraction)
        return aq["noise_sigma"] <= target, f"noise sigma {bq['noise_sigma']:.2f} → {aq['noise_sigma']:.2f}"
    if name == "gamma_correct":
        gain = aq["scores"]["exposure"] - bq["scores"]["exposure"]
        clipped_not_worse = aq["clipping_fraction"] <= bq["clipping_fraction"] + 0.005
        return gain >= config.min_exposure_score_gain and clipped_not_worse, f"exposure score {bq['scores']['exposure']:.3f} → {aq['scores']['exposure']:.3f}; clipping {bq['clipping_fraction']:.3f} → {aq['clipping_fraction']:.3f}"
    if name == "clahe_lab":
        gain = aq["scores"]["contrast"] - bq["scores"]["contrast"]
        noise_not_worse = aq["noise_sigma"] <= bq["noise_sigma"] + 0.5
        return gain >= config.min_contrast_score_gain and noise_not_worse, f"contrast score {bq['scores']['contrast']:.3f} → {aq['scores']['contrast']:.3f}; noise sigma {bq['noise_sigma']:.2f} → {aq['noise_sigma']:.2f}"
    if name == "white_balance":
        gain = bq["border_channel_spread"] - aq["border_channel_spread"]
        return gain >= 1.0, f"border channel spread {bq['border_channel_spread']:.2f} → {aq['border_channel_spread']:.2f}"
    return False, "No defect-specific acceptance criterion is defined."


def _similarity_to_original(original: np.ndarray, candidate: np.ndarray) -> float:
    original_gray = cv2.cvtColor(original[:, :, :3], cv2.COLOR_RGB2GRAY) if original.ndim == 3 else original
    candidate_gray = cv2.cvtColor(candidate[:, :, :3], cv2.COLOR_RGB2GRAY) if candidate.ndim == 3 else candidate
    if min(original_gray.shape[:2]) < 3:
        return float(np.clip(1.0 - np.mean(np.abs(original_gray.astype(np.float32) - candidate_gray.astype(np.float32))) / 255.0, 0, 1))
    return float(structural_similarity(original_gray, candidate_gray, data_range=255))


def auto_fix(
    image: np.ndarray,
    thresholds: QualityThresholds | None = None,
    rules: ComplianceRules | None = None,
    config: PlannerConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attempt only diagnosed, conservative fixes and preserve the input on rejection.

    Every attempted operation must improve its target metric, increase the weighted
    quality score by the configured minimum, avoid new defects, preserve previously
    passing presentation compliance, and remain perceptually similar to the original.
    Blur is diagnosed but not auto-sharpened because lost detail cannot be recovered.
    """
    cfg = config or PlannerConfig()
    original = np.asarray(image).copy()
    if original.size == 0:
        raise ValueError("Cannot auto-fix an empty image")
    current = original.copy()
    initial = inspect(current, thresholds, rules)
    before = initial
    history: list[dict[str, Any]] = []
    recommendations: list[str] = []
    flags = set(initial["quality"]["flags"])
    operations: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []

    # Grain is handled first so exposure/contrast operations do not amplify it.
    if "noise" in flags:
        if initial["quality"]["noise_confidence"] >= cfg.min_noise_confidence:
            operations.append(("bilateral_denoise", lambda im: bilateral_denoise(im, d=5, sigma_color=10, sigma_space=10)))
        else:
            recommendations.append("Noise estimate has low confidence; review the image before denoising.")
    if "exposure" in flags:
        median = initial["quality"]["median_brightness"]
        if median < 126:
            # Bounded gamma keeps highlights under control while lifting dark midtones.
            gamma = float(np.clip(np.log(0.55) / np.log(max(median, 1) / 255.0), 0.68, 0.88))
        else:
            gamma = 1.12
        operations.append(("gamma_correct", lambda im, g=gamma: gamma_correct(im, g)))
    if "colorcast" in flags:
        operations.append(("white_balance", white_balance))
    if "low_contrast" in flags:
        if "noise" not in flags or "bilateral_denoise" in [name for name, _ in operations]:
            operations.append(("clahe_lab", lambda im: clahe_lab(im, clip_limit=1.5)))
        else:
            recommendations.append("CLAHE skipped because detected noise could be amplified.")
    if "blur" in flags:
        recommendations.append("Blur detected. Automatic sharpening is disabled because it cannot restore lost source detail reliably.")

    for name, operation in operations:
        operation_before = before
        target_flag = {
            "bilateral_denoise": "noise", "gamma_correct": "exposure",
            "white_balance": "colorcast", "clahe_lab": "low_contrast",
        }[name]
        if target_flag not in operation_before["quality"]["flags"]:
            recommendations.append(f"Skipped {name}: its target defect was already resolved by an earlier accepted operation.")
            continue
        if name == "clahe_lab" and "noise" in operation_before["quality"]["flags"]:
            recommendations.append("CLAHE skipped because measurable noise remains after the denoising attempt.")
            continue
        candidate = operation(current)
        after = inspect(candidate, thresholds, rules)
        score_delta = after["quality"]["quality_score"] - before["quality"]["quality_score"]
        similarity = _similarity_to_original(original, candidate)
        target_ok, target_report = _target_improved(name, before, after, cfg)
        new_flags = sorted(set(after["quality"]["flags"]) - set(before["quality"]["flags"]))
        presentation_regressed = before["compliance"]["presentation_compliance"]["passed"] and not after["compliance"]["presentation_compliance"]["passed"]
        checks = {
            "target_improved": target_ok,
            "minimum_quality_gain": score_delta >= cfg.min_quality_improvement,
            "no_new_quality_defects": not new_flags,
            "similarity_guard": similarity >= cfg.min_ssim_to_original,
            "compliance_guard": not presentation_regressed,
        }
        accepted = all(checks.values())
        if accepted:
            current = candidate.copy()
            before = after
            reason = "Accepted: target metric improved and all quality, similarity and compliance guards passed."
        else:
            failures = [key for key, passed in checks.items() if not passed]
            reason = "Rejected: " + ", ".join(failures) + "."
        history.append({
            "operation": name, "accepted": accepted,
            "quality_before": operation_before["quality"]["quality_score"],
            "quality_after": after["quality"]["quality_score"], "quality_delta": float(score_delta),
            "target_metrics": target_report, "flags_before": operation_before["quality"]["flags"],
            "flags_after": after["quality"]["flags"], "new_flags": new_flags,
            "similarity_to_original": similarity, "checks": checks, "reason": reason,
        })
        logger.info("Auto-fix %s: %s (score delta %.3f)", name, "accepted" if accepted else "rejected", score_delta)

    final = inspect(current, thresholds, rules)
    accepted_any = any(row["accepted"] for row in history)
    remaining_initial_flags = sorted(set(initial["quality"]["flags"]) & set(final["quality"]["flags"]))
    if accepted_any and remaining_initial_flags:
        recommendations.append("Some originally detected defects remain after the accepted corrections: " + ", ".join(remaining_initial_flags) + ".")
    if not history and not recommendations:
        recommendations.append("No safe correction was indicated; image left unchanged.")
    report = {
        "before": initial, "after": final, "operations": history,
        "recommendations": recommendations,
        "status": ("corrected" if not remaining_initial_flags else "partially_improved") if accepted_any else "unchanged",
    }
    logger.debug("Inspection status=%s flags=%s", report["status"], final["quality"]["flags"])
    return current, report
