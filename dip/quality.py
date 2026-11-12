"""Explainable RGB/BGR image quality measurements and configurable defect thresholds."""
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class QualityThresholds:
    """Heuristic thresholds; tune on a representative validation set for deployment."""

    min_width: int = 256
    min_height: int = 256
    blur_laplacian: float = 80.0
    min_brightness: float = 95.0
    min_midtone_brightness: float = 132.0
    min_median_brightness: float = 150.0
    max_brightness: float = 250.0
    max_shadow_fraction: float = 0.38
    max_highlight_fraction: float = 0.95
    max_local_exposure_std: float = 58.0
    clipping_fraction: float = 0.04
    min_connected_clipping_fraction: float = 0.004
    min_contrast: float = 22.0
    max_contrast: float = 135.0
    max_noise: float = 6.5
    max_colorcast_spread: float = 30.0
    flat_gradient_percentile: float = 55.0
    weight_sharpness: float = 0.22
    weight_exposure: float = 0.22
    weight_contrast: float = 0.14
    weight_noise: float = 0.18
    weight_resolution: float = 0.14
    weight_color_balance: float = 0.10


def _gray(image: np.ndarray, color_order: str = "RGB") -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 2:
        return np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("Expected grayscale or RGB/BGR image")
    order = color_order.upper()
    if order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    code = cv2.COLOR_RGB2GRAY if order == "RGB" else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(np.clip(a[:, :, :3], 0, 255).astype(np.uint8), code)


def _mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values.astype(np.float32) - np.median(values)
    return float(1.4826 * np.median(np.abs(centered)))


def _estimate_flat_region_noise(gray: np.ndarray, percentile: float) -> dict[str, Any]:
    """Estimate sensor-like noise from low-gradient pixels at two spatial scales.

    Flat pixels are selected using a lightly smoothed image so grain itself has less
    influence on the selection. Two residual filters provide a stability check and
    reduce confusion between fine product texture and broadband noise.
    """
    f = gray.astype(np.float32)
    if min(f.shape) < 5:
        return {"sigma": 0.0, "confidence": 0.0, "flat_fraction": 0.0, "scale_estimates": {}}
    guide = cv2.GaussianBlur(f, (0, 0), 1.0)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    cutoff = float(np.percentile(grad, np.clip(percentile, 10, 90)))
    flat = grad <= cutoff
    # Use the full image if an unusually textured image has too few flat pixels.
    if int(flat.sum()) < max(64, int(flat.size * 0.08)):
        flat = np.ones_like(flat, dtype=bool)

    residual3 = f - cv2.medianBlur(gray, 3).astype(np.float32)
    residual5 = f - cv2.medianBlur(gray, 5).astype(np.float32)
    sigma3 = _mad_sigma(residual3[flat])
    sigma5 = _mad_sigma(residual5[flat])
    estimates = {"median3": sigma3, "median5": sigma5}
    sigma = float(np.median([sigma3, sigma5]))
    disagreement = abs(sigma3 - sigma5) / max(sigma3, sigma5, 1.0)
    flat_fraction = float(flat.mean())
    confidence = float(np.clip((0.55 + 0.35 * min(flat_fraction / 0.35, 1.0)) * (1.0 - 0.65 * disagreement), 0.0, 1.0))
    return {"sigma": sigma, "confidence": confidence, "flat_fraction": flat_fraction, "scale_estimates": estimates}


def analyze_quality(
    image: np.ndarray,
    thresholds: QualityThresholds | None = None,
    color_order: str = "RGB",
) -> dict[str, Any]:
    """Measure image quality and return normalized scores plus explainable flags.

    `color_order` controls grayscale conversion for 3-channel arrays. Brightness
    exposure evidence combines mean, median, shadow/highlight mass, and clipping.
    """
    t = thresholds or QualityThresholds()
    a = np.asarray(image)
    if a.size == 0 or a.ndim not in (2, 3):
        raise ValueError("Image is empty or has unsupported shape")
    h, w = a.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Image has zero width or height")
    g = _gray(a, color_order)
    f = g.astype(np.float32)
    lap = float(cv2.Laplacian(f, cv2.CV_32F).var()) if min(h, w) > 1 else 0.0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    brightness = float(f.mean())
    median_brightness = float(np.median(f))
    # Count near-saturation pixels, not merely bright pixels; ordinary grain near a
    # pale background should not by itself become an exposure defect.
    clipping = float(np.mean((f <= 2) | (f >= 253)))
    shadows = float(np.mean(f < 32))
    highlights = float(np.mean(f > 245))
    y_edges = np.linspace(0, h, 5, dtype=int)
    x_edges = np.linspace(0, w, 5, dtype=int)
    grid_means = [float(f[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]].mean())
                  for yi in range(4) for xi in range(4)
                  if y_edges[yi + 1] > y_edges[yi] and x_edges[xi + 1] > x_edges[xi]]
    local_exposure_std = float(np.std(grid_means)) if grid_means else 0.0
    border_gray = np.concatenate((f[0], f[-1], f[:, 0], f[:, -1]))
    border_mean = float(np.median(border_gray))
    saturated = ((f <= 2) | (f >= 253)).astype(np.uint8)
    _, sat_labels, sat_stats, _ = cv2.connectedComponentsWithStats(saturated, 8)
    largest_saturated_region = float(sat_stats[1:, cv2.CC_STAT_AREA].max() / (h * w)) if len(sat_stats) > 1 else 0.0
    coherent_clipping = largest_saturated_region >= t.min_connected_clipping_fraction or clipping >= 0.12
    contrast = float(f.std())
    noise = _estimate_flat_region_noise(g, t.flat_gradient_percentile)
    noise_sigma = noise["sigma"]

    flags: list[str] = []
    explanations: dict[str, str] = {}
    evidence: dict[str, float] = {}
    if lap < t.blur_laplacian:
        flags.append("blur")
        explanations["blur"] = f"Laplacian variance {lap:.1f} is below {t.blur_laplacian:.1f}."
        evidence["blur"] = float(np.clip(1 - lap / max(t.blur_laplacian, 1), 0, 1))

    midtone_under = brightness < t.min_midtone_brightness and median_brightness < t.min_median_brightness and border_mean > 50
    under = (brightness < t.min_brightness or midtone_under or shadows > t.max_shadow_fraction)
    over = brightness > t.max_brightness or median_brightness > 252 or highlights > t.max_highlight_fraction
    uneven = local_exposure_std > t.max_local_exposure_std
    clipped = clipping > t.clipping_fraction and coherent_clipping
    if under or over or clipped or uneven:
        flags.append("exposure")
        reasons = []
        if under:
            reasons.append("shadow/low-intensity distribution")
        if over:
            reasons.append("highlight/high-intensity distribution")
        if clipped:
            reasons.append("clipped pixels")
        if uneven:
            reasons.append("uneven local illumination")
        explanations["exposure"] = "Exposure concern from " + ", ".join(reasons) + "."
        evidence["exposure"] = float(np.clip(max(
            max(0.0, t.min_brightness - brightness) / t.min_brightness,
            max(0.0, t.min_midtone_brightness - brightness) / t.min_midtone_brightness if midtone_under else 0.0,
            max(0.0, t.min_median_brightness - median_brightness) / t.min_median_brightness if midtone_under else 0.0,
            max(0.0, brightness - t.max_brightness) / max(255 - t.max_brightness, 1),
            max(0.0, shadows - t.max_shadow_fraction) / max(1.0 - t.max_shadow_fraction, 1e-6),
            max(0.0, highlights - t.max_highlight_fraction) / max(1.0 - t.max_highlight_fraction, 1e-6),
            (clipping / max(t.clipping_fraction, 1e-6)) if coherent_clipping else 0.0,
            max(0.0, local_exposure_std - t.max_local_exposure_std) / max(t.max_local_exposure_std, 1e-6),
        ), 0, 1))

    if contrast < t.min_contrast:
        flags.append("low_contrast")
        explanations["low_contrast"] = f"RMS contrast {contrast:.1f} is below {t.min_contrast:.1f}."
        evidence["low_contrast"] = float(np.clip((t.min_contrast - contrast) / t.min_contrast, 0, 1))
    elif contrast > t.max_contrast:
        flags.append("excessive_contrast")
        explanations["excessive_contrast"] = f"RMS contrast {contrast:.1f} exceeds the configured {t.max_contrast:.1f} ceiling."
        evidence["excessive_contrast"] = float(np.clip((contrast - t.max_contrast) / t.max_contrast, 0, 1))

    if noise_sigma > t.max_noise:
        flags.append("noise")
        explanations["noise"] = f"Flat-region noise estimate {noise_sigma:.2f} exceeds {t.max_noise:.2f} (confidence {noise['confidence']:.2f})."
        evidence["noise"] = float(np.clip((noise_sigma - t.max_noise) / max(t.max_noise, 1), 0, 1))

    border_texture = 0.0
    if a.ndim == 3 and a.shape[2] >= 3:
        # Border channel cast is evidence only; it is intentionally not an automatic correction trigger alone.
        rgb = a[:, :, :3].astype(np.float32)
        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
        border_means = np.mean(border, axis=0)
        channel_spread = float(np.std(border_means))
        border_texture = float(np.mean(np.std(border, axis=0)))
        neutral_background = float(np.min(border_means)) > 100 and border_texture < 18
        if neutral_background and channel_spread > 3.5:
            flags.append("colorcast")
            explanations["colorcast"] = f"Border channel means vary by {channel_spread:.1f} intensity units."
            evidence["colorcast"] = float(np.clip(channel_spread / 30, 0, 1))
    else:
        channel_spread = 0.0

    if w < t.min_width or h < t.min_height:
        flags.append("low_resolution")
        explanations["low_resolution"] = f"Image is {w}x{h}; configured minimum is {t.min_width}x{t.min_height}."
        evidence["low_resolution"] = float(np.clip(1 - min(w / t.min_width, h / t.min_height), 0, 1))

    # Scores saturate in useful ranges; extreme contrast and sharpening cannot keep earning points.
    sharp_score = float(np.clip(lap / max(t.blur_laplacian * 2.0, 1), 0, 1))
    if lap > t.blur_laplacian * 20:
        sharp_score = min(sharp_score, 0.9)
    exposure_distance = max(
        max(0.0, t.min_brightness - brightness) / t.min_brightness,
        max(0.0, t.min_midtone_brightness - brightness) / t.min_midtone_brightness if midtone_under else 0.0,
        max(0.0, t.min_median_brightness - median_brightness) / t.min_median_brightness if midtone_under else 0.0,
        max(0.0, brightness - t.max_brightness) / max(255 - t.max_brightness, 1),
        max(0.0, shadows - t.max_shadow_fraction) / max(1.0 - t.max_shadow_fraction, 1e-6),
        max(0.0, highlights - t.max_highlight_fraction) / max(1.0 - t.max_highlight_fraction, 1e-6),
        (clipping / max(t.clipping_fraction, 1e-6)) if coherent_clipping else 0.0,
        max(0.0, local_exposure_std - t.max_local_exposure_std) / max(t.max_local_exposure_std, 1e-6),
    )
    exposure_score = float(np.clip(1.0 - exposure_distance, 0, 1))
    if contrast < t.min_contrast:
        contrast_score = float(np.clip(contrast / t.min_contrast, 0, 1))
    elif contrast <= t.max_contrast:
        contrast_score = 1.0
    else:
        contrast_score = float(np.clip(1 - (contrast - t.max_contrast) / t.max_contrast, 0, 1))
    noise_score = float(np.clip(1 - noise_sigma / max(t.max_noise * 2, 1), 0, 1))
    resolution_score = float(np.clip(min(w / t.min_width, h / t.min_height), 0, 1))
    scores = {
        "sharpness": sharp_score,
        "exposure": exposure_score,
        "contrast": contrast_score,
        "noise": noise_score,
        "resolution": resolution_score,
    }
    weights = {
        "sharpness": t.weight_sharpness,
        "exposure": t.weight_exposure,
        "contrast": t.weight_contrast,
        "noise": t.weight_noise,
        "resolution": t.weight_resolution,
    }
    effective_weights = dict(weights)
    # A dim/bright distribution also suppresses contrast; reduce that correlated term.
    if "exposure" in flags:
        effective_weights["contrast"] *= 0.5
    color_balance_score = 1.0
    if "colorcast" in flags:
        color_balance_score = float(np.clip(
            1.0 - max(0.0, channel_spread - 3.5) / max(t.max_colorcast_spread - 3.5, 1.0), 0, 1
        ))
    weight_sum = sum(max(0.0, v) for v in effective_weights.values()) or 1.0
    scores["color_balance"] = color_balance_score
    effective_weights["color_balance"] = t.weight_color_balance
    weight_sum = sum(max(0.0, v) for v in effective_weights.values()) or 1.0
    overall = 100.0 * sum(scores[k] * max(effective_weights[k], 0.0) for k in scores) / weight_sum
    return {
        "width": int(w), "height": int(h),
        "laplacian_variance": lap, "tenengrad": tenengrad,
        "mean_brightness": brightness, "median_brightness": median_brightness,
        "clipping_fraction": clipping, "coherent_clipping_fraction": largest_saturated_region,
        "border_median_brightness": border_mean, "shadow_fraction": shadows,
        "highlight_fraction": highlights, "local_exposure_std": local_exposure_std, "rms_contrast": contrast,
        "noise_sigma": noise_sigma, "noise_confidence": noise["confidence"],
        "noise_flat_fraction": noise["flat_fraction"],
        "noise_scale_estimates": noise["scale_estimates"],
        "border_channel_spread": channel_spread, "border_texture_std": border_texture,
        "flags": flags, "explanations": explanations, "defect_evidence": evidence,
        "scores": scores, "score_weights": effective_weights, "quality_score": float(overall),
    }
