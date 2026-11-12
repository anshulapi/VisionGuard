"""Confidence-aware product foreground segmentation using classical DIP."""
from typing import Any

import cv2
import numpy as np


def _fallback(h: int, w: int, warning: str) -> dict[str, Any]:
    return {
        "mask": np.zeros((h, w), np.uint8), "bbox": [0, 0, w, h],
        "centroid": [0.5, 0.5], "coverage": 0.0, "centering_score": 0.0,
        "background_std": 0.0, "background_mean": 0.0,
        "confidence": 0.0, "method": "fallback_empty_mask", "fallback": True,
        "warnings": [warning], "candidate_scores": {},
    }


def _component_mask(binary: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, dict[str, Any]] | None:
    # Close first to reconnect thin bands and small gaps; one gentle opening removes isolated specks.
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if min(mask.shape) >= 48:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    h, w = mask.shape
    image_area = h * w
    components = []
    for idx in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[idx])
        if area < max(4, int(image_area * 0.00015)):
            continue
        cx, cy = centers[idx]
        border_touch = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        coverage = area / image_area
        if coverage > 0.96:
            continue
        area_plausibility = float(np.clip(min(coverage / 0.01, 1.0, (0.92 - coverage) / 0.25), 0, 1))
        center_distance = float(np.hypot(cx / w - 0.5, cy / h - 0.5) / 0.7071)
        # Centering is a soft cue: off-centre products remain segmentable.
        center_plausibility = 0.82 + 0.18 * (1.0 - min(center_distance, 1.0))
        score = area_plausibility * center_plausibility * (0.62 if border_touch else 1.0)
        components.append((score, idx, x, y, bw, bh, area, border_touch))
    if not components:
        return None
    components.sort(reverse=True)
    best = components[0]
    _, idx, x, y, bw, bh, area, border_touch = best
    selected = [best]
    # Include substantial nearby disconnected pieces (e.g., a watch strap tip) without swallowing clutter.
    for item in components[1:]:
        _, other_idx, ox, oy, obw, obh, oarea, _ = item
        expanded = (x - max(5, bw // 8), y - max(5, bh // 8), x + bw + max(5, bw // 8), y + bh + max(5, bh // 8))
        ocx, ocy = ox + obw / 2, oy + obh / 2
        if oarea >= area * 0.025 and expanded[0] <= ocx <= expanded[2] and expanded[1] <= ocy <= expanded[3]:
            selected.append(item)
    out = np.zeros_like(mask)
    for item in selected:
        out[labels == item[1]] = 255
    total_foreground = sum(item[6] for item in components)
    artifact_fraction = float(max(0, total_foreground - sum(item[6] for item in selected)) / max(total_foreground, 1))
    ys, xs = np.where(out > 0)
    if not len(xs):
        return None
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    combined_coverage = float(len(xs) / image_area)
    combined_touches_border = x0 == 0 or y0 == 0 or x1 >= w or y1 >= h
    component_score = float(best[0])
    # The union of many connected rings often represents the background polarity.
    if combined_coverage > 0.82 and combined_touches_border:
        component_score *= 0.05
    elif combined_coverage > 0.72:
        component_score *= 0.25
    info = {
        "bbox": [x0, y0, x1 - x0, y1 - y0], "coverage": float(len(xs) / image_area),
        "border_touch": bool(combined_touches_border), "artifact_fraction": artifact_fraction,
        "centroid": [float(xs.mean() / w), float(ys.mean() / h)],
        "component_count": len(selected), "score": component_score,
    }
    return out, info


def segment_product(image: np.ndarray, refine: bool = True) -> dict[str, Any]:
    """Segment the likely product and report confidence instead of silently trusting Otsu.

    Border colour is estimated in LAB. Both Otsu polarities are evaluated; components
    touching the image edge, occupying implausible areas, or leaving substantial
    disconnected residue reduce confidence. Low-confidence masks are returned with
    explicit warnings and use a safe empty-mask fallback when no plausible region exists.
    """
    a = np.asarray(image)
    if a.size == 0:
        raise ValueError("Image is empty")
    if a.ndim == 2:
        rgb = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif a.ndim == 3 and a.shape[2] >= 3:
        rgb = np.clip(a[:, :, :3], 0, 255).astype(np.uint8)
    else:
        raise ValueError("Expected grayscale or RGB image")
    h, w = rgb.shape[:2]
    if min(h, w) < 3:
        return _fallback(h, w, "Image is too small for reliable segmentation.")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(distance * 4, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = min(7, max(3, (min(h, w) // 180) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    candidates: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    scores: dict[str, float] = {}
    for polarity, mode in (("foreground", cv2.THRESH_BINARY), ("background", cv2.THRESH_BINARY_INV)):
        _, raw = cv2.threshold(d8, otsu_threshold, 255, mode)
        result = _component_mask(raw, kernel)
        if result is None:
            scores[polarity] = 0.0
            continue
        mask, info = result
        scores[polarity] = info["score"] * (1 - 0.45 * info["artifact_fraction"])
        candidates[polarity] = (mask, info)
    if not candidates:
        return _fallback(h, w, "Otsu produced no plausible product component in either polarity.")

    method_key = max(scores, key=scores.get)
    mask, info = candidates[method_key]
    # Otsu separation gives a data-dependent cue about foreground/background ambiguity.
    fg_values = distance[mask > 0]
    bg_values = distance[mask == 0]
    separation = 0.0
    if fg_values.size and bg_values.size:
        delta = abs(float(np.median(fg_values) - np.median(bg_values)))
        spread = float(np.median(np.abs(fg_values - np.median(fg_values))) + np.median(np.abs(bg_values - np.median(bg_values))) + 1)
        separation = float(np.clip(delta / (delta + spread), 0, 1))
    confidence = float(np.clip(
        0.40 * scores[method_key]
        + 0.25 * separation
        + 0.20 * (0.45 if info["border_touch"] else 1.0)
        + 0.15 * (1.0 - info["artifact_fraction"]),
        0, 1,
    ))

    method = "otsu_morphology_" + method_key
    if refine and max(h, w) >= 32 and confidence >= 0.35:
        gc = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        # Definite seeds keep GrabCut from accepting an implausible all-foreground result.
        inset = max(1, min(h, w) // 40)
        gc[inset:-inset or None, inset:-inset or None][mask[inset:-inset or None, inset:-inset or None] > 0] = cv2.GC_FGD
        try:
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            rect = (0, 0, w, h)
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
            refined = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            if 0 < np.count_nonzero(refined) < h * w * 0.95:
                # Accept only if the refinement preserves a plausible amount of the seed mask.
                overlap = np.count_nonzero((refined > 0) & (mask > 0)) / max(1, np.count_nonzero(mask))
                if overlap >= 0.45:
                    mask = refined
                    method += "_grabcut"
        except cv2.error:
            pass

    ys, xs = np.where(mask > 0)
    if not len(xs):
        return _fallback(h, w, "Selected segmentation became empty; mask is unreliable.")
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    cx, cy = float(xs.mean() / w), float(ys.mean() / h)
    coverage = float(len(xs) / (h * w))
    center_score = float(np.clip(1 - np.hypot(cx - 0.5, cy - 0.5) / 0.7071, 0, 1))
    rgb_border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0).astype(np.float32)
    bg_std = float(np.mean(np.std(rgb_border, axis=0)))
    warnings: list[str] = []
    if confidence < 0.55:
        warnings.append("Segmentation confidence is low; review the mask before using layout measurements.")
    if info["border_touch"]:
        warnings.append("The selected foreground component touches the image border; clipping may affect its completeness.")
    if coverage < 0.02:
        warnings.append("Detected foreground coverage is very small; mask may contain only part of the product.")
    if info["artifact_fraction"] > 0.5:
        warnings.append("Many disconnected foreground pixels were discarded as likely background artifacts.")
    if confidence < 0.15:
        return _fallback(h, w, "Segmentation confidence is too low to provide a trustworthy mask.")
    return {
        "mask": mask, "bbox": [x0, y0, x1 - x0, y1 - y0],
        "centroid": [cx, cy], "coverage": coverage, "centering_score": center_score,
        "background_std": bg_std, "background_mean": float(rgb_border.mean()),
        "confidence": confidence, "method": method, "fallback": False,
        "warnings": warnings, "candidate_scores": scores,
        "component_count": info["component_count"], "artifact_fraction": info["artifact_fraction"],
    }
