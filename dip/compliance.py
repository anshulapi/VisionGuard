"""Separate technical quality diagnosis from configurable illustrative presentation checks."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from .segmentation import segment_product


@dataclass
class ComplianceRules:
    min_width: int = 500
    min_height: int = 500
    min_coverage: float = 0.08
    max_coverage: float = 0.90
    min_centering: float = 0.70
    max_background_std: float = 22.0
    border_margin: float = 0.005
    require_white_background: bool = True
    min_background_mean: float = 220.0
    min_segmentation_confidence: float = 0.45


def inspect_compliance(
    image: np.ndarray,
    quality: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    rules: ComplianceRules | None = None,
) -> dict[str, Any]:
    """Return technical findings separately from nonofficial marketplace-style rules."""
    r = rules or ComplianceRules()
    h, w = image.shape[:2]
    q = quality or {}
    s = segmentation or segment_product(image)
    x, y, bw, bh = s["bbox"]
    clipped = not s["fallback"] and (
        x <= w * r.border_margin or y <= h * r.border_margin
        or x + bw >= w * (1 - r.border_margin)
        or y + bh >= h * (1 - r.border_margin)
    )
    presentation = [
        ("resolution", w >= r.min_width and h >= r.min_height,
         f"Image is {w}x{h}; illustrative minimum is {r.min_width}x{r.min_height}."),
        ("centering", s["centering_score"] >= r.min_centering,
         f"Centering score {s['centering_score']:.3f}; configured minimum {r.min_centering:.2f}."),
        ("foreground_coverage", r.min_coverage <= s["coverage"] <= r.max_coverage,
         f"Foreground coverage {s['coverage']:.3f}; configured range {r.min_coverage:.2f}-{r.max_coverage:.2f}."),
        ("background_uniformity", s["background_std"] <= r.max_background_std,
         f"Border colour standard deviation {s['background_std']:.1f}; configured maximum {r.max_background_std:.1f}."),
        ("background_whiteness", (not r.require_white_background) or s.get("background_mean", 0) >= r.min_background_mean,
         (f"Border mean RGB intensity {s.get('background_mean', 0):.1f}; configured minimum {r.min_background_mean:.1f}."
          if r.require_white_background else "White background requirement is disabled.")),
        ("border_clipping", not clipped,
         "Detected foreground does not touch the image border." if not clipped else "Detected foreground touches or crosses the image border."),
        ("segmentation_confidence", s.get("confidence", 0) >= r.min_segmentation_confidence,
         f"Segmentation confidence {s.get('confidence', 0):.2f}; configured minimum {r.min_segmentation_confidence:.2f}."),
    ]
    technical = [{"name": "image_quality", "passed": not q.get("flags"),
                  "explanation": "No configured technical quality defects detected." if not q.get("flags")
                  else "Detected technical defects: " + ", ".join(q["flags"]) + "."}]
    presentation_rules = [
        {"name": name, "passed": bool(passed), "explanation": explanation,
         "category": "marketplace_presentation"}
        for name, passed, explanation in presentation
    ]
    technical_passed = all(row["passed"] for row in technical)
    presentation_passed = all(row["passed"] for row in presentation_rules)
    return {
        # Backwards compatible aliases: `passed` and `rules` refer to presentation checks.
        "passed": presentation_passed,
        "rules": technical + presentation_rules,
        "technical_quality": {"passed": technical_passed, "rules": technical},
        "presentation_compliance": {"passed": presentation_passed, "rules": presentation_rules},
        "disclaimer": "Illustrative marketplace-style checks only; not an official marketplace certification.",
    }
