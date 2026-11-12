"""Deterministic regression cases for quality, segmentation, compliance and auto-fix."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from agent.planner import PlannerConfig, auto_fix, inspect
from dip.compliance import ComplianceRules, inspect_compliance
from dip.quality import QualityThresholds, analyze_quality
from dip.segmentation import segment_product
from data.generate_dataset import generate_dataset
import pandas as pd


SEED = 1729


def watch_image(size=320, background=(238, 238, 238)):
    im = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(im)
    c = size // 2
    draw.rounded_rectangle((c-19, c-125, c+19, c-48), radius=12, fill=(35, 38, 42))
    draw.rounded_rectangle((c-19, c+48, c+19, c+125), radius=12, fill=(35, 38, 42))
    draw.ellipse((c-52, c-52, c+52, c+52), fill=(34, 40, 48), outline=(180, 164, 120), width=7)
    draw.ellipse((c-37, c-37, c+37, c+37), fill=(225, 222, 210), outline=(20, 20, 20), width=2)
    draw.line((c, c, c+20, c-18), fill=(30, 30, 30), width=3)
    draw.line((c, c, c-18, c+8), fill=(30, 30, 30), width=3)
    return np.asarray(im)


def noisy(image, sigma=12, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def product_on_dark_background(size=320):
    image = np.zeros((size, size, 3), np.uint8)
    cv2.rectangle(image, (135, 65), (185, 255), (225, 220, 190), -1)
    cv2.circle(image, (160, 160), 55, (230, 230, 225), -1)
    cv2.circle(image, (160, 160), 42, (100, 130, 170), -1)
    return image


def test_clean_product_not_flagged_as_noise_and_stays_unchanged():
    image = watch_image()
    report = analyze_quality(image)
    assert "noise" not in report["flags"]
    fixed, trace = auto_fix(image)
    assert np.array_equal(fixed, image)
    assert trace["status"] == "unchanged"
    assert trace["operations"] == []


def test_grayscale_tiny_and_rgba_inputs_are_handled():
    gray = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    fixed_gray, _ = auto_fix(gray)
    assert fixed_gray.shape == gray.shape
    tiny = np.array([[60, 70], [80, 90]], dtype=np.uint8)
    assert auto_fix(tiny)[0].shape == tiny.shape
    rgba = np.dstack((watch_image(), np.full((320, 320), 173, np.uint8)))
    fixed_rgba, _ = auto_fix(rgba)
    assert fixed_rgba.shape == rgba.shape
    assert np.array_equal(fixed_rgba[:, :, 3], rgba[:, :, 3])


def test_gaussian_noise_is_detected_independently_of_exposure():
    clean = watch_image()
    noisy_image = noisy(clean)
    result = analyze_quality(noisy_image)
    assert "noise" in result["flags"]
    assert "exposure" not in result["flags"]
    assert result["noise_sigma"] > QualityThresholds().max_noise
    assert 0 <= result["noise_confidence"] <= 1
    dark_noisy = noisy(np.clip(clean.astype(np.float32) * 0.38, 0, 255).astype(np.uint8), sigma=12)
    dark_result = analyze_quality(dark_noisy)
    assert "noise" in dark_result["flags"]
    assert "exposure" in dark_result["flags"]
    denoised, trace = auto_fix(noisy_image)
    assert trace["operations"][0]["operation"] == "bilateral_denoise"
    assert trace["operations"][0]["accepted"]
    assert analyze_quality(denoised)["noise_sigma"] < result["noise_sigma"]


def test_exposure_detection_and_gamma_preserve_working_underexposure_fix():
    clean = watch_image()
    under = np.clip(clean.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
    over = np.clip(clean.astype(np.float32) * 1.45, 0, 255).astype(np.uint8)
    assert "exposure" in analyze_quality(under)["flags"]
    assert "exposure" in analyze_quality(over)["flags"]
    fixed, report = auto_fix(under)
    assert report["operations"]
    assert report["after"]["quality"]["mean_brightness"] > report["before"]["quality"]["mean_brightness"]


def test_gaussian_and_motion_blur_are_detected():
    clean = watch_image()
    gaussian = cv2.GaussianBlur(clean, (11, 11), 3)
    motion_kernel = np.zeros((11, 11), np.float32)
    motion_kernel[5, :] = 1 / 11
    motion = cv2.filter2D(clean, -1, motion_kernel)
    assert "blur" in analyze_quality(gaussian)["flags"]
    assert "blur" in analyze_quality(motion)["flags"]


def test_low_contrast_and_complex_background_segmentation_warning():
    low = np.full((320, 320, 3), 125, np.uint8)
    cv2.rectangle(low, (115, 75), (205, 245), (134, 133, 130), -1)
    assert "low_contrast" in analyze_quality(low)["flags"]
    rng = np.random.default_rng(SEED)
    complex_bg = np.clip(watch_image().astype(np.int16) + rng.integers(-16, 17, (320, 320, 1)), 0, 255).astype(np.uint8)
    result = segment_product(complex_bg, refine=False)
    assert "confidence" in result and "warnings" in result
    assert result["mask"].shape == complex_bg.shape[:2]
    ambiguous = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
    uncertain = segment_product(ambiguous, refine=False)
    assert uncertain["confidence"] < 0.55
    assert uncertain["warnings"]


def test_small_product_and_dark_background_are_handled():
    small = np.full((320, 320, 3), 245, np.uint8)
    cv2.rectangle(small, (150, 150), (170, 170), (30, 50, 70), -1)
    small_seg = segment_product(small, refine=False)
    assert small_seg["coverage"] < 0.08
    small_compliance = inspect_compliance(small, segmentation=small_seg)
    coverage_rule = next(rule for rule in small_compliance["presentation_compliance"]["rules"] if rule["name"] == "foreground_coverage")
    assert not coverage_rule["passed"]

    dark_seg = segment_product(product_on_dark_background(), refine=False)
    assert dark_seg["coverage"] > 0.02
    assert not dark_seg["fallback"]


def test_auto_fix_rejects_ineffective_operation_without_mutating_input():
    under = np.clip(watch_image().astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
    original = under.copy()
    fixed, trace = auto_fix(under, config=PlannerConfig(min_quality_improvement=100.0))
    assert np.array_equal(under, original)
    assert trace["operations"]
    assert not any(row["accepted"] for row in trace["operations"])
    assert np.array_equal(fixed, original)


def test_compliance_categories_and_thresholds_are_configurable():
    image = watch_image()
    result = inspect(image)
    assert "technical_quality" in result["compliance"]
    assert "presentation_compliance" in result["compliance"]
    relaxed = inspect_compliance(image, segmentation=result["segmentation"], rules=ComplianceRules(
        min_width=1, min_height=1, min_coverage=0.0, max_coverage=1.0,
        min_centering=0.0, max_background_std=255, require_white_background=False,
        min_segmentation_confidence=0.0,
    ))
    assert relaxed["presentation_compliance"]["passed"]
    assert "not an official" in relaxed["disclaimer"]


def test_generated_groups_stay_in_one_split(tmp_path):
    manifest = generate_dataset(out_dir=tmp_path / "synthetic", groups=60, seed=SEED)
    frame = pd.read_csv(manifest)
    assert len(frame) == 60
    assert frame.groupby("group_id")["split"].nunique().max() == 1
    assert set(frame["split"]) == {"train", "validation", "test"}


def test_regeneration_cleans_only_owned_dataset_and_is_deterministic(tmp_path):
    out = tmp_path / "isolated"
    first = generate_dataset(out_dir=out, groups=60, seed=2026)
    assert len(list(out.rglob("input.png"))) == 60
    second = generate_dataset(out_dir=out, groups=24, seed=2026)
    frame = pd.read_csv(second)
    assert len(frame) == 24
    assert len(list(out.rglob("*.png"))) == 48
    assert frame.groupby("group_id")["split"].nunique().max() == 1
    manifest_bytes = second.read_bytes()
    sample_path = out / frame.iloc[0].image_path.removeprefix(str(out) + "/")
    sample_bytes = sample_path.read_bytes()
    again = generate_dataset(out_dir=out, groups=24, seed=2026)
    again_frame = pd.read_csv(again)
    again_path = out / again_frame.iloc[0].image_path.removeprefix(str(out) + "/")
    assert again.read_bytes() == manifest_bytes
    assert again_path.read_bytes() == sample_bytes


def test_exposure_score_tracks_midtone_flag_and_gamma_guard():
    clean = watch_image()
    midtone_dark = np.clip(clean.astype(np.float32) * 0.48, 0, 255).astype(np.uint8)
    before = analyze_quality(midtone_dark)
    assert "exposure" in before["flags"]
    assert before["scores"]["exposure"] < 1.0
    fixed, trace = auto_fix(midtone_dark)
    gamma = next(row for row in trace["operations"] if row["operation"] == "gamma_correct")
    assert gamma["checks"]["target_improved"]
    assert gamma["accepted"]
    assert trace["after"]["quality"]["scores"]["exposure"] > before["scores"]["exposure"]
    assert trace["status"] in {"corrected", "partially_improved"}


def test_color_cast_score_supports_measured_white_balance_and_partial_status():
    image = watch_image().astype(np.float32)
    image[:, :, 0] = np.clip(image[:, :, 0] * 1.20, 0, 255)
    cast = image.astype(np.uint8)
    before = analyze_quality(cast)
    assert "colorcast" in before["flags"]
    assert before["scores"]["color_balance"] < 1.0
    fixed, trace = auto_fix(cast)
    balance = next(row for row in trace["operations"] if row["operation"] == "white_balance")
    assert balance["checks"]["target_improved"]
    assert balance["accepted"]
    assert balance["quality_delta"] > 0
    assert trace["after"]["quality"]["border_channel_spread"] < before["border_channel_spread"]
    if "colorcast" in trace["after"]["quality"]["flags"]:
        assert trace["status"] == "partially_improved"
        assert any("defects remain" in note for note in trace["recommendations"])


def test_unowned_nonempty_dataset_output_is_never_overwritten(tmp_path):
    out = tmp_path / "not_visionguard"
    out.mkdir()
    (out / "keep.txt").write_text("user data")
    import pytest
    with pytest.raises(FileExistsError):
        generate_dataset(out_dir=out, groups=12, seed=2026)
    assert (out / "keep.txt").read_text() == "user data"
