"""VisionGuard Streamlit dashboard for local product image inspection."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from agent.planner import auto_fix, inspect
from dip.enhancement import clahe_lab
from dip.quality import QualityThresholds


st.set_page_config(page_title="VisionGuard", page_icon="🛡️", layout="wide")
st.title("VisionGuard")
st.caption("Automated e-commerce image quality and illustrative compliance inspector · HSR-AIML-056")
st.info("Local CPU classical-DIP pipeline. Marketplace-style checks are illustrative, not official certification.")
files = st.file_uploader(
    "Upload one or more product images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)


def _without_mask(segmentation: dict) -> dict:
    return {key: value for key, value in segmentation.items() if key != "mask"}


def _quality_rows(report: dict) -> list[dict]:
    scores = report["scores"]
    names = {"sharpness": "Sharpness", "exposure": "Exposure", "contrast": "Contrast", "noise": "Noise", "resolution": "Resolution", "color_balance": "Color balance"}
    return [{"Metric": names[key], "Score (0–1)": round(value, 3)} for key, value in scores.items()]


for uploaded in files or []:
    try:
        raw = uploaded.getvalue()
        image = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        cache_key = sha256(raw).hexdigest()
        before = inspect(image)
        session_key = f"visionguard_fix_{cache_key}"
        if st.button("Auto-Fix image", key=f"fix_{cache_key}"):
            fixed, trace = auto_fix(image)
            st.session_state[session_key] = {"image": fixed, "trace": trace}
        saved = st.session_state.get(session_key)
        corrected = saved["image"] if saved else image
        trace = saved["trace"] if saved else {"before": before, "after": before, "operations": [], "recommendations": [], "status": "not_run"}
        after = trace["after"]

        st.subheader(uploaded.name)
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Original", width="stretch")
            st.metric("Original quality", f"{before['quality']['quality_score']:.1f}/100")
        with right:
            st.image(corrected, caption="Corrected image" if saved else "Unchanged image", width="stretch")
            st.metric("Final quality", f"{after['quality']['quality_score']:.1f}/100", delta=f"{after['quality']['quality_score'] - before['quality']['quality_score']:+.2f}")
            st.caption(f"Correction status: **{trace['status']}**" if saved else "Correction status: Auto-Fix has not been run.")
            png_buffer = BytesIO()
            Image.fromarray(corrected).save(png_buffer, format="PNG")
            st.download_button("Download corrected image", png_buffer.getvalue(), f"{uploaded.name.rsplit('.', 1)[0]}_fixed.png", "image/png", key=f"png_{cache_key}")

        st.markdown("### Quality analysis — technical image quality")
        qleft, qright = st.columns([1, 1])
        with qleft:
            st.dataframe(_quality_rows(before["quality"]), hide_index=True, width="stretch")
            st.write(f"Detected defects: {', '.join(before['quality']['flags']) or 'none'}")
            if before["quality"]["explanations"]:
                st.json(before["quality"]["explanations"])
        with qright:
            st.write("**Original measurements**")
            st.json({key: before["quality"][key] for key in (
                "laplacian_variance", "tenengrad", "mean_brightness", "median_brightness",
                "clipping_fraction", "coherent_clipping_fraction", "border_median_brightness",
                "shadow_fraction", "highlight_fraction", "rms_contrast",
                "local_exposure_std", "noise_sigma", "noise_confidence", "noise_scale_estimates",
                "border_channel_spread", "border_texture_std", "width", "height", "quality_score",
            )})
            st.write("**Final measurements**")
            st.json({"quality_score": after["quality"]["quality_score"], "flags": after["quality"]["flags"], "scores": after["quality"]["scores"]})

        st.markdown("### Segmentation")
        segmentation = before["segmentation"]
        seg_left, seg_right = st.columns(2)
        overlay = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(overlay)
        x, y, width, height = segmentation["bbox"]
        if not segmentation["fallback"]:
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(255, 40, 40), width=max(2, image.shape[0] // 200))
        with seg_left:
            st.image(overlay, caption=f"Detected bounding box · {segmentation['method']}", width="stretch")
            st.image(segmentation["mask"], caption="Product foreground mask", clamp=True, width="stretch")
        with seg_right:
            st.write({"bbox_xywh": segmentation["bbox"], "foreground_coverage": round(segmentation["coverage"], 4), "confidence": round(segmentation["confidence"], 3), "fallback": segmentation["fallback"]})
            for warning in segmentation["warnings"]:
                st.warning(warning)

        st.markdown("### Auto-Fix history")
        if trace["operations"]:
            history_rows = []
            for operation in trace["operations"]:
                history_rows.append({
                    "Operation": operation["operation"], "Decision": "Accepted" if operation["accepted"] else "Rejected",
                    "Quality before": round(operation["quality_before"], 3), "Quality after": round(operation["quality_after"], 3),
                    "Quality delta": round(operation["quality_delta"], 3), "Target measurements": operation["target_metrics"],
                    "Defects after": ", ".join(operation["flags_after"]) or "none", "Reason": operation["reason"],
                })
            st.dataframe(history_rows, hide_index=True, width="stretch")
        else:
            st.caption("No correction operation has been accepted or attempted.")
        for recommendation in trace["recommendations"]:
            st.info(recommendation)

        st.markdown("### Marketplace-style presentation compliance")
        compliance = before["compliance"]
        st.caption(compliance["disclaimer"])
        st.write("Technical quality verdict:", "Pass" if compliance["technical_quality"]["passed"] else "Review required")
        st.write("Presentation rules verdict:", "Pass" if compliance["presentation_compliance"]["passed"] else "One or more illustrative rules failed")
        st.dataframe(compliance["presentation_compliance"]["rules"], hide_index=True, width="stretch")

        if "low_contrast" in before["quality"]["flags"] and before["quality"]["noise_sigma"] < QualityThresholds().max_noise * 0.75:
            with st.expander("Optional CLAHE diagnostic preview"):
                st.caption("Diagnostic preview only; this is not the corrected output. CLAHE is shown only when the measured noise is low enough to limit grain amplification.")
                st.image(clahe_lab(image, clip_limit=1.2), width="stretch")

        json_report = {
            "filename": uploaded.name,
            "quality": before["quality"],
            "segmentation": _without_mask(before["segmentation"]),
            "compliance": before["compliance"],
            "operations": trace["operations"],
            "recommendations": trace["recommendations"],
            "status": trace["status"],
            "after": {"quality": after["quality"], "segmentation": _without_mask(after["segmentation"]), "compliance": after["compliance"]},
        }
        st.download_button(
            "Download JSON inspection report", json.dumps(json_report, indent=2),
            f"{uploaded.name.rsplit('.', 1)[0]}_report.json", "application/json", key=f"json_{cache_key}",
        )
    except Exception as exc:
        st.error(f"Could not inspect {uploaded.name}: {exc}")
