"""Streamlit upload/Auto-Fix smoke test and score consistency check."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest


def test_uploaded_image_autofix_score_and_downloads_are_consistent():
    image = Image.new("RGB", (320, 320), (75, 75, 75))
    draw = ImageDraw.Draw(image)
    draw.ellipse((105, 90, 215, 230), fill=(70, 100, 135), outline=(230, 220, 190), width=5)
    content = BytesIO()
    image.save(content, format="PNG")
    payload = content.getvalue()

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.file_uploader[0].set_value(("underexposed.png", payload, "image/png")).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Auto-Fix image").click().run()
    assert not app.exception

    state_key = f"visionguard_fix_{sha256(payload).hexdigest()}"
    report = app.session_state[state_key]["trace"]
    final_ui_score = float(app.metric[1].proto.body.split("/")[0])
    assert final_ui_score == round(report["after"]["quality"]["quality_score"], 1)
    download_labels = {button.label for button in app.get("download_button")}
    assert "Download corrected image" in download_labels
    assert "Download JSON inspection report" in download_labels
