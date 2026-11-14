"""Tests de l'interface Streamlit — US-06-02."""

from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from app.image_utils import decode_image, image_to_tensor


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app/streamlit_app.py"


def make_png(color=(128, 128, 128)):
    """Crée une petite image de test en mémoire."""
    image = Image.new("RGB", (8, 8), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_conversion():
    image = decode_image(make_png())
    tensor = image_to_tensor(image)

    assert image.mode == "RGB"
    assert tensor.shape == (3, 8, 8)
    assert tensor.min() >= 0
    assert tensor.max() <= 1


def test_selectors():
    at = AppTest.from_file(str(APP)).run()

    assert not at.exception
    assert len(at.selectbox) == 3

    at.selectbox[0].set_value("transistor").run()

    assert at.selectbox[0].value == "transistor"
    assert at.selectbox[1].value in ("cae", "patchcore")
    assert at.selectbox[2].value == "demo-v1"


def test_missing_image():
    at = AppTest.from_file(str(APP)).run()

    at.button[0].click().run()

    assert not at.exception
    assert len(at.warning) >= 1


def test_normal_result():
    at = AppTest.from_file(str(APP)).run()

    at.file_uploader[0].upload(
        "normal.png",
        make_png(),
        "image/png",
    ).run()

    at.button[0].click().run()

    assert not at.exception
    assert len(at.success) == 1
    assert len(at.metric) == 3


def test_suspect_result():
    at = AppTest.from_file(str(APP)).run()

    at.file_uploader[0].upload(
        "black.png",
        make_png((0, 0, 0)),
        "image/png",
    ).run()

    at.button[0].click().run()

    assert not at.exception
    assert len(at.warning) >= 1
    assert len(at.metric) == 3


def test_invalid_image():
    at = AppTest.from_file(str(APP)).run()

    at.file_uploader[0].upload(
        "invalid.png",
        b"not a real image",
        "image/png",
    ).run()

    assert not at.exception
    assert len(at.error) >= 1
