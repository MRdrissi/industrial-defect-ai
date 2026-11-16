"""Interactions de la page d'analyse, sans charger les modèles."""

from io import BytesIO

import pytest
from PIL import Image

from app.image_utils import MAX_FILE_SIZE, decode_image, image_to_tensor


def make_png():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_conversion():
    image = decode_image(make_png())
    tensor = image_to_tensor(image)
    assert image.mode == "RGB"
    assert tensor.shape == (3, 8, 8)
    assert tensor.min() >= 0
    assert tensor.max() <= 1


def test_selectors(inference_app):
    at, _ = inference_app
    assert [box.label for box in at.selectbox] == ["Catégorie", "Méthode"]
    at.selectbox(key="category_selector").set_value("transistor").run()
    assert at.selectbox(key="category_selector").value == "transistor"
    assert at.selectbox(key="method_selector").value in ("cae", "patchcore")
    assert "Convolutional AutoEncoder" in at.selectbox(key="method_selector").options


def test_missing_image(inference_app):
    at, calls = inference_app
    assert at.button(key="analyze_image").disabled
    assert not at.metric
    assert not calls


@pytest.mark.parametrize("method, badge", [("cae", "NORMAL"), ("patchcore", "SUSPECT")])
def test_result(inference_app, method, badge):
    at, calls = inference_app
    at.selectbox(key="method_selector").set_value(method).run()
    at.file_uploader[0].upload("piece.png", make_png(), "image/png").run()
    at.button(key="analyze_image").click().run()
    assert not at.exception
    assert len(at.metric) == 4
    assert badge in at.metric[3].value
    assert at.button(key="analyze_image").disabled
    assert len(calls) == 1


def test_upload_constraints(inference_app):
    at, calls = inference_app
    uploader = at.file_uploader[0].proto
    assert uploader.max_upload_size_mb == 10
    assert not uploader.multiple_files
    assert set(uploader.type) == {".png", ".jpg", ".jpeg"}
    with pytest.raises(ValueError):
        decode_image(b"x" * (MAX_FILE_SIZE + 1))
    at.file_uploader[0].upload("invalid.png", b"not an image", "image/png").run()
    assert not at.exception
    assert at.error
    assert at.button(key="analyze_image").disabled
    assert not calls
