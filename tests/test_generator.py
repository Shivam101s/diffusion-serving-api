import base64
import io

from PIL import Image

from app.generator import MockGenerator


def test_mock_generator_returns_requested_count():
    images = MockGenerator().generate(num_samples=3)
    assert len(images) == 3


def test_mock_generator_output_is_valid_png():
    images = MockGenerator().generate(num_samples=1)
    raw = base64.b64decode(images[0])
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"
    assert img.size == (32, 32)


def test_mock_generator_is_deterministic_given_seed():
    a = MockGenerator().generate(num_samples=1, seed=42)
    b = MockGenerator().generate(num_samples=1, seed=42)
    assert a == b


def test_mock_generator_varies_without_seed():
    a = MockGenerator().generate(num_samples=1)
    b = MockGenerator().generate(num_samples=1)
    assert a != b
