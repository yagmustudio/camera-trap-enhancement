import cv2
import numpy as np
import pytest

import demo
from src.enhancement import equalize_histogram, load_grayscale


@pytest.fixture
def dark_image() -> np.ndarray:
    """Underexposed night-like image: all pixels between 0 and 40."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 41, size=(120, 160), dtype=np.uint8)


def test_output_has_same_shape_and_dtype(dark_image):
    result = equalize_histogram(dark_image)
    assert result.shape == dark_image.shape
    assert result.dtype == np.uint8


def test_dark_image_is_spread_over_full_range(dark_image):
    result = equalize_histogram(dark_image)
    assert result.max() == 255
    assert result.mean() > dark_image.mean() + 80
    assert int(result.max()) - int(result.min()) > 200


def test_input_is_not_modified(dark_image):
    original = dark_image.copy()
    equalize_histogram(dark_image)
    np.testing.assert_array_equal(dark_image, original)


@pytest.mark.parametrize("value", [0, 17, 255])
def test_constant_image_does_not_crash(value):
    flat = np.full((32, 32), value, dtype=np.uint8)
    result = equalize_histogram(flat)
    assert result.shape == flat.shape
    assert result.dtype == np.uint8
    assert np.unique(result).size == 1


def test_color_image_is_converted_to_grayscale(dark_image):
    bgr = cv2.cvtColor(dark_image, cv2.COLOR_GRAY2BGR)
    result = equalize_histogram(bgr)
    assert result.shape == dark_image.shape
    np.testing.assert_array_equal(result, equalize_histogram(dark_image))


def test_non_uint8_image_is_rejected():
    with pytest.raises(ValueError, match="uint8"):
        equalize_histogram(np.zeros((8, 8), dtype=np.float32))


def test_empty_image_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        equalize_histogram(np.zeros((0, 0), dtype=np.uint8))


def test_load_grayscale_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_grayscale(tmp_path / "does_not_exist.jpg")


def test_load_grayscale_undecodable_file_raises(tmp_path):
    fake = tmp_path / "not_an_image.jpg"
    fake.write_text("this is not a JPEG")
    with pytest.raises(ValueError, match="Could not decode"):
        load_grayscale(fake)


def test_load_grayscale_converts_color_file(tmp_path, dark_image):
    path = tmp_path / "color.png"
    cv2.imwrite(str(path), cv2.cvtColor(dark_image, cv2.COLOR_GRAY2BGR))
    loaded = load_grayscale(path)
    assert loaded.ndim == 2
    np.testing.assert_array_equal(loaded, dark_image)


def test_demo_writes_comparison_figure(tmp_path, dark_image):
    image_path = tmp_path / "night.png"
    output_path = tmp_path / "out" / "comparison.png"
    cv2.imwrite(str(image_path), dark_image)

    assert demo.main(["--input", str(image_path), "--output", str(output_path)]) == 0
    figure = cv2.imread(str(output_path))
    assert figure is not None and figure.size > 0


def test_demo_reports_missing_input(tmp_path, capsys):
    assert demo.main(["--input", str(tmp_path / "missing.jpg"), "--output", str(tmp_path / "x.png")]) == 1
    assert "not found" in capsys.readouterr().err
