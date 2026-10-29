import cv2
import numpy as np
import pytest
from PIL import Image

from services import pest_service
from utils.preprocessing import build_pest_input, load_leaf_image, preprocess_pest_image

from conftest import pest_fixture

EXPECTED = {
    "apple_scab": ("Apple", pest_service.STATUS_DISEASE),
    "apple_black_rot": ("Apple", pest_service.STATUS_DISEASE),
    "cherry_powdery_mildew": ("Cherry", pest_service.STATUS_DISEASE),
    "corn_common_rust": ("Corn (maize)", pest_service.STATUS_DISEASE),
    "grape_black_rot": ("Grape", pest_service.STATUS_DISEASE),
    "potato_late_blight": ("Potato", pest_service.STATUS_DISEASE),
    "strawberry_leaf_scorch": ("Strawberry", pest_service.STATUS_DISEASE),
    "tomato_early_blight": ("Tomato", pest_service.STATUS_DISEASE),
    "pepper_bacterial_spot": ("Bell pepper", pest_service.STATUS_DISEASE),
    "apple_healthy": ("Apple", pest_service.STATUS_HEALTHY),
    "corn_healthy": ("Corn (maize)", pest_service.STATUS_HEALTHY),
    "tomato_healthy": ("Tomato", pest_service.STATUS_HEALTHY),
}


@pytest.mark.parametrize("fixture_name", sorted(EXPECTED))
def test_supported_crops_are_identified_correctly(fixture_name):
    expected_crop, expected_status = EXPECTED[fixture_name]

    success, report, error = pest_service.predict_pest(pest_fixture(fixture_name))

    assert success, error
    assert report["status"] == expected_status
    assert report["crop"] == expected_crop
    assert report["confidence"] >= 0.6


def test_multiple_crops_do_not_collapse_to_one_label():
    crops = set()
    for fixture_name in ["apple_scab", "corn_common_rust", "grape_black_rot", "tomato_early_blight"]:
        success, report, error = pest_service.predict_pest(pest_fixture(fixture_name))
        assert success, error
        crops.add(report["crop"])

    assert len(crops) == 4


def test_pest_input_tensor_shape_and_range():
    tensor = preprocess_pest_image(pest_fixture("tomato_healthy"))

    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert tensor.dtype.is_floating_point


def test_exif_orientation_is_applied(tmp_path):
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:, :, 1] = 180
    exif = Image.Exif()
    exif[274] = 6

    image_path = tmp_path / "rotated.jpg"
    Image.fromarray(pixels).save(image_path, exif=exif)

    loaded = load_leaf_image(str(image_path))
    assert loaded.shape[:2] == (80, 40)


def test_grayscale_image_is_accepted(tmp_path):
    source = cv2.imread(pest_fixture("tomato_healthy"))
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    image_path = tmp_path / "gray.png"
    cv2.imwrite(str(image_path), gray)

    loaded = load_leaf_image(str(image_path))
    assert loaded.ndim == 3 and loaded.shape[2] == 3


def test_missing_file_returns_error():
    success, report, error = pest_service.predict_pest("/nonexistent/leaf.jpg")

    assert not success
    assert report is None
    assert error


def test_build_pest_input_accepts_bgr_array():
    image = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)

    tensor = build_pest_input(image)
    assert tuple(tensor.shape) == (3, 224, 224)
