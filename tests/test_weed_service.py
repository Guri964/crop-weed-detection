import cv2
import numpy as np
import pytest

from config import Config
from services import weed_service
from utils.visualization import draw_weed_detections

from conftest import pest_fixture

NAMES = {0: "crop", 1: "weed"}


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def item(self):
        return self.values[0] if isinstance(self.values, list) else self.values

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = FakeTensor(class_id)
        self.conf = FakeTensor(confidence)
        self.xyxy = [FakeTensor(xyxy)]


class FakeResult:
    def __init__(self, boxes, image):
        self.boxes = boxes
        self.orig_img = image


def test_class_kinds_use_model_names():
    kinds = weed_service.build_class_kinds(NAMES)

    assert kinds[0] == "crop"
    assert kinds[1] == "weed"


def test_detections_are_labelled_from_model_names():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = FakeResult([
        FakeBox(0, 0.9, [10.0, 10.0, 40.0, 40.0]),
        FakeBox(1, 0.8, [50.0, 50.0, 90.0, 95.0]),
    ], image)

    detections = weed_service.extract_detections(result, NAMES)

    assert [d["label"] for d in detections] == ["crop", "weed"]
    assert [d["kind"] for d in detections] == ["crop", "weed"]
    assert detections[1]["box"] == [50.0, 50.0, 90.0, 95.0]


def test_crop_only_detections_report_no_weed():
    detections = [
        {"class_id": 0, "label": "crop", "kind": "crop", "confidence": 0.9, "box": [0, 0, 10, 10]},
    ]

    report = weed_service.build_weed_report(detections, 0.25)

    assert report["state"] == weed_service.STATE_NO_WEED
    assert report["weed_count"] == 0
    assert report["crop_count"] == 1
    assert report["detections"] == []
    assert "No weeds" in report["summary"]


def test_mixed_detections_report_only_weeds():
    detections = [
        {"class_id": 0, "label": "crop", "kind": "crop", "confidence": 0.7, "box": [0, 0, 10, 10]},
        {"class_id": 1, "label": "weed", "kind": "weed", "confidence": 0.6, "box": [20, 20, 40, 40]},
        {"class_id": 1, "label": "weed", "kind": "weed", "confidence": 0.5, "box": [50, 50, 70, 70]},
    ]

    report = weed_service.build_weed_report(detections, 0.25)

    assert report["state"] == weed_service.STATE_WEED_DETECTED
    assert report["weed_count"] == 2
    assert report["crop_count"] == 1
    assert all(detection["label"] == "weed" for detection in report["detections"])
    assert report["detections"][0]["confidence"] == 0.6


def test_empty_detections_report_no_assessment():
    report = weed_service.build_weed_report([], 0.25)

    assert report["state"] == weed_service.STATE_NO_DETECTION
    assert report["weed_count"] == 0
    assert "cannot" in report["summary"] or "No crop or weed" in report["summary"]


def test_annotation_draws_only_supplied_weed_boxes():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    weed_box = {"label": "weed", "confidence": 0.87, "box": [30, 30, 80, 80]}

    annotated = draw_weed_detections(image, [weed_box])

    assert annotated.shape == image.shape
    assert annotated[55, 30].tolist() != [0, 0, 0]
    assert annotated[10, 10].tolist() == [0, 0, 0]
    assert annotated[110, 110].tolist() == [0, 0, 0]


def test_annotation_leaves_image_untouched_without_weeds():
    image = np.full((60, 60, 3), 40, dtype=np.uint8)

    annotated = draw_weed_detections(image, [])

    assert np.array_equal(annotated, image)


def test_predict_weed_annotates_only_weed_class(monkeypatch, tmp_path):
    image = np.full((160, 160, 3), 30, dtype=np.uint8)

    class FakeModel:
        names = {0: "crop", 1: "weed"}

        def predict(self, **kwargs):
            boxes = [
                FakeBox(0, 0.92, [10.0, 10.0, 70.0, 70.0]),
                FakeBox(1, 0.88, [90.0, 90.0, 150.0, 150.0]),
            ]
            return [FakeResult(boxes, image)]

    monkeypatch.setattr(weed_service, "load_weed_model", lambda: FakeModel())
    monkeypatch.setattr(weed_service.Config, "OUTPUT_WEED_DIR", str(tmp_path))

    success, report, error = weed_service.predict_weed(pest_fixture("tomato_healthy"))

    assert success, error
    assert report["state"] == weed_service.STATE_WEED_DETECTED
    assert report["weed_count"] == 1
    assert report["crop_count"] == 1

    annotated = cv2.imread(report["output_path"])
    assert annotated[90, 120].tolist() != [30, 30, 30]
    assert annotated[10, 40].tolist() == [30, 30, 30]
    assert annotated[40, 40].tolist() == [30, 30, 30]


def test_model_without_weed_class_is_rejected(monkeypatch):
    class FakeModel:
        names = {0: "crop"}

        def predict(self, **kwargs):
            return []

    monkeypatch.setattr(weed_service, "load_weed_model", lambda: FakeModel())

    success, report, error = weed_service.predict_weed(pest_fixture("tomato_healthy"))

    assert not success
    assert report is None
    assert "weed class" in error


def test_real_model_separates_crop_and_weed_classes():
    model = weed_service.load_weed_model()

    assert isinstance(model.names, dict)
    assert set(model.names.values()) == {"crop", "weed"}


@pytest.mark.parametrize("fixture_name", ["corn_healthy", "tomato_early_blight"])
def test_real_model_pipeline_states(fixture_name):
    success, report, error = weed_service.predict_weed(pest_fixture(fixture_name))

    assert success, error
    assert report["state"] in {
        weed_service.STATE_WEED_DETECTED,
        weed_service.STATE_NO_WEED,
        weed_service.STATE_NO_DETECTION,
    }
    assert report["weed_count"] == len(report["detections"])
    assert report["threshold"] == Config.WEED_CONFIDENCE_THRESHOLD
    assert report["scope_note"]
