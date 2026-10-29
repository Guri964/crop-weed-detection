import json
import os

import pytest
import torch

from config import Config
from services import efficient_classifier, pest_service
from services.pest_knowledge import TREATMENTS, is_healthy_class

from conftest import pest_fixture

with open(Config.PEST_CLASSES_PATH, "r", encoding="utf-8") as handle:
    PRIMARY_CLASSES = json.load(handle)

with open(Config.FALLBACK_CLASSES_PATH, "r", encoding="utf-8") as handle:
    SECONDARY_CLASSES = json.load(handle)

FORBIDDEN_CLIENT_TERMS = [
    "fallback",
    "secondary",
    "agrisense",
    "efficientnet",
    "model 1",
    "model 2",
    "model switch",
    "efficient_model",
    ".pth",
    ".pt",
    "traceback",
    "softmax",
    "logit",
]


def primary_vector(class_name, mass=1.0):
    vector = torch.zeros(len(PRIMARY_CLASSES))
    vector[PRIMARY_CLASSES.index(class_name)] = mass
    return vector


def primary_info(class_name, confidence, entropy, stable):
    report = pest_service.build_report(class_name, confidence, entropy, stable)
    return {
        "status": report["status"],
        "class": class_name,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "stable": stable,
        "accepted": report["status"] in pest_service.ACCEPTED_STATUSES,
    }


def install_primary(monkeypatch, class_name, confidence, entropy, stable, vector):
    info = primary_info(class_name, confidence, entropy, stable)
    report = pest_service.build_report(class_name, confidence, entropy, stable)

    def fake_run(image_bgr):
        return report, vector, PRIMARY_CLASSES, info

    monkeypatch.setattr(pest_service, "run_primary_prediction", fake_run)


def secondary_prediction(label, confidence, entropy, stable):
    app_class = efficient_classifier.resolve_app_class(label)
    return {
        "label": label,
        "app_class": app_class,
        "crop_family": efficient_classifier.app_crop_family(app_class),
        "confidence": confidence,
        "entropy": entropy,
        "stable": stable,
        "top3": [(label, confidence)],
        "probabilities": None,
    }


def test_primary_accepted_never_calls_secondary(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Early_blight",
        0.97,
        0.05,
        True,
        primary_vector("Tomato___Early_blight")
    )

    def explode(_image):
        raise AssertionError("secondary classifier must not run")

    monkeypatch.setattr(efficient_classifier, "predict", explode)

    success, report, diagnostics, error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert success, error
    assert report["status"] == pest_service.STATUS_DISEASE
    assert report["crop"] == "Tomato"
    assert diagnostics["decision"] == "primary"
    assert diagnostics["fallback"]["attempted"] is False


def test_primary_healthy_result_never_calls_secondary(monkeypatch):
    install_primary(
        monkeypatch,
        "Corn_(maize)___healthy",
        0.99,
        0.01,
        True,
        primary_vector("Corn_(maize)___healthy")
    )

    def explode(_image):
        raise AssertionError("secondary classifier must not run")

    monkeypatch.setattr(efficient_classifier, "predict", explode)

    success, report, error = pest_service.predict_pest(pest_fixture("corn_healthy"))
    assert success, error
    assert report["status"] == pest_service.STATUS_HEALTHY


def test_primary_uncertain_runs_eligible_secondary(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.42,
        1.30,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.93, 0.21, True)
    )

    success, report, diagnostics, error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert success, error
    assert diagnostics["fallback"]["attempted"] is True
    assert diagnostics["decision"] == "secondary"
    assert report["status"] == pest_service.STATUS_DISEASE
    assert report["crop"] == "Tomato"
    assert report["condition"] == "Early blight"
    assert report["treatment"] == TREATMENTS["Tomato___Early_blight"]["treatment"]


@pytest.mark.parametrize(
    "class_name, vector_class",
    [
        ("Tomato___Late_blight", "Tomato___Late_blight"),
        ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
    ],
)
def test_primary_low_confidence_triggers_secondary(monkeypatch, class_name, vector_class):
    install_primary(monkeypatch, class_name, 0.35, 0.90, True, primary_vector(vector_class))
    calls = []
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: calls.append(1) or secondary_prediction("Potato___Late_blight", 0.95, 0.15, True)
    )

    _success, _report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("potato_late_blight")
    )

    assert diagnostics["primary"]["status"] == pest_service.STATUS_UNCERTAIN
    assert calls == [1]
    assert diagnostics["fallback"]["attempted"] is True


def test_primary_unstable_triggers_secondary(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Early_blight",
        0.71,
        0.55,
        False,
        primary_vector("Tomato___Early_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.90, 0.30, True)
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert diagnostics["primary"]["stable"] is False
    assert diagnostics["fallback"]["attempted"] is True
    assert report["status"] == pest_service.STATUS_DISEASE


def test_primary_unsupported_runs_secondary_only_with_crop_eligibility(monkeypatch):
    info = {
        "status": pest_service.STATUS_UNSUPPORTED,
        "class": None,
        "confidence": 0.99,
        "entropy": 0.01,
        "stable": True,
        "accepted": False,
    }
    report = pest_service.build_report(None, 0.99, 0.01, True)
    monkeypatch.setattr(
        pest_service,
        "run_primary_prediction",
        lambda _image: (report, torch.zeros(len(PRIMARY_CLASSES)), PRIMARY_CLASSES, info)
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Late_blight", 0.96, 0.12, True)
    )

    _success, rejected_report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert diagnostics["primary"]["status"] == pest_service.STATUS_UNSUPPORTED
    assert diagnostics["fallback"]["attempted"] is True
    assert diagnostics["fallback"]["reason"] == "crop_eligibility_not_established"
    assert rejected_report["status"] == pest_service.STATUS_UNSUPPORTED
    assert rejected_report["crop"] is None


def test_unsupported_crop_is_not_diagnosed_by_secondary(monkeypatch):
    install_primary(
        monkeypatch,
        "Apple___Apple_scab",
        0.44,
        1.20,
        False,
        primary_vector("Apple___Apple_scab")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Septoria_leaf_spot", 0.97, 0.18, True)
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("apple_scab")
    )

    assert diagnostics["fallback"]["crop_family"] == "Tomato"
    assert diagnostics["fallback"]["primary_crop_mass"] == 0.0
    assert diagnostics["fallback"]["reason"] == "crop_eligibility_not_established"
    assert report["status"] == pest_service.STATUS_UNCERTAIN
    assert report["crop"] is None
    assert report["condition"] is None


def test_secondary_low_confidence_is_rejected(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.40,
        1.40,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction(
            "Tomato_Early_blight", Config.FALLBACK_MIN_CONFIDENCE - 0.05, 0.40, True
        )
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert diagnostics["fallback"]["reason"] == "low_confidence"
    assert diagnostics["decision"] == "inconclusive"
    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_secondary_unstable_prediction_is_rejected(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.40,
        1.40,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.99, 0.05, False)
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert diagnostics["fallback"]["reason"] == "unstable_prediction"
    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_secondary_unsupported_label_is_rejected(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.40,
        1.40,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: {
            "label": "Banana___healthy",
            "app_class": None,
            "crop_family": None,
            "confidence": 0.99,
            "entropy": 0.01,
            "stable": True,
            "top3": [("Banana___healthy", 0.99)],
            "probabilities": None,
        }
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert diagnostics["fallback"]["reason"] == "label_not_supported"
    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_both_models_rejecting_returns_safe_inconclusive(monkeypatch):
    install_primary(
        monkeypatch,
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        0.52,
        1.10,
        False,
        primary_vector("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.30, 1.90, False)
    )

    success, report, diagnostics, error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert success, error
    assert diagnostics["decision"] == "inconclusive"
    assert report["status"] == pest_service.STATUS_UNCERTAIN
    assert report["crop"] is None
    assert report["condition"] is None
    assert report["disease_name"] == "Inconclusive"
    assert "no treatment" in report["treatment"].lower()


def test_missing_secondary_model_is_handled_safely(monkeypatch):
    monkeypatch.setattr(Config, "FALLBACK_MODEL_PATH", "/nonexistent/model.pth")
    monkeypatch.setattr(Config, "FALLBACK_CLASSES_PATH", "/nonexistent/classes.json")
    monkeypatch.setattr(efficient_classifier, "_model", None)
    monkeypatch.setattr(efficient_classifier, "_classes", None)
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )

    success, report, diagnostics, error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert success, error
    assert diagnostics["fallback"]["available"] is False
    assert diagnostics["fallback"]["reason"] == "unavailable"
    assert report["status"] == pest_service.STATUS_UNCERTAIN
    assert report["crop"] is None


def test_secondary_inference_failure_is_handled_safely(monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )

    def boom(_image):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(efficient_classifier, "predict", boom)

    success, report, diagnostics, error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert success, error
    assert diagnostics["fallback"]["reason"] == "inference_error"
    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_secondary_disabled_flag_skips_without_crashing(monkeypatch):
    monkeypatch.setattr(Config, "FALLBACK_ENABLED", False)
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture("tomato_early_blight")
    )

    assert diagnostics["fallback"]["attempted"] is False
    assert diagnostics["fallback"]["reason"] == "disabled"
    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_secondary_class_index_mapping_covers_all_fifteen_classes():
    assert len(SECONDARY_CLASSES) == 15
    assert list(efficient_classifier.LABEL_TO_APP_CLASS) == SECONDARY_CLASSES


@pytest.mark.parametrize("index,label", list(enumerate(SECONDARY_CLASSES)))
def test_secondary_label_maps_to_supported_crop_and_treatment(index, label):
    app_class = efficient_classifier.resolve_app_class(label)

    assert app_class is not None
    assert app_class in TREATMENTS
    assert efficient_classifier.app_crop_family(app_class) in efficient_classifier.SUPPORTED_CROP_FAMILIES
    assert efficient_classifier.app_crop_family(app_class) == index_crop(index)


def index_crop(index):
    if index < 2:
        return "Pepper"
    if index < 5:
        return "Potato"
    return "Tomato"


def test_secondary_healthy_and_disease_classes_are_mapped_correctly():
    for label in SECONDARY_CLASSES:
        app_class = efficient_classifier.resolve_app_class(label)
        label_is_healthy = label.lower().endswith("healthy")
        assert is_healthy_class(app_class) is label_is_healthy


def test_previous_primary_report_shape_is_unchanged():
    report = pest_service.build_report("Tomato___Early_blight", 0.99, 0.02, True)

    for key in [
        "disease_name", "crop", "condition", "confidence", "confidence_label",
        "entropy", "stable", "status", "notice", "disclaimer", "root_cause",
        "treatment", "prevention",
    ]:
        assert key in report

    assert report["disclaimer"]
    assert report["status"] == pest_service.STATUS_DISEASE


def test_secondary_report_uses_its_own_thresholds():
    report = pest_service.build_fallback_report("Tomato___Early_blight", 0.80, 0.75, True)

    assert report["status"] == pest_service.STATUS_DISEASE
    assert report["crop"] == "Tomato"


def test_secondary_report_still_respects_its_thresholds():
    report = pest_service.build_fallback_report("Tomato___Early_blight", 0.55, 0.20, True)
    assert report["status"] == pest_service.STATUS_UNCERTAIN

    report = pest_service.build_fallback_report("Tomato___Early_blight", 0.95, 1.10, True)
    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_primary_crop_mass_only_counts_matching_family():
    vector = primary_vector("Potato___Late_blight", 0.5)
    vector[PRIMARY_CLASSES.index("Tomato___Late_blight")] = 0.25

    assert pest_service.primary_crop_mass(vector, PRIMARY_CLASSES, "Potato") == pytest.approx(0.5)
    assert pest_service.primary_crop_mass(vector, PRIMARY_CLASSES, "Tomato") == pytest.approx(0.25)
    assert pest_service.primary_crop_mass(vector, PRIMARY_CLASSES, "Pepper") == pytest.approx(0.0)
    assert pest_service.primary_crop_mass(vector, PRIMARY_CLASSES, None) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "fixture_name,expected_crop",
    [
        ("pepper_bacterial_spot", "Bell pepper"),
        ("potato_late_blight", "Potato"),
        ("tomato_early_blight", "Tomato"),
        ("tomato_healthy", "Tomato"),
    ],
)
def test_real_secondary_model_predicts_each_supported_crop(monkeypatch, fixture_name, expected_crop):
    if not efficient_classifier.is_available():
        pytest.skip("secondary classifier artifacts are not present")

    crop_class = {
        "Bell pepper": "Pepper,_bell___Bacterial_spot",
        "Potato": "Potato___Late_blight",
        "Tomato": "Tomato___Late_blight",
    }[expected_crop]

    install_primary(
        monkeypatch,
        crop_class,
        0.31,
        1.60,
        False,
        primary_vector(crop_class)
    )

    _success, report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
        pest_fixture(fixture_name)
    )

    assert diagnostics["decision"] == "secondary"
    assert report["crop"] == expected_crop
    assert report["status"] in (pest_service.STATUS_DISEASE, pest_service.STATUS_HEALTHY)


def test_real_secondary_model_rejects_unsupported_crop_family():
    if not efficient_classifier.is_available():
        pytest.skip("secondary classifier artifacts are not present")

    from services.pest_service import load_leaf_image

    image = load_leaf_image(pest_fixture("corn_healthy"))
    prediction = efficient_classifier.predict(image)
    vector = primary_vector("Corn_(maize)___healthy")

    mass = pest_service.primary_crop_mass(
        vector, PRIMARY_CLASSES, prediction["crop_family"]
    )

    assert mass < Config.FALLBACK_MIN_PRIMARY_CROP_MASS


def test_secondary_model_artifacts_are_loadable():
    if not efficient_classifier.is_available():
        pytest.skip("secondary classifier artifacts are not present")

    model, classes, _device = efficient_classifier.load_classifier()

    assert len(classes) == 15
    assert model.base_model.classifier[1].out_features == 15


def test_secondary_preprocessing_starts_from_rgb():
    if not efficient_classifier.is_available():
        pytest.skip("secondary classifier artifacts are not present")

    import numpy as np

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    tensor = efficient_classifier.build_input(image)
    assert tuple(tensor.shape) == (3, Config.FALLBACK_IMAGE_SIZE, Config.FALLBACK_IMAGE_SIZE)
    assert tensor[0].max().item() > 2.0


def assert_no_internal_terms(payload):
    lowered = payload.lower()
    for term in FORBIDDEN_CLIENT_TERMS:
        assert term not in lowered, f"client payload leaked internal term: {term}"


def test_api_response_does_not_leak_model_details(authenticated_client, monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.93, 0.21, True)
    )

    with open(pest_fixture("tomato_early_blight"), "rb") as handle:
        response = authenticated_client.post(
            "/api/pest/predict",
            data={"file": (handle, "leaf.jpg")},
            content_type="multipart/form-data"
        )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["data"]["crop"] == "Tomato"
    assert set(payload["data"]).issuperset(
        {"disease_name", "crop", "condition", "confidence", "treatment", "prevention"}
    )
    assert "diagnostics" not in payload
    assert_no_internal_terms(json.dumps(payload))


def test_result_page_does_not_leak_model_details(authenticated_client, monkeypatch):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.93, 0.21, True)
    )

    with open(pest_fixture("tomato_early_blight"), "rb") as handle:
        response = authenticated_client.post(
            "/pest/",
            data={"file": (handle, "leaf.jpg")},
            content_type="multipart/form-data"
        )

    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Early blight" in body
    assert "Prevention" in body
    assert_no_internal_terms(body)


def test_inconclusive_page_is_user_friendly(authenticated_client, monkeypatch):
    install_primary(
        monkeypatch,
        "Corn_(maize)___healthy",
        0.30,
        1.70,
        False,
        primary_vector("Corn_(maize)___healthy")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.25, 2.00, False)
    )

    with open(pest_fixture("corn_healthy"), "rb") as handle:
        response = authenticated_client.post(
            "/pest/",
            data={"file": (handle, "leaf.jpg")},
            content_type="multipart/form-data"
        )

    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Inconclusive" in body
    assert "Needs Better Image" in body
    assert_no_internal_terms(body)


def test_diagnostics_are_available_for_logging(monkeypatch, caplog):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.93, 0.21, True)
    )

    with caplog.at_level("INFO", logger="services.pest_service"):
        _success, _report, diagnostics, _error = pest_service.predict_pest_with_diagnostics(
            pest_fixture("tomato_early_blight")
        )

    assert diagnostics["primary"]["status"] == pest_service.STATUS_UNCERTAIN
    assert diagnostics["fallback"]["accepted"] is True
    assert diagnostics["decision"] == "secondary"
    assert any("leaf prediction" in record.message for record in caplog.records)


def test_diagnostic_log_line_contains_no_file_paths(monkeypatch, caplog):
    install_primary(
        monkeypatch,
        "Tomato___Late_blight",
        0.41,
        1.35,
        False,
        primary_vector("Tomato___Late_blight")
    )
    monkeypatch.setattr(
        efficient_classifier,
        "predict",
        lambda _image: secondary_prediction("Tomato_Early_blight", 0.93, 0.21, True)
    )

    with caplog.at_level("INFO", logger="services.pest_service"):
        pest_service.predict_pest_with_diagnostics(pest_fixture("tomato_early_blight"))

    joined = " ".join(record.getMessage() for record in caplog.records)
    assert Config.BASE_DIR not in joined
    assert os.sep + "fixtures" not in joined


def test_weed_module_is_untouched_by_the_pipeline():
    from services import weed_service

    assert weed_service.STATE_WEED_DETECTED == "weed_detected"
    assert set(weed_service.load_weed_model().names.values()) == {"crop", "weed"}
