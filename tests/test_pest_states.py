import torch

from config import Config
from services import pest_service
from services.pest_knowledge import TREATMENTS, UNCERTAIN_REPORT, UNSUPPORTED_REPORT


def test_confident_disease_reports_crop_and_condition():
    report = pest_service.build_report("Tomato___Early_blight", 0.97, 0.05, True)

    assert report["status"] == pest_service.STATUS_DISEASE
    assert report["crop"] == "Tomato"
    assert report["condition"] == "Early blight"
    assert report["treatment"] == TREATMENTS["Tomato___Early_blight"]["treatment"]
    assert report["prevention"] == TREATMENTS["Tomato___Early_blight"]["prevention"]


def test_confident_healthy_report_is_not_a_threat():
    report = pest_service.build_report("Corn_(maize)___healthy", 0.99, 0.01, True)

    assert report["status"] == pest_service.STATUS_HEALTHY
    assert report["crop"] == "Corn (maize)"
    assert "No disease" in report["notice"]
    assert "Attention" not in report["notice"]


def test_low_confidence_is_uncertain_and_claims_nothing():
    report = pest_service.build_report("Grape___Black_rot", 0.31, 0.10, True)

    assert report["status"] == pest_service.STATUS_UNCERTAIN
    assert report["crop"] is None
    assert report["condition"] is None
    assert report["disease_name"] == "Inconclusive"
    assert report["treatment"] == UNCERTAIN_REPORT["treatment"]


def test_unstable_prediction_is_uncertain():
    report = pest_service.build_report("Grape___Black_rot", 0.95, 0.05, False)

    assert report["status"] == pest_service.STATUS_UNCERTAIN
    assert report["crop"] is None


def test_high_entropy_is_uncertain():
    report = pest_service.build_report("Apple___Apple_scab", 0.99, 1.4, True)

    assert report["status"] == pest_service.STATUS_UNCERTAIN


def test_unsupported_class_is_not_diagnosed():
    report = pest_service.build_report("Banana___healthy", 0.99, 0.01, True)

    assert report["status"] == pest_service.STATUS_UNSUPPORTED
    assert report["crop"] is None
    assert report["root_cause"] == UNSUPPORTED_REPORT["root_cause"]
    assert report["treatment"] == UNSUPPORTED_REPORT["treatment"]


def test_missing_class_index_is_unsupported():
    report = pest_service.build_report(None, 0.99, 0.01, True)

    assert report["status"] == pest_service.STATUS_UNSUPPORTED
    assert report["crop"] is None


def test_confidence_gate_thresholds_are_enforced():
    assert pest_service.meets_confidence_gate(
        Config.PEST_MIN_CONFIDENCE, 0.0, True
    )
    assert not pest_service.meets_confidence_gate(
        Config.PEST_MIN_CONFIDENCE - 0.01, 0.0, True
    )
    assert not pest_service.meets_confidence_gate(
        0.99, Config.PEST_MAX_ENTROPY + 0.01, True
    )
    assert not pest_service.meets_confidence_gate(0.99, 0.0, False)


def test_entropy_of_one_hot_distribution_is_zero():
    probabilities = torch.tensor([1.0, 0.0, 0.0])
    assert pest_service.prediction_entropy(probabilities) == 0.0


def test_generate_treatment_report_wrapper_matches_build_report():
    wrapped = pest_service.generate_treatment_report("Potato___Late_blight", 0.9, 0.02, True)
    direct = pest_service.build_report("Potato___Late_blight", 0.9, 0.02, True)

    assert wrapped == direct
