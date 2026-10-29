import json
import logging
import os

import torch
from torchvision.models.mobilenetv2 import MobileNetV2

from config import Config
from services import efficient_classifier
from services.pest_knowledge import (
    UNCERTAIN_REPORT,
    UNSUPPORTED_REPORT,
    TREATMENTS,
    get_condition_display_name,
    get_crop_display_name,
    is_healthy_class,
    is_supported_class,
)
from utils.preprocessing import build_pest_input, build_stability_variants, load_leaf_image

logger = logging.getLogger(__name__)

STATUS_DISEASE = "disease"
STATUS_HEALTHY = "healthy"
STATUS_UNCERTAIN = "uncertain"
STATUS_UNSUPPORTED = "unsupported"

SCOPE_DISCLAIMER = (
    "This result describes only the submitted leaf image. It is not a field-wide "
    "diagnosis and does not replace an agronomist or extension service."
)

_pest_model = None
_pest_classes = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pest_model():
    global _pest_model, _pest_classes

    if _pest_model is None:
        torch.serialization.add_safe_globals([MobileNetV2])

        model = torch.load(
            Config.PEST_MODEL_PATH,
            map_location=_device,
            weights_only=False
        )

        model.to(_device)
        model.eval()
        _pest_model = model

    if _pest_classes is None:
        if not os.path.exists(Config.PEST_CLASSES_PATH):
            raise RuntimeError(f"Class label file not found: {Config.PEST_CLASSES_PATH}")

        with open(Config.PEST_CLASSES_PATH, "r", encoding="utf-8") as f:
            _pest_classes = json.load(f)

    return _pest_model, _pest_classes, _device


def confidence_label(confidence):
    if confidence is None:
        return "unknown"
    if confidence >= 0.85:
        return "high"
    if confidence >= Config.PEST_MIN_CONFIDENCE:
        return "moderate"
    return "low"


def prediction_entropy(probabilities):
    return float(-(probabilities * torch.log(probabilities + 1e-12)).sum())


def meets_confidence_gate(confidence, entropy, stable, min_confidence=None,
                          max_entropy=None, require_stable=None):
    min_confidence = Config.PEST_MIN_CONFIDENCE if min_confidence is None else min_confidence
    max_entropy = Config.PEST_MAX_ENTROPY if max_entropy is None else max_entropy
    require_stable = Config.PEST_STABILITY_CHECK if require_stable is None else require_stable

    if confidence < min_confidence:
        return False
    if entropy > max_entropy:
        return False
    if require_stable and not stable:
        return False
    return True


def generate_treatment_report(class_name, confidence=None, entropy=None, stable=True):
    return build_report(class_name, confidence, entropy, stable)


def build_report(class_name, confidence=None, entropy=None, stable=True,
                 min_confidence=None, max_entropy=None, require_stable=None):
    min_confidence = Config.PEST_MIN_CONFIDENCE if min_confidence is None else min_confidence
    max_entropy = Config.PEST_MAX_ENTROPY if max_entropy is None else max_entropy
    require_stable = Config.PEST_STABILITY_CHECK if require_stable is None else require_stable

    base = {
        "disease_name": class_name,
        "crop": None,
        "condition": None,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "entropy": entropy,
        "stable": stable,
        "status": STATUS_UNCERTAIN,
        "notice": "",
        "disclaimer": SCOPE_DISCLAIMER,
    }

    if class_name is None or not is_supported_class(class_name):
        base.update(UNSUPPORTED_REPORT)
        base["status"] = STATUS_UNSUPPORTED
        base["disease_name"] = "Unsupported image"
        base["notice"] = (
            "The image does not match any of the 38 supported plant classes, so no crop or "
            "disease is reported."
        )
        return base

    if not meets_confidence_gate(
        confidence if confidence is not None else 1.0,
        entropy if entropy is not None else 0.0,
        stable,
        min_confidence=min_confidence,
        max_entropy=max_entropy,
        require_stable=require_stable
    ):
        base.update(UNCERTAIN_REPORT)
        base["status"] = STATUS_UNCERTAIN
        base["disease_name"] = "Inconclusive"
        base["notice"] = (
            "Prediction confidence is too low or not stable enough to confirm a crop or "
            "disease for this image."
        )
        return base

    knowledge = TREATMENTS[class_name]
    base.update({
        "crop": get_crop_display_name(class_name),
        "condition": get_condition_display_name(class_name),
        "root_cause": knowledge["root_cause"],
        "treatment": knowledge["treatment"],
        "prevention": knowledge["prevention"],
    })

    if is_healthy_class(class_name):
        base["status"] = STATUS_HEALTHY
        base["notice"] = (
            "No disease or pest damage was detected on this leaf. This does not certify the "
            "rest of the plant or the field as healthy."
        )
    else:
        base["status"] = STATUS_DISEASE
        base["notice"] = (
            "Damage consistent with this condition was detected on the leaf. Confirm before "
            "treating, because several leaf diseases look alike in a single photo."
        )

    return base


PRIMARY_CROP_FAMILIES = {
    "Pepper,_bell": "Pepper",
    "Potato": "Potato",
    "Tomato": "Tomato",
}

ACCEPTED_STATUSES = (STATUS_DISEASE, STATUS_HEALTHY)


def primary_crop_mass(probabilities, classes, crop_family):
    """Probability mass the primary model assigns to a crop family."""
    if crop_family is None:
        return 0.0

    total = 0.0
    for index, class_name in enumerate(classes):
        if index >= len(probabilities):
            break
        if "___" not in class_name:
            continue
        if PRIMARY_CROP_FAMILIES.get(class_name.split("___", 1)[0]) == crop_family:
            total += float(probabilities[index])
    return total


def run_primary_prediction(image_bgr):
    model, classes, device = load_pest_model()

    variants = build_stability_variants(image_bgr) if Config.PEST_STABILITY_CHECK else [
        build_pest_input(image_bgr)
    ]

    batch = torch.stack(variants).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)

    primary = probabilities[0]
    class_idx = int(torch.argmax(primary).item())
    confidence = float(primary[class_idx].item())
    entropy = prediction_entropy(primary)
    stable = len({int(torch.argmax(p).item()) for p in probabilities}) == 1
    class_name = classes[class_idx] if class_idx < len(classes) else None

    report = build_report(class_name, confidence, entropy, stable)
    top = torch.topk(primary, min(3, len(classes)))
    info = {
        "status": report["status"],
        "class": class_name,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "stable": stable,
        "accepted": report["status"] in ACCEPTED_STATUSES,
        "top3": [
            (classes[int(index)], round(float(primary[int(index)].item()), 4))
            for index in top.indices
        ],
    }

    return report, primary, classes, info


def meets_fallback_criteria(prediction):
    if prediction["app_class"] is None:
        return False, "label_not_supported"
    if prediction["crop_family"] is None:
        return False, "crop_not_supported"
    if prediction["confidence"] < Config.FALLBACK_MIN_CONFIDENCE:
        return False, "low_confidence"
    if prediction["entropy"] > Config.FALLBACK_MAX_ENTROPY:
        return False, "high_entropy"
    if Config.FALLBACK_STABILITY_CHECK and not prediction["stable"]:
        return False, "unstable_prediction"
    return True, None


def build_fallback_report(app_class, confidence, entropy, stable):
    return build_report(
        app_class,
        confidence,
        entropy,
        stable,
        min_confidence=Config.FALLBACK_MIN_CONFIDENCE,
        max_entropy=Config.FALLBACK_MAX_ENTROPY,
        require_stable=Config.FALLBACK_STABILITY_CHECK
    )


def empty_fallback_info(available=None):
    return {
        "attempted": False,
        "available": efficient_classifier.is_available() if available is None else available,
        "accepted": False,
        "class": None,
        "crop_family": None,
        "confidence": None,
        "entropy": None,
        "stable": None,
        "top3": None,
        "primary_crop_mass": None,
        "reason": None,
    }


def attempt_fallback(image_bgr, primary_probabilities, primary_classes):
    """Try the secondary 15-class classifier for an otherwise rejected image."""
    info = empty_fallback_info()

    if not Config.FALLBACK_ENABLED:
        info["reason"] = "disabled"
        return None, info

    info["attempted"] = True

    try:
        prediction = efficient_classifier.predict(image_bgr)
    except efficient_classifier.ClassifierUnavailable as error:
        info["reason"] = "unavailable"
        logger.warning("Secondary classifier unavailable: %s", error)
        return None, info
    except Exception as error:
        info["reason"] = "inference_error"
        logger.warning("Secondary classifier inference failed: %s", type(error).__name__)
        return None, info

    info.update({
        "class": prediction["label"],
        "crop_family": prediction["crop_family"],
        "confidence": round(prediction["confidence"], 4),
        "entropy": round(prediction["entropy"], 4),
        "stable": prediction["stable"],
        "top3": prediction.get("top3"),
    })

    crop_mass = primary_crop_mass(
        primary_probabilities,
        primary_classes,
        prediction["crop_family"]
    )
    info["primary_crop_mass"] = round(crop_mass, 4)

    accepted, reason = meets_fallback_criteria(prediction)
    if not accepted:
        info["reason"] = reason
        return None, info

    if crop_mass < Config.FALLBACK_MIN_PRIMARY_CROP_MASS:
        info["reason"] = "crop_eligibility_not_established"
        return None, info

    report = build_fallback_report(
        prediction["app_class"],
        prediction["confidence"],
        prediction["entropy"],
        prediction["stable"]
    )

    if report["status"] not in ACCEPTED_STATUSES:
        info["reason"] = "report_not_accepted"
        return None, info

    info["accepted"] = True
    return report, info


def log_diagnostics(diagnostics):
    logger.info(
        "leaf prediction primary=%s primary_class=%s primary_conf=%s "
        "primary_entropy=%s primary_stable=%s fallback_attempted=%s "
        "fallback_available=%s fallback_class=%s fallback_conf=%s "
        "fallback_entropy=%s fallback_stable=%s primary_crop_mass=%s "
        "decision=%s reason=%s",
        diagnostics["primary"]["status"],
        diagnostics["primary"]["class"],
        diagnostics["primary"]["confidence"],
        diagnostics["primary"]["entropy"],
        diagnostics["primary"]["stable"],
        diagnostics["fallback"]["attempted"],
        diagnostics["fallback"]["available"],
        diagnostics["fallback"]["class"],
        diagnostics["fallback"]["confidence"],
        diagnostics["fallback"]["entropy"],
        diagnostics["fallback"]["stable"],
        diagnostics["fallback"]["primary_crop_mass"],
        diagnostics["decision"],
        diagnostics["reason"],
    )


def predict_pest_with_diagnostics(image_path):
    try:
        image = load_leaf_image(image_path)

        report, primary_probabilities, primary_classes, primary_info = \
            run_primary_prediction(image)

        diagnostics = {
            "primary": primary_info,
            "fallback": empty_fallback_info(),
            "decision": "primary",
            "reason": None,
        }

        if primary_info["accepted"]:
            log_diagnostics(diagnostics)
            return True, report, diagnostics, None

        diagnostics["reason"] = primary_info["status"]

        fallback_report, fallback_info = attempt_fallback(
            image,
            primary_probabilities,
            primary_classes
        )
        diagnostics["fallback"] = fallback_info

        if fallback_report is not None:
            diagnostics["decision"] = "secondary"
            diagnostics["reason"] = None
            log_diagnostics(diagnostics)
            return True, fallback_report, diagnostics, None

        diagnostics["decision"] = "inconclusive"
        diagnostics["reason"] = fallback_info.get("reason") or primary_info["status"]
        log_diagnostics(diagnostics)
        return True, report, diagnostics, None

    except Exception as e:
        logger.warning("Leaf prediction failed: %s", type(e).__name__)
        return False, None, None, str(e)


def predict_pest(image_path):
    success, report, _diagnostics, error = predict_pest_with_diagnostics(image_path)
    return success, report, error
