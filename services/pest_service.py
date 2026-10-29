import json
import os

import torch
from torchvision.models.mobilenetv2 import MobileNetV2

from config import Config
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


def meets_confidence_gate(confidence, entropy, stable):
    if confidence < Config.PEST_MIN_CONFIDENCE:
        return False
    if entropy > Config.PEST_MAX_ENTROPY:
        return False
    if Config.PEST_STABILITY_CHECK and not stable:
        return False
    return True


def generate_treatment_report(class_name, confidence=None, entropy=None, stable=True):
    return build_report(class_name, confidence, entropy, stable)


def build_report(class_name, confidence=None, entropy=None, stable=True):
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
        stable
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


def predict_pest(image_path):
    try:
        model, classes, device = load_pest_model()

        image = load_leaf_image(image_path)
        variants = build_stability_variants(image) if Config.PEST_STABILITY_CHECK else [
            build_pest_input(image)
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
        return True, report, None

    except Exception as e:
        return False, None, str(e)
