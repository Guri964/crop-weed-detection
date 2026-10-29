import json

import pytest

from config import Config
from services.pest_knowledge import (
    TREATMENTS,
    get_condition_display_name,
    get_crop_display_name,
    is_healthy_class,
    is_supported_class,
    split_class_label,
)


@pytest.fixture(scope="module")
def class_names():
    with open(Config.PEST_CLASSES_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_every_model_class_has_crop_and_condition(class_names):
    for class_name in class_names:
        assert get_crop_display_name(class_name)
        assert get_condition_display_name(class_name)


def test_every_model_class_has_recommendations(class_names):
    for class_name in class_names:
        assert is_supported_class(class_name)
        entry = TREATMENTS[class_name]
        assert entry["root_cause"].strip()
        assert entry["treatment"].strip()
        assert entry["prevention"].strip()


def test_model_covers_fourteen_crops(class_names):
    crops = {split_class_label(name)[0] for name in class_names}
    assert len(crops) == 14
    assert len(class_names) == 38


def test_healthy_classes_are_detected(class_names):
    healthy = [name for name in class_names if is_healthy_class(name)]
    assert len(healthy) == 12
    assert "Tomato___healthy" in healthy
    assert not is_healthy_class("Tomato___Early_blight")


def test_disease_recommendations_are_distinct(class_names):
    disease_entries = {
        name: TREATMENTS[name] for name in class_names if not is_healthy_class(name)
    }
    treatments = [entry["treatment"] for entry in disease_entries.values()]
    root_causes = [entry["root_cause"] for entry in disease_entries.values()]

    assert len(set(treatments)) == len(treatments)
    assert len(set(root_causes)) == len(root_causes)


def test_related_diseases_differ_between_crops(class_names):
    tomato_early = TREATMENTS["Tomato___Early_blight"]
    potato_early = TREATMENTS["Potato___Early_blight"]
    tomato_late = TREATMENTS["Tomato___Late_blight"]
    potato_late = TREATMENTS["Potato___Late_blight"]

    assert tomato_early["treatment"] != potato_early["treatment"]
    assert tomato_late["treatment"] != potato_late["treatment"]
    assert tomato_early["treatment"] != tomato_late["treatment"]


def test_recommendations_avoid_unverified_dosages():
    forbidden = ["ml/l", "ml per litre", "g/l", "kg/ha", "litre/ha", "per acre"]
    for entry in TREATMENTS.values():
        lowered = (entry["treatment"] + " " + entry["prevention"]).lower()
        for token in forbidden:
            assert token not in lowered


def test_unknown_label_is_unsupported():
    assert not is_supported_class("Banana___healthy")
    assert split_class_label("Banana___healthy") == ("Banana", "healthy")
    assert get_crop_display_name("Banana___healthy") is None
    assert split_class_label("ModelOutput") == (None, "ModelOutput")
