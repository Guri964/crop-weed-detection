import os
import json
import torch
from torchvision.models.mobilenetv2 import MobileNetV2

from config import Config
from utils.preprocessing import preprocess_pest_image

_pest_model = None
_pest_classes = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pest_model():
    global _pest_model, _pest_classes

    if _pest_model is None:
        torch.serialization.add_safe_globals([MobileNetV2])

        _pest_model = torch.load(
            Config.PEST_MODEL_PATH,
            map_location=_device,
            weights_only=False
        )

        _pest_model.to(_device)
        _pest_model.eval()

    if _pest_classes is None:
        if os.path.exists(Config.PEST_CLASSES_PATH):
            with open(Config.PEST_CLASSES_PATH, "r", encoding="utf-8") as f:
                _pest_classes = json.load(f)
        else:
            _pest_classes = [str(i) for i in range(38)]

    return _pest_model, _pest_classes, _device


def generate_treatment_report(class_name):
    class_name_lower = class_name.lower()

    if "healthy" in class_name_lower:
        return {
            "disease_name": class_name,
            "root_cause": "Optimal conditions. No visible pest or disease.",
            "treatment": "Maintain current watering and fertilizer schedule."
        }

    return {
        "disease_name": class_name,
        "root_cause": "Environmental stress, fungal/bacterial infection, or active pest infestation.",
        "treatment": "Isolate affected plants. Remove infected leaves and apply appropriate organic or chemical fungicides/pesticides based on local agricultural guidelines."
    }


def predict_pest(image_path):
    try:
        model, classes, device = load_pest_model()

        img_tensor = preprocess_pest_image(image_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        class_idx = predicted.item()

        if class_idx < len(classes):
            class_name = classes[class_idx]
        else:
            class_name = f"Class_{class_idx}"

        report = generate_treatment_report(class_name)

        return True, report, None

    except Exception as e:
        return False, None, str(e)