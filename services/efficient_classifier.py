"""Secondary 15-class leaf classifier (EfficientNet-B3) used when the primary
38-class model cannot produce an accepted prediction.

The architecture below is rebuilt to match the state dict shipped as
``models/efficient_model/efficientnet_plant.pth`` exactly (strict load). The
checkpoint was trained on RGB images resized to 300x300 with ImageNet
normalisation, which is the preprocessing implemented here. Do not reuse the
primary model's HSV preprocessing for this model.
"""

import json
import logging
import os

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3

logger = logging.getLogger(__name__)

from config import Config

SE_REDUCTION = 16
SUPPORTED_CROP_FAMILIES = ("Pepper", "Potato", "Tomato")

LABEL_TO_APP_CLASS = {
    "Pepper__bell___Bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "Pepper__bell___healthy": "Pepper,_bell___healthy",
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Tomato_Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Tomato_Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot": "Tomato___Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_healthy": "Tomato___healthy",
}

APP_CROP_FAMILY = {
    "Pepper,_bell": "Pepper",
    "Potato": "Potato",
    "Tomato": "Tomato",
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_classes = None


class ClassifierUnavailable(RuntimeError):
    pass


class _SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // SE_REDUCTION, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // SE_REDUCTION, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features):
        scale = self.fc(features.mean(dim=(2, 3)))
        return features * scale.view(scale.size(0), scale.size(1), 1, 1)


class EfficientPlantClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = efficientnet_b3(weights=None)
        channels = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(channels, num_classes)
        self.se = _SqueezeExcitation(channels)

    def forward(self, images):
        features = self.base_model.features(images)
        features = self.se(features)
        pooled = self.base_model.avgpool(features)
        return self.base_model.classifier(torch.flatten(pooled, 1))


def is_available():
    return (
        os.path.exists(Config.FALLBACK_MODEL_PATH)
        and os.path.exists(Config.FALLBACK_CLASSES_PATH)
    )


def load_classifier():
    global _model, _classes

    if _model is None or _classes is None:
        if not is_available():
            raise ClassifierUnavailable(
                "secondary classifier artifacts are missing "
                f"(expected {Config.FALLBACK_MODEL_DIR})"
            )

        with open(Config.FALLBACK_CLASSES_PATH, "r", encoding="utf-8") as handle:
            labels = json.load(handle)

        model = EfficientPlantClassifier(len(labels))
        state = torch.load(
            Config.FALLBACK_MODEL_PATH,
            map_location=_device,
            weights_only=True
        )
        model.load_state_dict(state, strict=True)
        model.to(_device)
        model.eval()

        _classes = labels
        _model = model

    return _model, _classes, _device


_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((Config.FALLBACK_IMAGE_SIZE, Config.FALLBACK_IMAGE_SIZE), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_input(image_bgr):
    return _preprocess(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def build_variants(image_bgr):
    height, width = image_bgr.shape[:2]
    margin_y, margin_x = int(0.08 * height), int(0.08 * width)
    centre_crop = image_bgr[margin_y:height - margin_y, margin_x:width - margin_x]

    return [
        build_input(image_bgr),
        build_input(cv2.flip(image_bgr, 1)),
        build_input(centre_crop),
    ]


def prediction_entropy(probabilities):
    return float(-(probabilities * torch.log(probabilities + 1e-12)).sum())


def app_crop_family(app_class):
    if app_class is None or "___" not in app_class:
        return None
    return APP_CROP_FAMILY.get(app_class.split("___", 1)[0])


def resolve_app_class(label):
    return LABEL_TO_APP_CLASS.get(label)


def predict(image_bgr):
    model, classes, device = load_classifier()

    variants = build_variants(image_bgr) if Config.FALLBACK_STABILITY_CHECK else [
        build_input(image_bgr)
    ]

    batch = torch.stack(variants).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)

    primary = probabilities[0]
    index = int(torch.argmax(primary).item())
    label = classes[index] if index < len(classes) else None
    top = torch.topk(primary, min(3, len(classes)))

    return {
        "label": label,
        "app_class": resolve_app_class(label),
        "crop_family": app_crop_family(resolve_app_class(label)),
        "confidence": float(primary[index].item()) if label else 0.0,
        "entropy": prediction_entropy(primary),
        "stable": len({int(torch.argmax(p).item()) for p in probabilities}) == 1,
        "top3": [
            (classes[int(i)], round(float(primary[int(i)].item()), 4))
            for i in top.indices
        ],
        "probabilities": primary.detach().cpu().numpy(),
    }
