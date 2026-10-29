import os

import cv2
from ultralytics import YOLO

from config import Config
from utils.visualization import draw_weed_detections, save_annotated_image

STATE_WEED_DETECTED = "weed_detected"
STATE_NO_WEED = "no_weed_detected"
STATE_NO_DETECTION = "no_detection"
STATE_UNSUPPORTED = "unsupported"

WEED_LABEL_KEYWORDS = ("weed",)
CROP_LABEL_KEYWORDS = ("crop", "plant", "maize", "soybean", "rice", "wheat")

SCOPE_NOTE = (
    "This analysis covers only the submitted image. A single frame cannot confirm that a "
    "whole field is weed-free."
)

_weed_model = None


def load_weed_model():
    global _weed_model
    if _weed_model is None:
        _weed_model = YOLO(Config.WEED_MODEL_PATH)
    return _weed_model


def classify_class_name(name):
    lowered = str(name).strip().lower()

    if any(keyword in lowered for keyword in WEED_LABEL_KEYWORDS):
        return "weed"
    if any(keyword in lowered for keyword in CROP_LABEL_KEYWORDS):
        return "crop"
    return "other"


def build_class_kinds(names):
    return {int(class_id): classify_class_name(name) for class_id, name in names.items()}


def extract_detections(result, names):
    detections = []
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        class_id = int(box.cls.item())
        label = names.get(class_id, str(class_id))
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        detections.append({
            "class_id": class_id,
            "label": label,
            "kind": classify_class_name(label),
            "confidence": round(float(box.conf.item()), 4),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    return detections


def build_weed_report(detections, threshold):
    weeds = [d for d in detections if d["kind"] == "weed"]
    crops = [d for d in detections if d["kind"] == "crop"]

    if weeds:
        state = STATE_WEED_DETECTED
        summary = (
            f"{len(weeds)} weed plant{'s' if len(weeds) != 1 else ''} detected in this image."
        )
    elif crops:
        state = STATE_NO_WEED
        summary = (
            "No weeds were detected in this image; the model only found crop plants."
        )
    else:
        state = STATE_NO_DETECTION
        summary = (
            "No crop or weed objects were detected in this image, so no weed assessment can be "
            "made from it."
        )

    return {
        "state": state,
        "weed_count": len(weeds),
        "crop_count": len(crops),
        "detections": weeds,
        "summary": summary,
        "scope_note": SCOPE_NOTE,
        "threshold": threshold,
    }


def predict_weed(image_path):
    try:
        model = load_weed_model()
        names = {int(key): value for key, value in model.names.items()}
        kinds = build_class_kinds(names)

        if "weed" not in kinds.values():
            return False, None, "The detection model does not expose a weed class."

        results = model.predict(
            source=image_path,
            imgsz=Config.WEED_IMAGE_SIZE,
            conf=Config.WEED_CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False
        )

        if not results:
            return False, None, "No detections generated."

        result = results[0]
        detections = extract_detections(result, names)
        report = build_weed_report(detections, Config.WEED_CONFIDENCE_THRESHOLD)

        original_image = result.orig_img if result.orig_img is not None else cv2.imread(image_path)
        annotated_image = draw_weed_detections(original_image, report["detections"])

        filename = f"predicted_{os.path.basename(image_path)}"
        output_path = save_annotated_image(
            annotated_image,
            Config.OUTPUT_WEED_DIR,
            filename
        )

        report["output_path"] = output_path
        return True, report, None

    except Exception as e:
        return False, None, str(e)
