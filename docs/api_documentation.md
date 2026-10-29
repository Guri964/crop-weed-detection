# API Documentation

All endpoints accept `multipart/form-data` uploads in the `file` field
(`png`, `jpg`, `jpeg`, maximum 16 MB) and require an authenticated session.

## POST /api/pest/predict

Leaf image classification with the MobileNetV2 model (38 PlantVillage classes, 14 crops).

```json
{
  "status": "success",
  "image_file": "0f1c....jpg",
  "data": {
    "status": "disease",
    "disease_name": "Tomato___Early_blight",
    "crop": "Tomato",
    "condition": "Early blight",
    "confidence": 0.9997,
    "confidence_label": "high",
    "entropy": 0.002,
    "stable": true,
    "root_cause": "...",
    "treatment": "...",
    "prevention": "...",
    "notice": "...",
    "disclaimer": "..."
  }
}
```

`data.status` is one of:

| status | meaning | crop/condition |
| --- | --- | --- |
| `disease` | confident, stable prediction of a plant disease | populated |
| `healthy` | confident, stable prediction of a healthy leaf | populated |
| `uncertain` | low confidence, high entropy or unstable across image variants | `null` |
| `unsupported` | prediction falls outside the supported label set | `null` |

For `uncertain` and `unsupported` results the report contains an explanation and
safe next steps instead of a crop, disease or treatment claim.

## POST /api/weed/predict

Crop/weed detection with the YOLOv8 model (`crop`, `weed` classes).

```json
{
  "status": "success",
  "annotated_image_file": "predicted_0f1c....jpg",
  "data": {
    "state": "weed_detected",
    "weed_count": 3,
    "crop_count": 0,
    "detections": [
      {"class_id": 1, "label": "weed", "kind": "weed", "confidence": 0.678, "box": [161.0, 0.1, 429.9, 178.1]}
    ],
    "summary": "3 weed plants detected in this image.",
    "scope_note": "...",
    "threshold": 0.25
  }
}
```

`data.state` is one of `weed_detected`, `no_weed_detected` (crop objects found
but no weeds), or `no_detection` (no crop or weed objects in the frame).
`detections` contains weed boxes only; crop detections are counted in
`crop_count` and are never drawn on the annotated image.

## Static result endpoints

- `GET /api/uploads/pest/<filename>` — the uploaded leaf image.
- `GET /api/outputs/weed/<filename>` — the annotated field image (weed boxes only).
