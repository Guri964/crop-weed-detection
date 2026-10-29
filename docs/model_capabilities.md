# Model Capabilities and Known Limitations

## Pest / disease classifier

- Artifact: `models/pest_classifier/best_model.pth` (serialised `torchvision` MobileNetV2), labels in `classes.json`.
- Training recipe (see `notebooks/crop-risk-evaluation1.ipynb`): PlantVillage augmented dataset,
  BGR -> HSV conversion, `ToTensor`, resize to 224x224, ImageNet mean/std normalisation,
  38 classes, validation accuracy 99.68% on the PlantVillage validation split.
- Supported label set: 38 labels = 14 crops x (diseases + healthy leaf), listed in `classes.json`.
- The model is a **single 38-way leaf classifier**. It does not predict crops outside that label
  set and it has no "unknown" class, so out-of-distribution inputs (other species, whole-plant or
  field photos, non-plant images) still receive a softmax label.
- Because of that, inference adds a decision gate (`services/pest_service.py`): a label is only
  reported when softmax confidence >= 0.60, entropy <= 0.60 and the predicted class is identical
  for the original image, a horizontally flipped copy and a centre crop. Otherwise the result is
  reported as `uncertain` with no crop, disease or treatment claim.
- Validation observed on 114 held-out PlantVillage images (3 per class, downloaded separately):
  112 were accepted by the gate and all 112 were correct. Real-world web photos of leaves were
  frequently rejected as `uncertain`, which is the intended conservative behaviour of this model
  outside its training distribution.

## Weed / crop detector

- Artifact: `models/weed_detector/best.pt` (Ultralytics YOLOv8n, 50 epochs, `imgsz=640`).
- Class names embedded in the checkpoint: `{0: 'crop', 1: 'weed'}`.
- The detector does **not** support weed species identification, only the crop/weed split.
- Inference keeps the model's documented defaults (640 px, confidence 0.25, NMS) and filters the
  output by class name, so only `weed` boxes are drawn; `crop` boxes are counted and reported.
- On the small public "Object-detection-Dataset-Crop-Weed2" sample (6 images, 46 ground-truth
  boxes) the checkpoint found some weeds but also mislabelled several crop plants as weeds and
  missed most annotated weeds. Improving recall/precision needs retraining on the original
  dataset (the `dataset.yaml` referenced by the checkpoint is `/kaggle/working/dataset.yaml`,
  which is not shipped with this repository). No retraining was performed.
