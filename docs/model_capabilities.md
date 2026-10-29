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

## Secondary leaf classifier (used only when the primary result is rejected)

- Artifacts: `models/efficient_model/efficientnet_plant.pth` (44.6 MB state dict) and
  `models/efficient_model/classes.json` (15 labels, original training order).
- Architecture recovered from the checkpoint and verified with a strict state-dict load
  (576/576 keys, 0 missing, 0 unexpected): `torchvision` EfficientNet-B3 backbone, a
  squeeze-excitation gate on the 1536-channel feature map, then the standard
  EfficientNet classifier head rebuilt for 15 outputs.
- Preprocessing verified empirically (90 labelled PlantVillage images across all 15 classes,
  6 per class):

  | input | accuracy |
  | --- | --- |
  | RGB, resize 300, ImageNet mean/std | 90/90 |
  | RGB, resize 224 / 260 / 320 / 384 | 90/90 |
  | RGB, resize 160 | 83/90 |
  | RGB, resize 512 | 88/90 |
  | HSV, resize 300 | 31/90 |
  | RGB, resize 300, no normalisation | 81/90 |

  The model therefore needs RGB (not the HSV pipeline of the primary model), square resize
  (300 used by default) and ImageNet normalisation. Both models keep separate preprocessing.
- Own acceptance policy (`Config.FALLBACK_*`): confidence >= 0.70, entropy <= 0.90 and a stable
  class across original / horizontal-flip / centre-crop views. These are not the primary model's
  thresholds.
- Crop eligibility gate: the secondary model only returns 15 solanaceous classes, so a prediction
  is only accepted when the **primary** model independently assigns at least 0.30 probability mass
  to the same crop family. Measured separation: 90/90 candidate images had primary family mass
  >= 0.998, while 57/57 images of Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Raspberry,
  Soybean, Squash and Strawberry had primary family mass 0.000.
- End-to-end measurement with the primary gate forced to reject every image (so the secondary path
  is exercised on all images): 90/90 correct crop and condition, and 57/57 non-solanaceous images
  correctly returned as inconclusive.

### Secondary classifier limitations

- It is trained on PlantVillage-style single-leaf images. On ordinary field photographs
  (background clutter, whole plants, several leaves) its confidence and view-stability drop, and
  the eligibility gate will then return an inconclusive result instead of a diagnosis.
- The gate is deliberately conservative: when the primary model does not corroborate the crop
  family (e.g. garden photos where the primary model lands on an unrelated species), a correct
  secondary prediction can be withheld. This is intentional - a wrong crop/disease claim is worse
  than an inconclusive result - and it means a real-world pepper/potato/tomato photo can still
  produce "Inconclusive".
- Real-world accuracy of the combined pipeline has **not** been validated on a labelled
  real-world dataset; only PlantVillage-derived images were used for the measurements above.

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
