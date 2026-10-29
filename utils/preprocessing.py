import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

from config import Config

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def load_leaf_image(image_path):
    """Read an image as a BGR array, honouring EXIF orientation and forcing 3 channels."""
    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.array(pil_image)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_pest_input(bgr_image):
    """Mirror the training recipe: BGR -> HSV, ToTensor, resize to 224, ImageNet normalisation."""
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((Config.PEST_IMAGE_SIZE, Config.PEST_IMAGE_SIZE), antialias=True),
        NORMALIZE
    ])

    return transform(hsv_image)


def preprocess_pest_image(image_path):
    return build_pest_input(load_leaf_image(image_path)).unsqueeze(0)


def build_stability_variants(bgr_image):
    """Original, horizontally flipped and centre-cropped views used for a stability check."""
    height, width = bgr_image.shape[:2]
    margin_y, margin_x = int(0.08 * height), int(0.08 * width)
    centre_crop = bgr_image[margin_y:height - margin_y, margin_x:width - margin_x]

    return [
        build_pest_input(bgr_image),
        build_pest_input(cv2.flip(bgr_image, 1)),
        build_pest_input(centre_crop)
    ]
