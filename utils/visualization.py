import cv2
import os

WEED_BOX_COLOR = (216, 78, 29)
WEED_LABEL_COLOR = (255, 255, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def save_annotated_image(image_array, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image_array)
    return output_path


def draw_weed_detections(image_array, detections):
    """Draw only the supplied detections (weed class) so crop plants stay unmarked."""
    canvas = image_array.copy()
    height, width = canvas.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), WEED_BOX_COLOR, BOX_THICKNESS)

        caption = f"{detection['label']} {detection['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, FONT, FONT_SCALE, FONT_THICKNESS
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            canvas,
            (x1, label_top),
            (x1 + text_width + 6, label_top + text_height + baseline + 4),
            WEED_BOX_COLOR,
            -1
        )
        cv2.putText(
            canvas,
            caption,
            (x1 + 3, label_top + text_height + 2),
            FONT,
            FONT_SCALE,
            WEED_LABEL_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return canvas
