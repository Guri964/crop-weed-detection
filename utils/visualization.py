import cv2
import os

def save_annotated_image(image_array, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image_array)
    return output_path