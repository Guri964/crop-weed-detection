import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def convert_to_hsv(image_path, save_path=None):
    """
    Reads an image, converts it to HSV color space, and optionally saves it.
    This exactly mimics the first preprocessing step used during the 
    MobileNetV2 Pest Classifier model training.
    
    Args:
        image_path (str): Path to the input RGB/BGR image.
        save_path (str, optional): Path to save the resulting HSV image.
        
    Returns:
        numpy.ndarray: The HSV image array, or None if reading fails.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None

    # Read image using OpenCV (loads in BGR format by default)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}. It might be corrupted.")
        return None

    # Convert BGR to HSV color space
    # The training notebook specifically used cv2.COLOR_BGR2HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if save_path:
        # Note: Saving an HSV array directly as an image file (like .jpg or .png) 
        # will cause standard image viewers to misinterpret the channels as BGR/RGB, 
        # leading to strange colors. This is normal and expected when saving raw HSV.
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, hsv_img)
        print(f"Success! Raw HSV image array saved to {save_path}")

    return hsv_img

def visualize_channels(image_path):
    """
    Visualizes the original RGB image alongside its separated H, S, and V channels.
    Useful for debugging and understanding what features the model is learning from.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image for visualization at {image_path}")
        return

    # Convert to RGB for proper display in matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for channel separation
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # Set up the matplotlib figure
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original (RGB)')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    # Hue represents color type. Using hsv colormap represents it accurately.
    plt.imshow(h, cmap='hsv')
    plt.title('Hue (H) Channel')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    # Saturation represents intensity/purity of color
    plt.imshow(s, cmap='Greys')
    plt.title('Saturation (S) Channel')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    # Value represents brightness
    plt.imshow(v, cmap='Greys')
    plt.title('Value (V) Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Pest Image HSV Preprocessing Utility")
    parser.add_argument("-i", "--image", required=True, help="Path to the input crop leaf image")
    parser.add_argument("-o", "--output", default=None, help="Path to save the HSV converted image (optional)")
    parser.add_argument("-v", "--visualize", action="store_true", help="Pop up a matplotlib window to visualize the HSV channels")

    args = parser.parse_args()

    # Run the core conversion
    hsv_result = convert_to_hsv(args.image, args.output)

    # Trigger visualization if requested and image was valid
    if args.visualize and hsv_result is not None:
        print("Opening visualization window...")
        visualize_channels(args.image)