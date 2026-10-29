import os
from ultralytics import YOLO
from config import Config
from utils.visualization import save_annotated_image

_weed_model = None

def load_weed_model():
    global _weed_model
    if _weed_model is None:
        _weed_model = YOLO(Config.WEED_MODEL_PATH)
    return _weed_model

def predict_weed(image_path):
    try:
        model = load_weed_model()
        
        results = model.predict(source=image_path, imgsz=640, conf=0.25, save=False)
        
        if not results:
            return False, None, "No detections generated."
            
        result = results[0]
        annotated_image_array = result.plot()
        
        filename = f"predicted_{os.path.basename(image_path)}"
        output_path = save_annotated_image(
            annotated_image_array, 
            Config.OUTPUT_WEED_DIR, 
            filename
        )
        
        return True, output_path, None
        
    except Exception as e:
        return False, None, str(e)