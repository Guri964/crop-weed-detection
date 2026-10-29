import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-crop-risk'
    
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'crop_risk.db')
    SCHEMA_PATH = os.path.join(BASE_DIR, 'database', 'schema.sql')
    
    UPLOAD_PEST_DIR = os.path.join(BASE_DIR, 'uploads', 'pest')
    UPLOAD_WEED_DIR = os.path.join(BASE_DIR, 'uploads', 'weed')
    OUTPUT_PEST_DIR = os.path.join(BASE_DIR, 'outputs', 'pest_reports')
    OUTPUT_WEED_DIR = os.path.join(BASE_DIR, 'outputs', 'weed_predictions')
    
    PEST_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'pest_classifier')
    PEST_MODEL_PATH = os.path.join(PEST_MODEL_DIR, 'best_model.pth')
    PEST_CLASSES_PATH = os.path.join(PEST_MODEL_DIR, 'classes.json')
    
    WEED_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'weed_detector')
    WEED_MODEL_PATH = os.path.join(WEED_MODEL_DIR, 'best.pt')
    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_PEST_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_WEED_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_PEST_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_WEED_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.BASE_DIR, 'database'), exist_ok=True)