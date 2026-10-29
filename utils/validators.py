from config import Config

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_image_request(request):
    if 'file' not in request.files:
        return False, "No file part in the request.", None
        
    file = request.files['file']
    
    if file.filename == '':
        return False, "No file selected for uploading.", None
        
    if not allowed_file(file.filename):
        return False, f"Allowed file types are {', '.join(Config.ALLOWED_EXTENSIONS)}.", None
        
    return True, None, file