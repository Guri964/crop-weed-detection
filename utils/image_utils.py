import os
from werkzeug.utils import secure_filename
from utils.helpers import generate_unique_id, get_file_extension

def save_uploaded_file(file, upload_folder):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder, exist_ok=True)
        
    original_filename = secure_filename(file.filename)
    ext = get_file_extension(original_filename)
    if not ext:
        ext = 'jpg'
        
    unique_filename = f"{generate_unique_id()}.{ext}"
    filepath = os.path.join(upload_folder, unique_filename)
    
    file.save(filepath)
    return filepath, unique_filename