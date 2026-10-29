from utils.validators import validate_image_request
from utils.image_utils import save_uploaded_file

def process_image_upload(request, upload_folder):
    is_valid, error_msg, file = validate_image_request(request)
    
    if not is_valid:
        return False, error_msg, None
        
    filepath, unique_filename = save_uploaded_file(file, upload_folder)
    return True, None, filepath