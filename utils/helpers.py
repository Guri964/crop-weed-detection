import uuid

def generate_unique_id():
    return str(uuid.uuid4())

def get_file_extension(filename):
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''