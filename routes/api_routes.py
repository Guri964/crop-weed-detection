import os
from flask import Blueprint, request, jsonify, send_from_directory
from services.image_service import process_image_upload
from services.pest_service import predict_pest
from services.weed_service import predict_weed
from config import Config

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/pest/predict', methods=['POST'])
def api_predict_pest():
    success, error_msg, filepath = process_image_upload(request, Config.UPLOAD_PEST_DIR)
    
    if not success:
        return jsonify({"status": "error", "message": error_msg}), 400
        
    predict_success, report, predict_error = predict_pest(filepath)
    
    if not predict_success:
        return jsonify({"status": "error", "message": predict_error}), 500
        
    return jsonify({
        "status": "success",
        "data": report,
        "image_file": os.path.basename(filepath)
    })

@api_bp.route('/weed/predict', methods=['POST'])
def api_predict_weed():
    success, error_msg, filepath = process_image_upload(request, Config.UPLOAD_WEED_DIR)
    
    if not success:
        return jsonify({"status": "error", "message": error_msg}), 400
        
    predict_success, output_path, predict_error = predict_weed(filepath)
    
    if not predict_success:
        return jsonify({"status": "error", "message": predict_error}), 500
        
    return jsonify({
        "status": "success",
        "annotated_image_file": os.path.basename(output_path)
    })

@api_bp.route('/uploads/pest/<filename>')
def serve_pest_upload(filename):
    return send_from_directory(Config.UPLOAD_PEST_DIR, filename)

@api_bp.route('/outputs/weed/<filename>')
def serve_weed_output(filename):
    return send_from_directory(Config.OUTPUT_WEED_DIR, filename)