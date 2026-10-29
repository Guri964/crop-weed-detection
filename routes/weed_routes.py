import os
from flask import Blueprint, render_template, request, flash, redirect
from utils.auth import login_required
from services.image_service import process_image_upload
from services.weed_service import predict_weed
from config import Config

weed_bp = Blueprint('weed', __name__, url_prefix='/weed')

@weed_bp.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        success, error_msg, filepath = process_image_upload(request, Config.UPLOAD_WEED_DIR)
        
        if not success:
            flash(error_msg, "error")
            return redirect(request.url)
            
        predict_success, report, predict_error = predict_weed(filepath)
        
        if not predict_success:
            flash(f"Prediction failed: {predict_error}", "error")
            return redirect(request.url)
            
        filename = os.path.basename(report["output_path"])
        return render_template('result.html', module="weed", report=report, filename=filename)
        
    return render_template('weed.html')