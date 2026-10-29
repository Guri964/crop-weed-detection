import os
from flask import Flask, render_template, g, session
from dotenv import load_dotenv

load_dotenv()

from config import Config
from database import init_db, close_db
from services.auth_service import get_user_by_id
from utils.auth import login_required

from routes.auth_routes import auth_bp
from routes.pest_routes import pest_bp
from routes.weed_routes import weed_bp
from routes.api_routes import api_bp

app = Flask(__name__)
app.config.from_object(Config)

Config.init_app(app)

with app.app_context():
    try:
        init_db()
    except Exception as e:
        print(f"Database initialization warning: {e}")

app.teardown_appcontext(close_db)

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = get_user_by_id(user_id)

app.register_blueprint(auth_bp)
app.register_blueprint(pest_bp)
app.register_blueprint(weed_bp)
app.register_blueprint(api_bp)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)