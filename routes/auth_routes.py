from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from services.auth_service import register_user, authenticate_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        success, error_message = register_user(username, password)
        
        if success:
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for('auth.login'))
            
        flash(error_message, "error")
        
    return render_template('signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user, error_message = authenticate_user(username, password)
        
        if user:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
            
        flash(error_message, "error")
        
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('auth.login'))