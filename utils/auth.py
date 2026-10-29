import functools
from flask import session, redirect, url_for, flash

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if session.get('user_id') is None:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view