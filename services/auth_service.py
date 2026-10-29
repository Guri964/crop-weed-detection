import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db

def register_user(username, password):
    if not username or not password:
        return False, "Username and password are required."
        
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        db.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already taken."
    except Exception as e:
        return False, str(e)

def authenticate_user(username, password):
    if not username or not password:
        return None, "Username and password are required."

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()

    if user is None:
        return None, "Invalid credentials."
    elif not check_password_hash(user['password'], password):
        return None, "Invalid credentials."
        
    return dict(user), None

def get_user_by_id(user_id):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return dict(user) if user else None