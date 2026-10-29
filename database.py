import sqlite3
from flask import g
from config import Config

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            Config.DATABASE_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    with open(Config.SCHEMA_PATH, 'r', encoding='utf-8') as f:
        db.executescript(f.read())