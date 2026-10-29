import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import app as flask_app

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
PEST_FIXTURES = FIXTURE_DIR / "pest"


@pytest.fixture(scope="session")
def app():
    flask_app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def authenticated_client(app, tmp_path):
    client = app.test_client()
    username = "tester"
    password = "StrongPass1!"

    client.post("/signup", data={"username": username, "password": password})
    client.post("/login", data={"username": username, "password": password})
    return client


def pest_fixture(name):
    return str(PEST_FIXTURES / f"{name}.jpg")
