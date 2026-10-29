import io

from conftest import pest_fixture


def upload(client, url, path):
    with open(path, "rb") as handle:
        return client.post(
            url,
            data={"file": (io.BytesIO(handle.read()), path.split("/")[-1])},
            content_type="multipart/form-data"
        )


def test_public_pages_render(client):
    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 200
    assert client.get("/signup").status_code == 200


def test_dashboard_requires_login(client):
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_signup_login_logout_flow(app):
    client = app.test_client()

    client.post("/signup", data={"username": "route-user", "password": "StrongPass1!"})
    login = client.post(
        "/login", data={"username": "route-user", "password": "StrongPass1!"}
    )

    assert login.status_code == 302
    assert client.get("/dashboard").status_code == 200
    assert client.get("/logout").status_code == 302


def test_module_pages_require_login(client):
    assert client.get("/pest/").status_code == 302
    assert client.get("/weed/").status_code == 302


def test_pest_page_renders_treatment_and_crop(authenticated_client):
    response = upload(authenticated_client, "/pest/", pest_fixture("tomato_early_blight"))
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Crop Identified" in body
    assert "Tomato" in body
    assert "Early blight" in body
    assert "Prevention" in body


def test_pest_page_healthy_result_is_not_a_threat(authenticated_client):
    response = upload(authenticated_client, "/pest/", pest_fixture("tomato_healthy"))
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "No Disease Detected" in body
    assert "Attention Needed" not in body


def test_pest_upload_rejects_missing_file(authenticated_client):
    response = authenticated_client.post("/pest/", data={}, content_type="multipart/form-data")
    assert response.status_code == 302


def test_weed_page_reports_counts(authenticated_client):
    response = upload(authenticated_client, "/weed/", pest_fixture("corn_healthy"))
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Weed plants:" in body
    assert "Crop plants:" in body


def test_pest_api_contract(authenticated_client):
    response = upload(authenticated_client, "/api/pest/predict", pest_fixture("grape_black_rot"))
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["image_file"]
    assert payload["data"]["crop"] == "Grape"
    assert payload["data"]["status"] == "disease"
    assert payload["data"]["treatment"]
    assert payload["data"]["confidence"] > 0


def test_weed_api_contract(authenticated_client):
    response = upload(authenticated_client, "/api/weed/predict", pest_fixture("corn_healthy"))
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["annotated_image_file"]
    assert payload["data"]["state"] in {
        "weed_detected", "no_weed_detected", "no_detection"
    }
    assert payload["data"]["weed_count"] >= 0


def test_api_rejects_missing_file(authenticated_client):
    response = authenticated_client.post("/api/pest/predict", data={})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_api_rejects_unsupported_extension(authenticated_client):
    response = authenticated_client.post(
        "/api/weed/predict",
        data={"file": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 400


def test_served_outputs_are_reachable(authenticated_client):
    upload(authenticated_client, "/weed/", pest_fixture("corn_healthy"))
    payload = upload(
        authenticated_client, "/api/weed/predict", pest_fixture("corn_healthy")
    ).get_json()

    image_response = authenticated_client.get(
        f"/api/outputs/weed/{payload['annotated_image_file']}"
    )
    assert image_response.status_code == 200
    image_response.close()
