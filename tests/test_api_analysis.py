"""API smoke tests: auth flow + patch analysis + case listing.

Builds an isolated FastAPI app (only the routers under test) backed by a
temp SQLite DB, with the classifier/GradCAM explainer replaced by fakes —
so this suite never needs the real model and runs in well under a second.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.api import db as db_module
from src.api.deps import settings_dep
from src.api.routers import analysis, auth, cases, jobs, reports
from src.inference.classifier import ClassificationResult
from src.utils.config import PathsConfig


class _FakeClassifier:
    def __init__(self, class_names: list[str], malignant_name: str):
        self.class_names = class_names
        self.malignant_name = malignant_name

    def classify(self, image) -> ClassificationResult:
        probs = {n: (0.87 if n == self.malignant_name else 0.13) for n in self.class_names}
        return ClassificationResult(
            predicted_class=self.class_names.index(self.malignant_name),
            predicted_label=self.malignant_name,
            confidence=0.87,
            probabilities=probs,
        )


class _FakeExplainer:
    def explain(self, image, target_class):
        size = 224
        cam = np.zeros((size, size), dtype=np.float32)
        cam[50:70, 50:70] = 0.9
        overlay = np.zeros((size, size, 3), dtype=np.uint8)
        return cam, overlay


def _build_test_app(tmp_path, base_settings, monkeypatch):
    paths = PathsConfig(
        data_dir=str(tmp_path),
        uploads_dir=str(tmp_path / "uploads"),
        heatmaps_dir=str(tmp_path / "heatmaps"),
        reports_dir=str(tmp_path / "reports"),
        db_path=str(tmp_path / "db" / "test.sqlite3"),
        log_dir=str(tmp_path / "logs"),
    )
    test_settings = base_settings.model_copy(update={"paths": paths})
    db_module.init_db(test_settings)

    malignant_name = test_settings.model.class_names[test_settings.model.malignant_class_index]
    monkeypatch.setattr(
        analysis, "get_classifier",
        lambda: _FakeClassifier(test_settings.model.class_names, malignant_name),
    )
    monkeypatch.setattr(analysis, "get_explainer", lambda: _FakeExplainer())

    app = FastAPI()
    for router in (auth.router, analysis.router, jobs.router, cases.router, reports.router):
        app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[settings_dep] = lambda: test_settings
    return app, test_settings


def _png_bytes() -> bytes:
    image = Image.fromarray(np.full((224, 224, 3), 200, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def _login(client: TestClient, test_settings) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": test_settings.auth.seed_user.email,
            "password": test_settings.auth.seed_user.password,
        },
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_login_then_analyze_patch_then_list_cases(tmp_path, settings, monkeypatch):
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    analyze_response = client.post(
        "/api/v1/analyze/patch",
        files={"file": ("patch.png", _png_bytes(), "image/png")},
        headers=headers,
    )
    assert analyze_response.status_code == 200
    body = analyze_response.json()
    assert body["is_malignant"] is True
    assert body["confidence"] == pytest.approx(0.87)
    assert len(body["top_regions"]) == 1

    cases_response = client.get("/api/v1/cases", headers=headers)
    assert cases_response.status_code == 200
    cases_list = cases_response.json()
    assert len(cases_list) == 1
    assert cases_list[0]["id"] == body["case_id"]

    detail_response = client.get(f"/api/v1/cases/{body['case_id']}", headers=headers)
    assert detail_response.status_code == 200
    assert detail_response.json()["analysis_mode"] == "patch"


def test_analyze_patch_requires_auth(tmp_path, settings, monkeypatch):
    app, _ = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze/patch", files={"file": ("patch.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 401


def test_analyze_patch_rejects_invalid_image(tmp_path, settings, monkeypatch):
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    response = client.post(
        "/api/v1/analyze/patch",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
        headers=headers,
    )
    assert response.status_code == 400


def test_case_status_update(tmp_path, settings, monkeypatch):
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    case_id = client.post(
        "/api/v1/analyze/patch",
        files={"file": ("patch.png", _png_bytes(), "image/png")},
        headers=headers,
    ).json()["case_id"]

    update_response = client.patch(
        f"/api/v1/cases/{case_id}", json={"status": "confirmed"}, headers=headers
    )
    assert update_response.status_code == 200
    assert update_response.json()["status"] == "confirmed"
