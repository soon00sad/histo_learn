"""API smoke tests: auth flow + patch segmentation + case listing.

Builds an isolated FastAPI app (only the routers under test) backed by a
temp SQLite DB, with the segmentation model replaced by a fake — so this
suite never needs the real trained checkpoint (models/segmentation.pth,
not committed to git) and runs in well under a second.
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
from src.inference.segmenter import SegmentationResult
from src.utils.bcss_classes import load_bcss_classes
from src.utils.config import PathsConfig

_PATCH_SIZE = 256


class _FakeSegmenter:
    """Returns a mask with a clear tumor block (~15% of the patch, well
    above the malignant_area_threshold) on a stroma background, so both the
    verdict and the class-area breakdown have something real to assert on."""

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    def segment(self, image) -> SegmentationResult:
        tumor_idx = next(c.model_index for c in self.taxonomy.classes if c.name_en == "tumor")
        stroma_idx = next(c.model_index for c in self.taxonomy.classes if c.name_en == "stroma")
        mask = np.full((_PATCH_SIZE, _PATCH_SIZE), stroma_idx, dtype=np.uint8)
        mask[50:150, 50:150] = tumor_idx

        total = mask.size
        fractions = {
            "tumor": float((mask == tumor_idx).sum()) / total,
            "stroma": float((mask == stroma_idx).sum()) / total,
        }
        return SegmentationResult(mask=mask, class_pixel_fractions=fractions, mean_confidence=0.9)


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

    taxonomy = load_bcss_classes()
    monkeypatch.setattr(analysis, "get_segmenter", lambda: _FakeSegmenter(taxonomy))

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
    tumor_fraction = 10000 / (_PATCH_SIZE * _PATCH_SIZE)
    assert body["is_malignant"] is True
    assert body["tumor_area_fraction"] == pytest.approx(tumor_fraction, abs=1e-4)
    # sorted by fraction descending — stroma (~85%) covers more area than the tumor block
    assert [c["name_en"] for c in body["class_areas"]] == ["stroma", "tumor"]

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


def test_case_review_agree_confirms_and_disagree_rejects_with_a_record(tmp_path, settings, monkeypatch):
    """Review is a one-time decision (see the next test), so agree/disagree
    are exercised on two separate cases here rather than the same one."""
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    def _new_case_id() -> str:
        return client.post(
            "/api/v1/analyze/patch",
            files={"file": ("patch.png", _png_bytes(), "image/png")},
            headers=headers,
        ).json()["case_id"]

    agree_case_id = _new_case_id()
    agree_response = client.post(
        f"/api/v1/cases/{agree_case_id}/review", json={"agreed": True}, headers=headers
    )
    assert agree_response.status_code == 200
    assert agree_response.json()["status"] == "confirmed"

    disagree_case_id = _new_case_id()
    disagree_response = client.post(
        f"/api/v1/cases/{disagree_case_id}/review",
        json={"agreed": False, "comment": "Недостаточно ткани", "corrected_verdict_label": "Доброкачественная"},
        headers=headers,
    )
    assert disagree_response.status_code == 200
    assert disagree_response.json()["status"] == "rejected"

    with db_module.session_scope() as session:
        agree_review = session.query(db_module.CaseReview).filter_by(case_id=agree_case_id).one()
        disagree_review = session.query(db_module.CaseReview).filter_by(case_id=disagree_case_id).one()
    assert agree_review.agreed is True and agree_review.corrected_verdict_label is None
    assert disagree_review.agreed is False
    assert disagree_review.comment == "Недостаточно ткани"
    assert disagree_review.corrected_verdict_label == "Доброкачественная"
    assert disagree_review.system_verdict_label == disagree_response.json()["verdict_label"]


def test_case_review_is_one_time_only(tmp_path, settings, monkeypatch):
    """Once a case has been reviewed (agreed or disagreed), a second review
    attempt must be rejected, not silently overwrite the first decision —
    the doctor's call should be a single, final action."""
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    case_id = client.post(
        "/api/v1/analyze/patch",
        files={"file": ("patch.png", _png_bytes(), "image/png")},
        headers=headers,
    ).json()["case_id"]

    first = client.post(f"/api/v1/cases/{case_id}/review", json={"agreed": True}, headers=headers)
    assert first.status_code == 200

    second = client.post(f"/api/v1/cases/{case_id}/review", json={"agreed": False}, headers=headers)
    assert second.status_code == 409

    # The original decision must be untouched, and only one review recorded.
    detail = client.get(f"/api/v1/cases/{case_id}", headers=headers).json()
    assert detail["status"] == "confirmed"
    with db_module.session_scope() as session:
        reviews = session.query(db_module.CaseReview).filter_by(case_id=case_id).all()
    assert len(reviews) == 1


def test_list_cases_default_priority_sort_ranks_malignant_first(tmp_path, settings, monkeypatch):
    """Regardless of upload order, a malignant case should rank above a
    benign one in the default (priority) sort — this is the worklist
    ordering doctors see, not just recency."""
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    taxonomy = load_bcss_classes()

    class _BenignSegmenter(_FakeSegmenter):
        def segment(self, image):
            stroma_idx = next(c.model_index for c in self.taxonomy.classes if c.name_en == "stroma")
            mask = np.full((_PATCH_SIZE, _PATCH_SIZE), stroma_idx, dtype=np.uint8)
            return SegmentationResult(mask=mask, class_pixel_fractions={"stroma": 1.0}, mean_confidence=0.9)

    # Upload the malignant case FIRST and benign SECOND, so a plain recency
    # sort would put the benign one on top — the opposite of what priority
    # sort should do, making the two behaviors actually distinguishable.
    monkeypatch.setattr(analysis, "get_segmenter", lambda: _FakeSegmenter(taxonomy))
    malignant_id = client.post(
        "/api/v1/analyze/patch", files={"file": ("patch.png", _png_bytes(), "image/png")}, headers=headers
    ).json()["case_id"]

    monkeypatch.setattr(analysis, "get_segmenter", lambda: _BenignSegmenter(taxonomy))
    benign_id = client.post(
        "/api/v1/analyze/patch", files={"file": ("patch.png", _png_bytes(), "image/png")}, headers=headers
    ).json()["case_id"]

    response = client.get("/api/v1/cases", headers=headers)  # default sort=priority
    ids_in_order = [c["id"] for c in response.json()]
    assert ids_in_order == [malignant_id, benign_id]

    date_response = client.get("/api/v1/cases", params={"sort": "date"}, headers=headers)
    date_ids_in_order = [c["id"] for c in date_response.json()]
    assert date_ids_in_order == [benign_id, malignant_id]  # benign was uploaded more recently


def test_case_mask_is_a_lossless_png_not_a_jpeg_heatmap(tmp_path, settings, monkeypatch):
    """The segmentation path stores/serves a lossless indexed PNG (exact
    per-pixel class IDs) at /mask, replacing the old JPEG /heatmap
    endpoint — see case_service.save_mask_png."""
    app, test_settings = _build_test_app(tmp_path, settings, monkeypatch)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_login(client, test_settings)}"}

    case_id = client.post(
        "/api/v1/analyze/patch",
        files={"file": ("patch.png", _png_bytes(), "image/png")},
        headers=headers,
    ).json()["case_id"]

    mask_response = client.get(f"/api/v1/cases/{case_id}/mask", headers=headers)
    assert mask_response.status_code == 200
    assert mask_response.headers["content-type"] == "image/png"

    loaded = Image.open(io.BytesIO(mask_response.content))
    assert loaded.mode == "P"
    assert loaded.size == (_PATCH_SIZE, _PATCH_SIZE)
