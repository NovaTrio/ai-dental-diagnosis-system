from __future__ import annotations

import unittest

import cv2
import numpy as np

try:
    import httpx  # noqa: F401
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError):
    TestClient = None

from api.app import app
from api.schemas import ToothSelectionRequest
from api.services.common_service import (
    PipelineError,
    extract_selected_roi,
    preprocess_uploaded_image,
)


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app) if TestClient is not None else None

    @unittest.skipIf(TestClient is None, "httpx is not installed")
    def test_health_reports_models(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @unittest.skipIf(TestClient is None, "httpx is not installed")
    def test_upload_returns_selection_image(self) -> None:
        image = np.zeros((300, 200), dtype=np.uint8)
        cv2.rectangle(image, (70, 20), (130, 280), 180, -1)
        encoded, data = cv2.imencode(".png", image)
        self.assertTrue(encoded)
        response = self.client.post(
            "/api/v1/cases",
            files={"image": ("radiograph.png", data.tobytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["image_width"], 256)
        self.assertEqual(body["image_height"], 256)
        artifact = self.client.get(body["selection_image_url"])
        self.assertEqual(artifact.status_code, 200)

    def test_common_preprocessing_returns_uint8_selection_image(self) -> None:
        image = np.tile(np.arange(80, dtype=np.uint8), (120, 1))
        result = preprocess_uploaded_image(image)
        self.assertEqual(result.shape, (256, 256))
        self.assertEqual(result.dtype, np.uint8)

    def test_selection_rejects_identical_points(self) -> None:
        image = np.zeros((256, 256), dtype=np.uint8)
        selection = ToothSelectionRequest(
            selected_x=100,
            selected_y=100,
            direction_x=100,
            direction_y=100,
        )
        with self.assertRaises(PipelineError):
            extract_selected_roi(image, selection)


if __name__ == "__main__":
    unittest.main()
