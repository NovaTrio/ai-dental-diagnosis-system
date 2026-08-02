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
from api.schemas import LesionResponse, ScaleSelectionRequest, ToothSelectionRequest
from api.services.common_service import (
    PipelineError,
    extract_selected_roi,
    preprocess_uploaded_image,
    save_scale_selection,
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

    def test_scale_selection_calculates_mm_per_pixel(self) -> None:
        from api.services import common_service
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary_directory:
            original_case_root = common_service.CASE_ROOT
            common_service.CASE_ROOT = Path(temporary_directory)
            try:
                case_dir = common_service.CASE_ROOT / "scale-test"
                case_dir.mkdir()
                cv2.imwrite(
                    str(case_dir / "selection_image.png"),
                    np.zeros((256, 256), dtype=np.uint8),
                )
                cv2.imwrite(
                    str(case_dir / "original.png"),
                    np.zeros((256, 256), dtype=np.uint8),
                )
                result = save_scale_selection(
                    "scale-test",
                    ScaleSelectionRequest(
                        start_x=20,
                        start_y=40,
                        end_x=120,
                        end_y=40,
                        known_length_mm=10,
                    ),
                )
                self.assertAlmostEqual(result["scale_length_px"], 100.0)
                self.assertAlmostEqual(result["raw_scale_length_px"], 100.0)
                self.assertAlmostEqual(result["mm_per_pixel"], 0.1)
                self.assertTrue((case_dir / "scale_calibration.json").is_file())
            finally:
                common_service.CASE_ROOT = original_case_root

    def test_scale_selection_rejects_identical_points(self) -> None:
        from api.services import common_service
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary_directory:
            original_case_root = common_service.CASE_ROOT
            common_service.CASE_ROOT = Path(temporary_directory)
            try:
                case_dir = common_service.CASE_ROOT / "scale-test"
                case_dir.mkdir()
                cv2.imwrite(
                    str(case_dir / "selection_image.png"),
                    np.zeros((256, 256), dtype=np.uint8),
                )
                cv2.imwrite(
                    str(case_dir / "original.png"),
                    np.zeros((256, 256), dtype=np.uint8),
                )
                with self.assertRaises(PipelineError):
                    save_scale_selection(
                        "scale-test",
                        ScaleSelectionRequest(
                            start_x=20,
                            start_y=40,
                            end_x=20,
                            end_y=40,
                        ),
                    )
            finally:
                common_service.CASE_ROOT = original_case_root

    def test_lesion_response_exposes_major_axis_as_lesion_diameter(self) -> None:
        response = LesionResponse.model_validate(
            {
                "case_id": "case",
                "status": "completed",
                "lesion_detected": True,
                "mm_per_pixel": 0.0351,
                "lesion_diameter_px": 381.1,
                "lesion_diameter_mm": 13.38,
                "major_diameter_px": 381.1,
                "major_diameter_mm": 13.38,
                "artifacts": {
                    "preprocessed_tooth_roi": "/preprocessed.png",
                    "periapical_crop": "/crop.png",
                    "tooth_mask": "/tooth.png",
                    "lesion_mask": "/lesion.png",
                    "lesion_overlay": "/overlay.png",
                },
                "warnings": [],
            }
        )
        self.assertEqual(response.lesion_diameter_mm, 13.38)
        self.assertEqual(response.lesion_diameter_mm, response.major_diameter_mm)


if __name__ == "__main__":
    unittest.main()
