import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from app.main import add_configured_cors
from scripts.phase16_measure_resources import summarize_latency, tree_size
from scripts.data_paths import PROJECT_ROOT, data_path, data_root


class Phase16SetupTests(unittest.TestCase):
    def test_data_root_defaults_to_repository_data_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(data_root(), PROJECT_ROOT / "data")
            self.assertEqual(data_path("cleaned", "movies.csv"), PROJECT_ROOT / "data" / "cleaned" / "movies.csv")

    def test_data_root_can_be_configured_for_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CINEMATCH_DATA_ROOT": directory}):
                self.assertEqual(data_root(), Path(directory))
                self.assertEqual(data_path("models", "vectors.npz"), Path(directory) / "models" / "vectors.npz")

    def test_disk_measurement_skips_symlinks_and_counts_hardlinks_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "artifact.bin"
            source.write_bytes(b"12345")
            hardlink = root / "artifact-hardlink.bin"
            hardlink.hardlink_to(source)
            symlink = root / "artifact-symlink.bin"
            symlink.symlink_to(source)

            self.assertEqual(tree_size(root), 5)
            self.assertEqual(tree_size(symlink), 0)

    def test_nearest_rank_p95_uses_the_maximum_for_five_samples(self):
        summary = summarize_latency([0.001, 0.002, 0.003, 0.004, 0.005])
        self.assertEqual(summary["p95_ms_nearest_rank"], 5.0)

    def test_latency_summary_rejects_empty_samples(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            summarize_latency([])

    def test_deployment_cors_allows_only_configured_frontend_origins(self):
        app = FastAPI()

        @app.get("/api/v1/health")
        def health():
            return {"status": "ok"}

        with patch.dict(os.environ, {"CINEMATCH_CORS_ORIGINS": " https://cinematch.pages.dev/ "}):
            origins = add_configured_cors(app)

        self.assertEqual(origins, ["https://cinematch.pages.dev"])
        with TestClient(app) as client:
            allowed = client.options(
                "/api/v1/health",
                headers={
                    "Origin": "https://cinematch.pages.dev",
                    "Access-Control-Request-Method": "GET",
                },
            )
            denied = client.options(
                "/api/v1/health",
                headers={
                    "Origin": "https://other.example",
                    "Access-Control-Request-Method": "GET",
                },
            )

        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], "https://cinematch.pages.dev")
        self.assertEqual(denied.status_code, 400)
        self.assertNotIn("access-control-allow-origin", denied.headers)


if __name__ == "__main__":
    unittest.main()
