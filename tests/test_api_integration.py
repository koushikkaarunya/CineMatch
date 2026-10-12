"""Integration tests for CineMatch FastAPI application with live recommendation models."""

import asyncio
import unittest

from app import main
from tests.test_api import asgi_request


class CineMatchLiveApiIntegrationTests(unittest.TestCase):
    """Verify live ASGI requests against the full catalog and recommendation engine."""

    @classmethod
    def setUpClass(cls):
        cls.lifespan = main.app.router.lifespan_context(main.app)
        asyncio.run(cls.lifespan.__aenter__())

    @classmethod
    def tearDownClass(cls):
        asyncio.run(cls.lifespan.__aexit__(None, None, None))

    def request(self, method: str, url: str, body: dict | None = None):
        return asyncio.run(asgi_request(method, url, body))

    def test_live_health_endpoint(self):
        status, payload = self.request("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_live_movie_search_and_empty_results(self):
        status, result = self.request("GET", "/api/v1/movies?q=Toy%20Story&limit=3")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["movie_id"], 1)
        self.assertIn("Adventure", result["items"][0]["genres"])

        status, empty = self.request("GET", "/api/v1/movies?q=NonExistentMovieTitleXYZ&limit=5")
        self.assertEqual(status, 200)
        self.assertEqual(empty, {"items": [], "count": 0})

    def test_live_movie_details_and_sparse_metadata(self):
        status, details = self.request("GET", "/api/v1/movies/1")
        self.assertEqual(status, 200)
        self.assertEqual(details["movie_id"], 1)
        self.assertEqual(details["title"], "Toy Story (1995)")
        self.assertEqual(details["release_year"], 1995)
        self.assertTrue(len(details["directors"]) >= 1)

        # Movie 284 has missing directors in catalog
        status, sparse = self.request("GET", "/api/v1/movies/284")
        self.assertEqual(status, 200)
        self.assertEqual(sparse["movie_id"], 284)
        self.assertEqual(sparse["directors"], [])
        self.assertIsNone(sparse["plot"])

        # Unknown movie ID returns 404
        status, not_found = self.request("GET", "/api/v1/movies/999999")
        self.assertEqual(status, 404)

    def test_live_movie_recommendations(self):
        status, payload = self.request("GET", "/api/v1/movies/1/recommendations?top_k=3")
        self.assertEqual(status, 200)
        self.assertIn("items", payload)
        self.assertEqual(len(payload["items"]), 3)
        self.assertTrue(payload["content_available"])
        self.assertEqual(payload["items"][0]["movie_id"], 3114)  # Toy Story 2

    def test_live_user_recommendations(self):
        status, payload = self.request("GET", "/api/v1/users/1/recommendations?top_k=3")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "hybrid_recommendations")
        self.assertEqual(len(payload["items"]), 3)
        self.assertIn("fusion_score", payload["items"][0])

        # Unknown MovieLens user returns 404
        status, unknown = self.request("GET", "/api/v1/users/999999/recommendations?top_k=3")
        self.assertEqual(status, 404)
        self.assertEqual(unknown["detail"]["status"], "unknown_user")

    def test_live_favorite_recommendations(self):
        status, payload = self.request(
            "POST",
            "/api/v1/recommendations/from-favorites",
            body={"favorite_movie_ids": [1, 260], "top_k": 3},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "content_based_from_selected_movies")
        self.assertEqual(len(payload["items"]), 3)

        # Unknown favorite movie ID returns 404
        status, err = self.request(
            "POST",
            "/api/v1/recommendations/from-favorites",
            body={"favorite_movie_ids": [999999], "top_k": 3},
        )
        self.assertEqual(status, 404)
        self.assertIn("Unknown movie_id", err["detail"])

    def test_live_validation_errors(self):
        # Empty whitespace title query
        status, _ = self.request("GET", "/api/v1/movies?q=%20%20")
        self.assertEqual(status, 400)

        # top_k out of range
        status, _ = self.request("GET", "/api/v1/movies/1/recommendations?top_k=99")
        self.assertEqual(status, 400)

        # Invalid path parameter
        status, _ = self.request("GET", "/api/v1/movies/invalid-id")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
