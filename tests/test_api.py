import asyncio
import json
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from app import main
from scripts.recommend_hybrid import HybridResponse


def people(name: str) -> str:
    return json.dumps([{"wikidata_id": "Q1", "name": name}])


def sample_catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "movie_id": 1, "imdb_id": "tt0000001", "title": "Toy Story (1995)",
            "release_year": 1995.0, "genres": "Adventure|Animation", "plot": "A toy comes alive.",
            "plot_tags": "friendship, toys", "directors": people("Director One"),
            "writers": "invalid json", "cast": pd.NA, "selected_crew": pd.NA,
            "tmdb_id": 862.0, "wikidata_id": "Q123", "metadata_source": "test",
        },
        {
            "movie_id": 2, "imdb_id": "tt0000002", "title": "Toy Story 2 (1999)",
            "release_year": 1999.0, "genres": "(no genres listed)", "plot": pd.NA,
            "plot_tags": pd.NA, "directors": pd.NA, "writers": pd.NA,
            "cast": pd.NA, "selected_crew": pd.NA, "tmdb_id": np.nan,
            "wikidata_id": pd.NA, "metadata_source": pd.NA,
        },
        {
            "movie_id": 3, "imdb_id": "tt0000003", "title": "Other Film (2000)",
            "release_year": 2000.0, "genres": "Drama", "plot": "A family reunites.",
            "plot_tags": "family", "directors": pd.NA, "writers": pd.NA,
            "cast": pd.NA, "selected_crew": pd.NA, "tmdb_id": 3.0,
            "wikidata_id": pd.NA, "metadata_source": pd.NA,
        },
    ])


class FakeRecommender:
    def __init__(self) -> None:
        self.movies = sample_catalog().set_index("movie_id", drop=False)
        self.movie_calls = []
        self.user_calls = []
        self.favorite_calls = []

    @staticmethod
    def _response(status="hybrid_recommendations"):
        recommendations = pd.DataFrame([{
            "movie_id": np.int64(3),
            "title": "Other Film (2000)",
            "fusion_score": 0.75,
            "predicted_rating": np.nan,
            "content_explanation": {np.int64(1): "genre evidence", 2: pd.NA},
        }])
        return HybridResponse(
            status=status,
            message="Scores order the result.",
            recommendation_source="content",
            recommendations=recommendations,
            positive_seed_movie_ids=(1,),
            collaborative_available=False,
            content_available=True,
        )

    def recommend_for_movie(self, movie_id, top_k=10):
        self.movie_calls.append((movie_id, top_k))
        if movie_id == 2:
            return HybridResponse(
                status="no_content_metadata",
                message="No content fields were available.",
                recommendation_source="none",
                recommendations=pd.DataFrame(),
            )
        return self._response("content_fallback_without_collaborative_evidence")

    def recommend_for_user(self, user_id, top_k=10):
        self.user_calls.append((user_id, top_k))
        if user_id == 999:
            return self._response("unknown_user")
        if user_id == 2:
            return self._response("needs_preferences")
        return self._response()

    def recommend_for_new_user(self, favorite_movie_ids, top_k=10):
        self.favorite_calls.append((list(favorite_movie_ids), top_k))
        if not favorite_movie_ids:
            return self._response("needs_preferences")
        return self._response("content_based_from_selected_movies")


async def asgi_request(method: str, url: str, body: dict | None = None):
    """Send one in-process HTTP request without adding an HTTP client dependency."""
    parsed = urlsplit(url)
    payload = b"" if body is None else json.dumps(body).encode("utf-8")
    messages = []
    received_request = False

    async def receive():
        nonlocal received_request
        if received_request:
            return {"type": "http.disconnect"}
        received_request = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http",
        "path": parsed.path, "raw_path": parsed.path.encode("ascii"),
        "query_string": parsed.query.encode("ascii"), "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "server": ("testserver", 80), "client": ("testclient", 50000),
    }
    await main.app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(response_body) if response_body else None


class CineMatchApiTests(unittest.TestCase):
    def setUp(self):
        self.recommender = FakeRecommender()
        self.builder = Mock(return_value=self.recommender)
        self.builder_patch = patch("app.main.build_hybrid_recommender", self.builder)
        self.builder_patch.start()
        self.lifespan = main.app.router.lifespan_context(main.app)
        asyncio.run(self.lifespan.__aenter__())

    def tearDown(self):
        asyncio.run(self.lifespan.__aexit__(None, None, None))
        self.builder_patch.stop()

    def request(self, method, url, body=None):
        return asyncio.run(asgi_request(method, url, body))

    def test_health_does_not_rebuild_recommender(self):
        status, payload = self.request("GET", "/api/v1/health")
        self.assertEqual((status, payload), (200, {"status": "ok"}))
        self.request("GET", "/api/v1/health")
        self.assertEqual(self.builder.call_count, 1)

    def test_movie_search_is_case_insensitive_and_empty_results_are_empty(self):
        status, result = self.request("GET", "/api/v1/movies?q=tOy%20sToRy&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["movie_id"], 1)
        self.assertEqual(result["items"][0]["genres"], ["Adventure", "Animation"])
        self.assertEqual(self.request("GET", "/api/v1/movies?q=no-match")[1], {"items": [], "count": 0})

    def test_search_validation(self):
        for url in ("/api/v1/movies?q=%20%20", "/api/v1/movies?q=toy&limit=0",
                    "/api/v1/movies?q=toy&limit=51", "/api/v1/movies?q=toy&limit=bad"):
            self.assertEqual(self.request("GET", url)[0], 400, url)

    def test_details_parse_credits_and_preserve_missing_values(self):
        status, result = self.request("GET", "/api/v1/movies/1")
        self.assertEqual(status, 200)
        self.assertEqual(result["release_year"], 1995)
        self.assertEqual(result["directors"], [{"wikidata_id": "Q1", "name": "Director One"}])
        self.assertEqual(result["writers"], [])
        self.assertEqual(result["external_ids"], {
            "imdb_id": "tt0000001", "tmdb_id": 862, "wikidata_id": "Q123",
        })
        status, missing = self.request("GET", "/api/v1/movies/2")
        self.assertEqual(status, 200)
        self.assertIsNone(missing["plot"])
        self.assertEqual(missing["genres"], [])
        self.assertEqual(missing["directors"], [])
        self.assertIsNone(missing["external_ids"]["tmdb_id"])
        self.assertEqual(self.request("GET", "/api/v1/movies/999")[0], 404)

    def test_invalid_path_id_is_a_clear_400(self):
        status, payload = self.request("GET", "/api/v1/movies/not-an-id")
        self.assertEqual(status, 400)
        self.assertIn("detail", payload)

    def test_movie_recommendations_preserve_status_and_json_safe_records(self):
        status, payload = self.request("GET", "/api/v1/movies/1/recommendations?top_k=5")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "content_fallback_without_collaborative_evidence")
        self.assertEqual(payload["recommendation_source"], "content")
        self.assertEqual(payload["items"][0]["movie_id"], 3)
        self.assertIsNone(payload["items"][0]["predicted_rating"])
        self.assertEqual(payload["items"][0]["content_explanation"], {
            "1": "genre evidence", "2": None,
        })
        self.assertEqual(self.recommender.movie_calls, [(1, 5)])
        status, empty = self.request("GET", "/api/v1/movies/2/recommendations")
        self.assertEqual(status, 200)
        self.assertEqual(empty["status"], "no_content_metadata")
        self.assertEqual(empty["items"], [])
        self.assertEqual(self.request("GET", "/api/v1/movies/999/recommendations")[0], 404)

    def test_recommendation_top_k_bounds(self):
        for top_k in (0, 51, "bad"):
            status, _ = self.request("GET", f"/api/v1/movies/1/recommendations?top_k={top_k}")
            self.assertEqual(status, 400)

    def test_existing_user_and_unknown_user_convention(self):
        status, payload = self.request("GET", "/api/v1/users/10/recommendations")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "hybrid_recommendations")
        status, payload = self.request("GET", "/api/v1/users/999/recommendations")
        self.assertEqual(status, 404)
        self.assertEqual(payload["detail"]["status"], "unknown_user")
        self.assertEqual(self.request("GET", "/api/v1/users/2/recommendations")[1]["status"], "needs_preferences")

    def test_favorites_empty_deduplicated_and_unknown(self):
        status, payload = self.request("POST", "/api/v1/recommendations/from-favorites", {
            "favorite_movie_ids": [1, 1, 2], "top_k": 4,
        })
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "content_based_from_selected_movies")
        self.assertEqual(self.recommender.favorite_calls[-1], ([1, 2], 4))
        status, payload = self.request("POST", "/api/v1/recommendations/from-favorites", {
            "favorite_movie_ids": [],
        })
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "needs_preferences")
        self.assertEqual(self.request("POST", "/api/v1/recommendations/from-favorites", {
            "favorite_movie_ids": [999],
        })[0], 404)

    def test_favorites_reject_invalid_ids_and_top_k(self):
        for body in (
            {"favorite_movie_ids": [1.5]},
            {"favorite_movie_ids": [True]},
            {"favorite_movie_ids": [1], "top_k": 0},
            {"favorite_movie_ids": [1], "top_k": 51},
            {"favorite_movie_ids": [1], "unexpected": 2},
        ):
            self.assertEqual(self.request("POST", "/api/v1/recommendations/from-favorites", body)[0], 400)

    def test_recommender_is_initialized_once_for_multiple_requests(self):
        self.request("GET", "/api/v1/health")
        self.request("GET", "/api/v1/movies?q=toy")
        self.request("GET", "/api/v1/movies/1")
        self.request("GET", "/api/v1/movies/1/recommendations")
        self.request("GET", "/api/v1/users/10/recommendations")
        self.assertEqual(self.builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
