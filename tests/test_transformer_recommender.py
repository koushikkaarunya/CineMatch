import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.recommend_transformer import TransformerMovieRecommender


class FakeSentenceModel:
    """Return controlled 384-dimensional vectors without downloading a model."""

    def __init__(self, vectors_by_plot: dict[str, np.ndarray]) -> None:
        self.vectors_by_plot = vectors_by_plot
        self.calls = 0

    def encode(self, plots, **kwargs) -> np.ndarray:
        self.calls += 1
        vectors = np.vstack([self.vectors_by_plot[plot] for plot in plots]).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(384, dtype=np.float32)
    vector[index] = 1.0
    return vector


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame([
        {"movie_id": 20, "title": "Rescue", "release_year": 2001,
         "plot": "A pilot rescues colonists on Mars."},
        {"movie_id": 10, "title": "Mars Mission", "release_year": 2002,
         "plot": "A crew saves settlers on Mars."},
        {"movie_id": 30, "title": "Cafe", "release_year": 2003,
         "plot": "A baker opens a seaside cafe."},
        {"movie_id": 40, "title": "Missing", "release_year": pd.NA, "plot": pd.NA},
        {"movie_id": 50, "title": "Blank", "release_year": 2004, "plot": "   "},
    ])


class TransformerRecommenderTests(unittest.TestCase):
    def make_model(self, movies=None, cache_path=None) -> tuple[TransformerMovieRecommender, FakeSentenceModel]:
        vectors = {
            "A pilot rescues colonists on Mars.": unit_vector(0),
            "A crew saves settlers on Mars.": unit_vector(0) * 0.8 + unit_vector(1) * 0.6,
            "A baker opens a seaside cafe.": unit_vector(2),
        }
        model = FakeSentenceModel(vectors)
        recommender = TransformerMovieRecommender(
            sample_movies() if movies is None else movies,
            model=model,
            cache_path=cache_path,
        )
        return recommender, model

    def test_embedding_generation_dimension_and_id_alignment(self) -> None:
        recommender, model = self.make_model()
        self.assertEqual(model.calls, 1)
        self.assertEqual(recommender.embeddings.shape, (3, 384))
        self.assertEqual(recommender.movie_id_to_row, {20: 0, 10: 1, 30: 2})
        np.testing.assert_allclose(recommender.embeddings[1], model.encode(
            ["A crew saves settlers on Mars."]
        )[0])

    def test_related_plot_ranks_first_and_query_is_excluded(self) -> None:
        recommender, _ = self.make_model()
        results = recommender.recommend(20, top_k=10)
        self.assertEqual(results.iloc[0]["movie_id"], 10)
        self.assertNotIn(20, results["movie_id"].tolist())
        self.assertTrue(results["movie_id"].is_unique)
        self.assertAlmostEqual(float(results.iloc[0]["similarity"]), 0.8)
        self.assertNotIn(40, results["movie_id"].tolist())
        self.assertNotIn(50, results["movie_id"].tolist())

    def test_missing_empty_unknown_and_invalid_inputs(self) -> None:
        recommender, _ = self.make_model()
        for missing_id in (40, 50):
            with self.assertRaisesRegex(ValueError, "no usable plot"):
                recommender.recommend(missing_id)
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            recommender.recommend(999)
        for value in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "top_k"):
                recommender.recommend(20, top_k=value)

    def test_stable_tie_order_uses_movie_id(self) -> None:
        movies = sample_movies().iloc[:3].copy()
        movies.loc[movies.movie_id.eq(10), "plot"] = "A pilot rescues colonists on Mars."
        movies.loc[movies.movie_id.eq(30), "plot"] = "A pilot rescues colonists on Mars."
        recommender, _ = self.make_model(movies)
        results = recommender.recommend(20, top_k=2)
        self.assertEqual(results["movie_id"].tolist(), [10, 30])

    def test_cache_keeps_ids_and_avoids_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.npz"
            first, first_model = self.make_model(cache_path=cache_path)
            second, second_model = self.make_model(cache_path=cache_path)
            self.assertEqual(first_model.calls, 1)
            self.assertEqual(second_model.calls, 0)
            np.testing.assert_array_equal(first.embeddings, second.embeddings)

    def test_changed_plot_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.npz"
            self.make_model(cache_path=cache_path)
            movies = sample_movies()
            changed_plot = "A pilot rescues settlers on Mars."
            movies.loc[movies.movie_id.eq(20), "plot"] = changed_plot
            model = FakeSentenceModel({
                changed_plot: unit_vector(0),
                "A crew saves settlers on Mars.": unit_vector(0) * 0.8 + unit_vector(1) * 0.6,
                "A baker opens a seaside cafe.": unit_vector(2),
            })
            TransformerMovieRecommender(movies, model=model, cache_path=cache_path)
            self.assertEqual(model.calls, 1)

    def test_ratings_are_not_required_or_used(self) -> None:
        movies = sample_movies()
        baseline, _ = self.make_model(movies)
        movies["rating"] = [0, 5, 1, 3, 4]
        rated_input, _ = self.make_model(movies)
        pd.testing.assert_frame_equal(baseline.recommend(20), rated_input.recommend(20))

    def test_duplicate_ids_and_invalid_embedding_shapes_are_rejected(self) -> None:
        movies = sample_movies()
        with self.assertRaisesRegex(ValueError, "duplicate movie_id"):
            self.make_model(pd.concat([movies, movies.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "Expected embeddings"):
            TransformerMovieRecommender(movies, embeddings=np.zeros((3, 12), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
