import unittest

import pandas as pd

from scripts.recommend_metadata import (
    jaccard_similarity,
    prepare_catalog,
    recommend_similar_movies,
)


def sample_catalog() -> pd.DataFrame:
    return prepare_catalog(pd.DataFrame([
        {"movie_id": 1, "title": "Action One", "release_year": 2000,
         "genres": "Action|Sci-Fi", "plot_tags": "space, hero"},
        {"movie_id": 2, "title": "Action Two", "release_year": 2005,
         "genres": "Action|Sci-Fi", "plot_tags": pd.NA},
        {"movie_id": 3, "title": "Space Drama", "release_year": 2010,
         "genres": "Drama", "plot_tags": "space, hero"},
        {"movie_id": 4, "title": "No Metadata", "release_year": pd.NA,
         "genres": "(no genres listed)", "plot_tags": pd.NA},
    ]))


class MetadataRecommenderTests(unittest.TestCase):
    def test_jaccard_similarity(self) -> None:
        self.assertEqual(jaccard_similarity({"action", "sci-fi"}, {"action"}), 0.5)
        self.assertEqual(jaccard_similarity(set(), set()), 0.0)

    def test_recommendations_exclude_query_and_show_shared_metadata(self) -> None:
        results = recommend_similar_movies(sample_catalog(), 1, top_k=5)
        self.assertNotIn(1, results["movie_id"].tolist())
        self.assertEqual(results["movie_id"].tolist(), [2, 3])
        self.assertEqual(results.iloc[0]["shared_genres"], "action, sci-fi")
        self.assertAlmostEqual(float(results.iloc[0]["similarity"]), 0.7)
        self.assertEqual(results.iloc[1]["shared_tags"], "hero, space")
        self.assertAlmostEqual(float(results.iloc[1]["similarity"]), 0.3)
        self.assertTrue(results["movie_id"].is_unique)

    def test_missing_tags_fall_back_to_genres_and_empty_metadata_returns_nothing(self) -> None:
        catalog = sample_catalog()
        results = recommend_similar_movies(catalog, 2, top_k=5)
        self.assertIn(1, results["movie_id"].tolist())
        self.assertEqual(float(results.iloc[0]["similarity"]), 1.0)
        self.assertTrue(recommend_similar_movies(catalog, 4).empty)

    def test_unknown_movie_and_invalid_top_k_are_clear_errors(self) -> None:
        catalog = sample_catalog()
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            recommend_similar_movies(catalog, 999)
        with self.assertRaisesRegex(ValueError, "top_k"):
            recommend_similar_movies(catalog, 1, top_k=0)

    def test_duplicate_movie_ids_are_rejected(self) -> None:
        movies = pd.DataFrame([
            {"movie_id": 1, "title": "A", "release_year": 2000,
             "genres": "Action", "plot_tags": pd.NA},
            {"movie_id": 1, "title": "A duplicate", "release_year": 2001,
             "genres": "Action", "plot_tags": pd.NA},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate movie_id"):
            prepare_catalog(movies)


if __name__ == "__main__":
    unittest.main()
