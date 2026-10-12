import unittest

import pandas as pd
from scipy.sparse import issparse

from scripts.recommend_tfidf import TfidfMovieRecommender


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame([
        {"movie_id": 20, "title": "Space Pilot", "release_year": 2001,
         "plot": "A space pilot travels to Mars to rescue the colonists."},
        {"movie_id": 10, "title": "Mars Rescue", "release_year": 2002,
         "plot": "The pilot travels through space to rescue colonists on Mars."},
        {"movie_id": 30, "title": "Seaside Baker", "release_year": 2003,
         "plot": "A baker opens a seaside cafe and befriends a violin player."},
        {"movie_id": 31, "title": "Space Mechanic", "release_year": 2004,
         "plot": "A mechanic repairs a broken space rocket."},
        {"movie_id": 40, "title": "Plot Missing", "release_year": pd.NA,
         "plot": pd.NA},
        {"movie_id": 50, "title": "Blank Plot", "release_year": 2004,
         "plot": "   "},
    ])


class TfidfRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TfidfMovieRecommender(sample_movies())

    def test_vectorization_creates_sparse_matrix_and_expected_terms(self) -> None:
        self.assertTrue(issparse(self.model.tfidf_matrix))
        self.assertEqual(self.model.tfidf_matrix.shape[0], 4)
        self.assertIn("space", self.model.vectorizer.vocabulary_)
        self.assertIn("space pilot", self.model.vectorizer.vocabulary_)

    def test_similarity_ranks_related_plot_above_unrelated_plot(self) -> None:
        results = self.model.recommend(20, top_k=2)
        self.assertEqual(results.iloc[0]["movie_id"], 10)
        self.assertEqual(results.iloc[1]["movie_id"], 31)
        self.assertGreater(results.iloc[0]["similarity"], results.iloc[1]["similarity"])

    def test_missing_plot_query_errors_and_missing_plot_candidates_are_excluded(self) -> None:
        with self.assertRaisesRegex(ValueError, "no usable plot"):
            self.model.recommend(40)
        results = self.model.recommend(20, top_k=10)
        self.assertNotIn(40, results["movie_id"].tolist())
        self.assertNotIn(50, results["movie_id"].tolist())

    def test_unknown_movie_and_invalid_top_k(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            self.model.recommend(999)
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.model.recommend(20, top_k=0)

    def test_query_is_excluded_results_are_unique_and_row_mapping_is_correct(self) -> None:
        self.assertEqual(self.model.movie_id_to_row[20], 0)
        self.assertEqual(self.model.plot_movies.iloc[0]["movie_id"], 20)
        self.assertEqual(self.model.tfidf_matrix.shape[0], len(self.model.plot_movies))
        row_for_ten = self.model.movie_id_to_row[10]
        self.assertEqual(self.model.plot_movies.iloc[row_for_ten]["movie_id"], 10)
        plot_for_ten = self.model.plot_movies.iloc[row_for_ten]["plot"]
        expected_vector = self.model.vectorizer.transform([plot_for_ten])
        self.assertEqual((self.model.tfidf_matrix[row_for_ten] != expected_vector).nnz, 0)
        results = self.model.recommend(20, top_k=10)
        self.assertNotIn(20, results["movie_id"].tolist())
        self.assertTrue(results["movie_id"].is_unique)

    def test_duplicate_movie_ids_are_rejected(self) -> None:
        movies = sample_movies()
        duplicate = movies.iloc[[0]]
        with self.assertRaisesRegex(ValueError, "duplicate movie_id"):
            TfidfMovieRecommender(pd.concat([movies, duplicate], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
