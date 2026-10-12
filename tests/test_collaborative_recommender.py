import unittest

import numpy as np
import pandas as pd

from scripts.recommend_collaborative import (
    ItemItemCollaborativeRecommender,
    clip_predicted_rating,
    predict_rating,
)


def sample_ratings() -> pd.DataFrame:
    # A and B have matching user-centered ratings; C moves in the opposite direction.
    rows = [
        (1, 10, 5), (1, 20, 5), (1, 30, 1),
        (2, 10, 4), (2, 20, 4), (2, 30, 2),
        (3, 10, 2), (3, 20, 2), (3, 30, 5),
        (4, 10, 5), (4, 20, 5), (4, 30, 1),
        # User 5 has a preference history but has not rated movie 20.
        (5, 10, 5), (5, 30, 1),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame([
        {"movie_id": 10, "title": "A"},
        {"movie_id": 20, "title": "B"},
        {"movie_id": 30, "title": "C"},
        {"movie_id": 40, "title": "D"},
    ])


class CollaborativeRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ratings = sample_ratings()
        self.model = ItemItemCollaborativeRecommender(
            self.ratings, sample_movies(), min_common_users=3, shrinkage=10
        )

    def test_rating_validation_and_catalog_relationships(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns.*user_id"):
            ItemItemCollaborativeRecommender(self.ratings.drop(columns="user_id"))
        for bad_value in (0, 5.5, np.inf, np.nan, 2.25):
            broken = self.ratings.astype({"rating": "float64"}).copy()
            broken.loc[0, "rating"] = bad_value
            with self.assertRaises(ValueError):
                ItemItemCollaborativeRecommender(broken)
        duplicate = pd.concat([self.ratings, self.ratings.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate user_id/movie_id"):
            ItemItemCollaborativeRecommender(duplicate)
        bad_catalog = sample_movies().loc[lambda frame: frame.movie_id.ne(30)]
        with self.assertRaisesRegex(ValueError, "absent from the supplied catalog"):
            ItemItemCollaborativeRecommender(self.ratings, bad_catalog)

    def test_integer_user_and_movie_ids_are_preserved(self) -> None:
        self.assertEqual(self.model.users, [1, 2, 3, 4, 5])
        self.assertEqual(self.model.movie_ids, [10, 20, 30])
        self.assertEqual(self.model.user_history[5], {10: 5.0, 30: 1.0})

    def test_similarity_uses_mean_centered_ratings_and_shrinkage(self) -> None:
        # Four co-raters give an unshrunk A/B similarity of 1; 4/(4+10) shrinks it.
        self.assertAlmostEqual(self.model.similarity(10, 20), 4 / 14)
        self.assertLess(self.model.similarity(10, 30), 0)
        self.assertEqual(self.model.similarity(20, 10), self.model.similarity(10, 20))

    def test_rater_average_tendencies_are_removed_before_similarity(self) -> None:
        rows = []
        patterns = [
            (1, [5, 5, 1]), (2, [4, 4, 2]), (3, [2, 2, 5]),
            (4, [5, 5, 1]), (5, [3, 3, 0.5]),
        ]
        for user_id, values in patterns:
            rows.extend((user_id, movie_id, rating)
                        for movie_id, rating in zip((10, 20, 30), values))
        model = ItemItemCollaborativeRecommender(
            pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"]),
            min_common_users=3, shrinkage=0,
        )
        self.assertAlmostEqual(model.similarity(10, 20), 1.0)

    def test_minimum_overlap_and_shrinkage_remove_weak_evidence(self) -> None:
        strict = ItemItemCollaborativeRecommender(
            self.ratings, min_common_users=5, shrinkage=10
        )
        self.assertIsNone(strict.similarity(10, 20))
        heavy_shrinkage = ItemItemCollaborativeRecommender(
            self.ratings, min_common_users=3, shrinkage=100
        )
        self.assertLess(heavy_shrinkage.similarity(10, 20), self.model.similarity(10, 20))

    def test_recommendations_use_user_patterns_and_exclude_seen_movies(self) -> None:
        results = self.model.recommend(5, top_k=5)
        self.assertEqual(results.iloc[0].movie_id, 20)
        self.assertNotIn(10, results.movie_id.tolist())
        self.assertNotIn(30, results.movie_id.tolist())
        self.assertTrue(results.movie_id.is_unique)
        self.assertAlmostEqual(results.iloc[0].predicted_rating, 3.7272727273)
        self.assertEqual(results.iloc[0].support_count, 2)
        self.assertIn("A", results.iloc[0].supporting_movies)

    def test_unknown_users_movies_and_invalid_top_k_are_clear(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown user_id"):
            self.model.recommend(999)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "top_k"):
                self.model.recommend(5, invalid)
        for movie_id in (999,):
            with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
                self.model.similarity(movie_id, 10)
        self.assertIsNone(self.model.similarity(40, 10))

    def test_known_user_with_too_little_or_no_rating_variation_is_distinguished(self) -> None:
        one_rating = pd.DataFrame([(1, 10, 4)], columns=["user_id", "movie_id", "rating"])
        model = ItemItemCollaborativeRecommender(one_rating)
        with self.assertRaisesRegex(ValueError, "only 1 rating"):
            model.recommend(1)
        constant = pd.DataFrame([
            (1, 10, 4), (1, 20, 4),
            (2, 10, 3), (2, 20, 3),
        ], columns=["user_id", "movie_id", "rating"])
        model = ItemItemCollaborativeRecommender(constant)
        with self.assertRaisesRegex(ValueError, "no rating variation"):
            model.recommend(1)

    def test_no_evidence_returns_empty_and_empty_training_data_is_rejected(self) -> None:
        sparse_history = pd.DataFrame([
            (1, 10, 5), (1, 30, 1),
            (2, 10, 4), (2, 40, 1),
        ], columns=["user_id", "movie_id", "rating"])
        model = ItemItemCollaborativeRecommender(sparse_history, min_common_users=3)
        self.assertTrue(model.recommend(1).empty)
        with self.assertRaisesRegex(ValueError, "At least one rating"):
            ItemItemCollaborativeRecommender(self.ratings.iloc[0:0])

    def test_predicted_rating_is_clipped_to_movie_lens_range(self) -> None:
        self.assertEqual(clip_predicted_rating(5.7), 5.0)
        self.assertEqual(clip_predicted_rating(0.1), 0.5)
        self.assertEqual(clip_predicted_rating(3.25), 3.25)
        self.assertEqual(predict_rating(4.8, 2.0, 0.5, prediction_shrinkage=0), 5.0)
        self.assertEqual(predict_rating(0.7, -2.0, 0.5, prediction_shrinkage=0), 0.5)

    def test_deterministic_ties_and_metadata_do_not_change_recommendations(self) -> None:
        base = self.model.recommend(5, top_k=10)
        enriched = sample_movies()
        enriched["genres"] = ["Comedy", "Thriller", "Documentary", "Drama"]
        enriched["plot"] = ["Different stories"] * 4
        with_metadata = ItemItemCollaborativeRecommender(
            self.ratings, enriched, min_common_users=3, shrinkage=10
        ).recommend(5, top_k=10)
        self.assertEqual(base.movie_id.tolist(), with_metadata.movie_id.tolist())
        np.testing.assert_allclose(base.predicted_rating, with_metadata.predicted_rating)
        self.assertEqual(base.movie_id.tolist(), self.model.recommend(5, 10).movie_id.tolist())

    def test_restricted_training_data_controls_history_and_similarity(self) -> None:
        restricted = self.ratings.loc[self.ratings.user_id.isin([1, 2, 3, 4])]
        model = ItemItemCollaborativeRecommender(restricted, min_common_users=3)
        self.assertNotIn(5, model.user_history)
        with self.assertRaisesRegex(KeyError, "Unknown user_id"):
            model.recommend(5)
        self.assertAlmostEqual(model.similarity(10, 20), 4 / 14)


if __name__ == "__main__":
    unittest.main()
