import unittest

import numpy as np
import pandas as pd

from scripts.recommender_evaluation import (
    aggregate_rating_metrics,
    make_temporal_split,
    ranking_metrics,
    split_ratings_by_manifest,
    user_bootstrap_error_difference,
)
from scripts.evaluate_recommenders import (
    _aggregate_seed_rankings,
    _feature_coverage,
    history_group,
    positive_seed_ids,
)
from scripts.recommend_tfidf import TfidfMovieRecommender
from scripts.recommend_collaborative import ItemItemCollaborativeRecommender


def sample_ratings():
    return pd.DataFrame([
        {"user_id": 8, "movie_id": 90, "rating": 4.0, "timestamp": 10},
        {"user_id": 8, "movie_id": 12, "rating": 3.0, "timestamp": 20},
        {"user_id": 8, "movie_id": 500, "rating": 5.0, "timestamp": 20},
        {"user_id": 8, "movie_id": 2, "rating": 4.5, "timestamp": 20},
        {"user_id": 8, "movie_id": 300, "rating": 1.0, "timestamp": 40},
        {"user_id": 8, "movie_id": 45, "rating": 4.0, "timestamp": 50},
        {"user_id": 3, "movie_id": 200, "rating": 2.0, "timestamp": 11},
        {"user_id": 3, "movie_id": 100, "rating": 4.0, "timestamp": 21},
        {"user_id": 3, "movie_id": 400, "rating": 4.5, "timestamp": 31},
        {"user_id": 3, "movie_id": 150, "rating": 5.0, "timestamp": 41},
        {"user_id": 3, "movie_id": 350, "rating": 1.0, "timestamp": 51},
        {"user_id": 3, "movie_id": 250, "rating": 4.0, "timestamp": 61},
    ])


class TemporalSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_handles_timestamp_ties_by_movie_id(self):
        ratings = sample_ratings()
        first, summary = make_temporal_split(ratings, holdout_fraction=0.34, minimum_training_ratings=3)
        shuffled, _ = make_temporal_split(ratings.sample(frac=1, random_state=4), holdout_fraction=0.34, minimum_training_ratings=3)
        pd.testing.assert_frame_equal(first, shuffled)
        user8 = first.loc[first.user_id.eq(8)]
        tied = user8.loc[user8.timestamp.eq(20)].sort_values("movie_id")
        self.assertEqual(tied.movie_id.tolist(), [2, 12, 500])
        self.assertEqual(summary["users_with_timestamp_tie_at_split_boundary"], 1)

    def test_no_overlap_and_each_users_training_precedes_test(self):
        ratings = sample_ratings()
        manifest, _ = make_temporal_split(ratings, holdout_fraction=0.34, minimum_training_ratings=3)
        train, test = split_ratings_by_manifest(ratings, manifest)
        self.assertFalse(set(map(tuple, train[["user_id", "movie_id"]].to_numpy())) & set(map(tuple, test[["user_id", "movie_id"]].to_numpy())))
        for user in train.user_id.unique():
            self.assertLessEqual(train.loc[train.user_id.eq(user), "timestamp"].max(), test.loc[test.user_id.eq(user), "timestamp"].min())

    def test_changing_test_ratings_does_not_change_train_partition(self):
        ratings = sample_ratings()
        manifest, _ = make_temporal_split(ratings, holdout_fraction=0.34, minimum_training_ratings=3)
        train_a, test_a = split_ratings_by_manifest(ratings, manifest)
        changed = ratings.copy()
        test_keys = set(map(tuple, manifest.loc[manifest.partition.eq("test"), ["user_id", "movie_id"]].to_numpy()))
        changed.loc[[tuple(row) in test_keys for row in changed[["user_id", "movie_id"]].to_numpy()], "rating"] = 0.5
        train_b, test_b = split_ratings_by_manifest(changed, manifest)
        pd.testing.assert_frame_equal(train_a.sort_values(["user_id", "movie_id"]).reset_index(drop=True), train_b.sort_values(["user_id", "movie_id"]).reset_index(drop=True))
        self.assertFalse(test_a.rating.equals(test_b.rating))
        model_a = ItemItemCollaborativeRecommender(train_a, min_common_users=1)
        model_b = ItemItemCollaborativeRecommender(train_b, min_common_users=1)
        pd.testing.assert_frame_equal(model_a.ratings, model_b.ratings)
        self.assertEqual(model_a.user_means, model_b.user_means)
        np.testing.assert_array_equal(model_a.similarity_matrix.toarray(), model_b.similarity_matrix.toarray())
        self.assertEqual(
            positive_seed_ids(train_a.loc[train_a.user_id.eq(8)]),
            positive_seed_ids(train_b.loc[train_b.user_id.eq(8)]),
        )

    def test_manifest_rejects_duplicate_user_movie_pairs(self):
        ratings = pd.concat([sample_ratings(), sample_ratings().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            make_temporal_split(ratings)


class RankingMetricTests(unittest.TestCase):
    def test_precision_recall_ndcg_hand_calculation(self):
        metrics = ranking_metrics([10, 20, 30, 40], {20, 40}, 3)
        expected_dcg = 1 / np.log2(3)  # one hit at rank two
        ideal_dcg = 1 + 1 / np.log2(3)
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["ndcg"], expected_dcg / ideal_dcg)

    def test_no_relevant_test_movies_have_precision_but_undefined_recall_and_ndcg(self):
        metrics = ranking_metrics([1], set(), 5)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["ndcg"])

    def test_short_list_uses_k_denominator_and_duplicate_ids_count_once(self):
        metrics = ranking_metrics([7, 7, 9], {7}, 5)
        self.assertEqual(metrics["precision"], 0.2)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["recall"], 1.0)

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([], set(), 0)

    def test_aggregate_rating_metrics(self):
        metrics = aggregate_rating_metrics([3.0, 4.0], [4.0, 2.0])
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5))

    def test_cluster_bootstrap_is_reproducible(self):
        model = {1: np.array([1.0]), 2: np.array([-2.0])}
        baseline = {1: np.array([2.0]), 2: np.array([-1.0])}
        first = user_bootstrap_error_difference(model, baseline, metric="mae", resamples=50)
        second = user_bootstrap_error_difference(model, baseline, metric="mae", resamples=50)
        self.assertEqual(first, second)


class PersonalizationAdapterTests(unittest.TestCase):
    def test_positive_seeds_use_training_mean_and_stable_order(self):
        history = pd.DataFrame({"movie_id": [70, 20, 30, 40], "rating": [5.0, 5.0, 3.0, 1.0]})
        self.assertEqual(positive_seed_ids(history), (20, 70))

    def test_seed_aggregation_excludes_all_training_seen_movies(self):
        recommendations = _aggregate_seed_rankings(
            [[100, 200, 300], [200, 400, 100]],
            seeds_to_remove_before_ranking={100}, seen={100, 900}, top_k=3
        )
        self.assertEqual(recommendations, [200, 300, 400])
        self.assertNotIn(900, recommendations)

    def test_other_seen_movies_keep_their_seed_rank_before_final_exclusion(self):
        ranked = _aggregate_seed_rankings(
            [[900, 200, 300], [200, 900, 400]],
            seeds_to_remove_before_ranking=set(), seen={900}, top_k=3,
        )
        self.assertEqual(ranked, [200, 300, 400])

    def test_history_group_boundaries(self):
        self.assertEqual(history_group(20), "sparse_5_to_20")
        self.assertEqual(history_group(21), "moderate_21_to_50")
        self.assertEqual(history_group(50), "moderate_21_to_50")
        self.assertEqual(history_group(51), "substantial_over_50")

    def test_missing_plot_and_missing_credit_are_reported_as_unavailable(self):
        clean = pd.DataFrame({
            "movie_id": [10, 20], "title": ["Has plot", "No plot"],
            "release_year": [2000, 2001], "genres": ["Drama", "Comedy"],
            "plot": ["A complete story about a family.", ""], "plot_tags": ["family", ""],
        })
        enriched = clean.copy()
        for field in ("directors", "writers", "cast", "selected_crew"):
            enriched[field] = ["[]", ""]
        specs = {
            1: {"seeds": (10,), "seen": {10}},
            2: {"seeds": (20,), "seen": {20}},
        }
        coverage = _feature_coverage(clean, enriched, specs, {1: {20}})
        self.assertEqual(coverage["positive_seed_movie_field_coverage"]["usable_plot"]["available"], 1)
        self.assertEqual(coverage["positive_seed_movie_field_coverage"]["directors"]["available"], 0)
        self.assertEqual(coverage["relevant_heldout_target_field_coverage"]["usable_plot"]["missing"], 1)
        self.assertEqual(coverage["relevant_heldout_target_field_coverage"]["cast"]["missing"], 1)

    def test_empty_content_results_and_missing_plot_are_explicit(self):
        movies = pd.DataFrame({
            "movie_id": [3, 900], "title": ["No plot", "Only distinct text"],
            "release_year": pd.Series([2000, 2001], dtype="Int64"),
            "plot": pd.Series(["", "A completely unrelated vocabulary."], dtype="string"),
        })
        model = TfidfMovieRecommender(movies)
        with self.assertRaisesRegex(ValueError, "no usable plot"):
            model.recommend(3)
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            model.recommend(77)
        self.assertTrue(model.recommend(900).empty)


if __name__ == "__main__":
    unittest.main()
