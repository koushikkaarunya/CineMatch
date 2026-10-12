import unittest

import pandas as pd

from scripts.phase14_optimization import _feature_coverage_diagnostics, _weighted_rank_fusion
from scripts.recommender_evaluation import make_temporal_split, split_ratings_by_manifest


class WeightedFusionTests(unittest.TestCase):
    def test_equal_weights_match_reciprocal_rank_fusion_order(self):
        ids, details = _weighted_rank_fusion(
            [10, 20, 30], [20, 40, 10],
            weights={"collaborative": 0.5, "content": 0.5}, top_k=4,
        )
        self.assertEqual(ids, [20, 10, 40, 30])
        self.assertAlmostEqual(details[10]["fusion_score"], 0.5 * (1 + 1 / 3))
        self.assertEqual(details[10]["source_ranks"], {"collaborative": 1, "content": 3})

    def test_weighted_content_can_raise_shared_content_candidates(self):
        ids, details = _weighted_rank_fusion(
            [1, 2], [3, 2],
            weights={"collaborative": 0.25, "content": 0.75}, top_k=3,
        )
        self.assertEqual(ids[0], 3)
        self.assertEqual(details[3]["contributing_sources"], ["content"])
        self.assertEqual(details[1]["contributing_sources"], ["collaborative"])

    def test_empty_source_is_removed_from_weight_normalizer(self):
        ids, details = _weighted_rank_fusion(
            [], [8, 7], weights={"collaborative": 0.25, "content": 0.75}, top_k=2,
        )
        self.assertEqual(ids, [8, 7])
        self.assertEqual(details[8]["fusion_score"], 1.0)
        self.assertEqual(_weighted_rank_fusion(
            [], [], weights={"collaborative": 0.25, "content": 0.75}, top_k=5
        ), ([], {}))

    def test_duplicate_candidate_ids_do_not_receive_multiple_ranks(self):
        ids, details = _weighted_rank_fusion(
            [5, 5, 6], [6, 7], weights={"collaborative": 0.5, "content": 0.5}, top_k=5,
        )
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(details[6]["source_ranks"], {"collaborative": 2, "content": 1})


class NestedTemporalSplitTests(unittest.TestCase):
    def test_phase14_split_only_partitions_supplied_phase13_training_rows(self):
        ratings = pd.DataFrame({
            "user_id": [1] * 8 + [2] * 8,
            "movie_id": list(range(1, 9)) + list(range(11, 19)),
            "rating": [3.0, 4.0, 2.0, 5.0, 3.5, 4.5, 2.5, 5.0] * 2,
            "timestamp": list(range(1, 9)) + list(range(1, 9)),
        })
        phase13_train = ratings.iloc[:12].copy()
        phase13_test = ratings.iloc[12:].copy()
        manifest, _ = make_temporal_split(phase13_train, holdout_fraction=0.25, minimum_training_ratings=3)
        train, validation = split_ratings_by_manifest(phase13_train, manifest)
        train_keys = set(map(tuple, train[["user_id", "movie_id"]].to_numpy()))
        validation_keys = set(map(tuple, validation[["user_id", "movie_id"]].to_numpy()))
        test_keys = set(map(tuple, phase13_test[["user_id", "movie_id"]].to_numpy()))
        self.assertFalse(train_keys & validation_keys)
        self.assertFalse(validation_keys & test_keys)
        self.assertTrue(validation_keys.issubset(set(map(tuple, phase13_train[["user_id", "movie_id"]].to_numpy()))))


class MissingFeatureDiagnosticsTests(unittest.TestCase):
    def test_reports_missing_plot_and_credit_slices_without_inventing_values(self):
        clean = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "plot": ["A usable plot", pd.NA, "Another plot"],
        })
        enriched = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "genres": ["Comedy", "Drama", "Drama"],
            "plot_tags": ["funny", pd.NA, "family"],
            "directors": ['[{"role":"director","wikidata_id":"Q1"}]', pd.NA, pd.NA],
            "writers": [pd.NA, pd.NA, pd.NA],
            "cast": [pd.NA, pd.NA, pd.NA],
            "selected_crew": [pd.NA, pd.NA, pd.NA],
        })
        specs = {
            10: {"seeds": (1,), "history_group": "sparse_5_to_20"},
            20: {"seeds": (2,), "history_group": "sparse_5_to_20"},
        }
        relevant = {10: {2}, 20: {3}}
        models = ("phase9_content_hybrid", "phase10_collaborative", "phase12_unified_hybrid", "phase12_weighted_content_75")
        rankings = {name: {10: [2, 3], 20: [3, 1]} for name in models}
        report = _feature_coverage_diagnostics(clean, enriched, specs, relevant, rankings)
        self.assertEqual(report["catalog_field_coverage"]["usable_plot"]["missing_or_unretrieved"], 1)
        self.assertEqual(report["catalog_field_coverage"]["directors"]["available_movies"], 1)
        self.assertEqual(report["positive_seed_feature_groups"]["usable_plot"]["at_least_one_seed_with_usable_plot"]["users"], 1)
        self.assertEqual(report["positive_seed_feature_groups"]["any_usable_credit"]["no_positive_seed_with_usable_credit"]["users"], 1)
        self.assertEqual(report["conditional_target_quality"]["usable_plot"]["source_value_missing_or_unretrieved"]["relevant_interactions"], 1)


if __name__ == "__main__":
    unittest.main()
