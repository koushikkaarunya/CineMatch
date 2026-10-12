import json
import unittest

import pandas as pd
from scipy import sparse

from scripts.recommend_cold_start import ColdStartRecommender


def people(qid: str, name: str) -> str:
    return json.dumps([{"wikidata_id": qid, "name": name}])


def catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {"movie_id": 1, "title": "Seed One", "genres": "Adventure", "plot_tags": "space",
         "plot": "A pilot travels through space.", "directors": people("Q1", "Director One")},
        {"movie_id": 2, "title": "Candidate Two", "genres": "Adventure", "plot_tags": "journey",
         "plot": "A traveler finds a new world.", "directors": pd.NA},
        {"movie_id": 3, "title": "Candidate Three", "genres": pd.NA, "plot_tags": pd.NA,
         "plot": pd.NA, "directors": pd.NA},
        {"movie_id": 4, "title": "No Plot Seed", "genres": "Drama", "plot_tags": pd.NA,
         "plot": pd.NA, "directors": pd.NA},
        {"movie_id": 5, "title": "Seed Five", "genres": "Comedy", "plot_tags": "friendship",
         "plot": "Two friends rebuild their lives.", "directors": people("Q5", "Director Five")},
        {"movie_id": 6, "title": "No Content", "genres": "(no genres listed)", "plot_tags": pd.NA,
         "plot": pd.NA, "directors": pd.NA},
    ])


def result_rows(rows: list[tuple[int, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["movie_id", "hybrid_score", "explanation"])


class FakeContentRecommender:
    def __init__(self, results: dict[int, pd.DataFrame]) -> None:
        self.results = results
        self.calls: list[tuple[int, int]] = []

    def recommend(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        self.calls.append((movie_id, top_k))
        return self.results.get(movie_id, result_rows([])).copy(deep=True)


class FakeCollaborativeRecommender:
    def __init__(self) -> None:
        self.catalog_ids = set(range(1, 7))
        self.movie_to_column = {1: 0, 2: 1, 5: 2}
        self.similarity_matrix = sparse.csr_matrix([
            [0.0, 0.4, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        self.user_history = {
            10: {1: 5.0},
            11: {1: 3.0, 2: 3.0},
            12: {1: 5.0, 2: 2.0},
            13: {1: 4.0, 2: 2.0},
        }
        self.user_means = {10: 5.0, 11: 3.0, 12: 3.5, 13: 3.0}
        self.calls: list[int] = []

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        self.calls.append(user_id)
        if user_id == 12:
            return pd.DataFrame([{
                "movie_id": 4, "title": "No Plot Seed", "predicted_rating": 4.4,
                "support_count": 2, "evidence_weight": 0.8, "supporting_movies": "Seed One",
            }]).head(top_k)
        return pd.DataFrame(columns=["movie_id", "title", "predicted_rating"])


def make_recommender() -> tuple[ColdStartRecommender, FakeContentRecommender, FakeCollaborativeRecommender]:
    content = FakeContentRecommender({
        1: result_rows([
            (2, 0.91, "genre evidence"),
            (3, 0.81, "plot evidence"),
            (4, 0.81, "metadata evidence"),
            (4, 0.79, "duplicate row"),
            (5, 0.78, "selected seed should be filtered"),
        ]),
        4: result_rows([]),
        5: result_rows([
            (3, 0.95, "plot evidence"),
            (2, 0.80, "genre evidence"),
        ]),
        6: result_rows([]),
    })
    collaborative = FakeCollaborativeRecommender()
    return ColdStartRecommender(catalog(), content, collaborative), content, collaborative


class ColdStartRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recommender, self.content, self.collaborative = make_recommender()

    def test_new_user_without_preferences_gets_clear_state_not_generic_list(self) -> None:
        response = self.recommender.recommend_from_favorites([])
        self.assertEqual(response.status, "needs_preferences")
        self.assertEqual(response.recommendation_source, "none")
        self.assertTrue(response.recommendations.empty)
        self.assertIn("Choose", response.message)
        self.assertEqual(self.content.calls, [])

    def test_one_valid_seed_returns_phase9_content_recommendations(self) -> None:
        response = self.recommender.recommend_from_favorites([1], top_k=5)
        self.assertEqual(response.status, "content_based_from_selected_movies")
        self.assertEqual(response.recommendation_source, "phase9_content_rank_aggregation")
        self.assertEqual(response.seed_movie_ids, (1,))
        self.assertEqual(response.recommendations.movie_id.tolist(), [2, 3, 4, 5])
        self.assertNotIn(1, response.recommendations.movie_id.tolist())
        self.assertNotIn("predicted_rating", response.recommendations.columns)
        self.assertIn("phase9_hybrid_scores", response.recommendations.columns)

    def test_multiple_seeds_aggregate_rankings_and_count_shared_candidates(self) -> None:
        response = self.recommender.recommend_from_favorites([1, 5], top_k=5)
        recommendations = response.recommendations.set_index("movie_id")
        self.assertEqual(response.seed_movie_ids, (1, 5))
        self.assertEqual(recommendations.loc[2, "seed_support_count"], 2)
        self.assertEqual(recommendations.loc[2, "seed_ranks"], {1: 1, 5: 2})
        self.assertAlmostEqual(recommendations.loc[2, "mean_reciprocal_rank"], 0.75)
        self.assertAlmostEqual(recommendations.loc[3, "mean_reciprocal_rank"], 0.75)
        self.assertEqual(response.recommendations.movie_id.tolist()[:2], [2, 3])
        self.assertNotIn(1, recommendations.index)
        self.assertNotIn(5, recommendations.index)
        self.assertEqual(recommendations.loc[2, "phase9_hybrid_scores"], {1: 0.91, 5: 0.80})

    def test_duplicate_seed_ids_are_ignored_without_changing_order_or_score(self) -> None:
        response = self.recommender.recommend_from_favorites([1, 1, 5], top_k=5)
        self.assertEqual(response.seed_movie_ids, (1, 5))
        row = response.recommendations.loc[response.recommendations.movie_id.eq(2)].iloc[0]
        self.assertAlmostEqual(row.mean_reciprocal_rank, 0.75)

    def test_unknown_or_invalid_seed_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown movie_id: 999"):
            self.recommender.recommend_from_favorites([999])
        for values in (["1"], [True], "1"):
            with self.assertRaises(ValueError):
                self.recommender.recommend_from_favorites(values)
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.recommender.recommend_from_favorites([1], top_k=0)

    def test_all_empty_seed_results_and_partial_empty_seed_results_are_explained(self) -> None:
        empty = self.recommender.recommend_from_favorites([4, 6])
        self.assertEqual(empty.status, "no_content_evidence")
        self.assertTrue(empty.recommendations.empty)
        partial = self.recommender.recommend_from_favorites([1, 4])
        self.assertEqual(partial.status, "content_based_from_partial_seed_evidence")
        self.assertIn("1 of 2", partial.message)
        self.assertNotIn(4, partial.recommendations.movie_id.tolist())

    def test_seed_movies_are_removed_before_aggregation_ranks_are_assigned(self) -> None:
        response = self.recommender.recommend_from_favorites([1, 5], top_k=5)
        row = response.recommendations.loc[response.recommendations.movie_id.eq(3)].iloc[0]
        self.assertEqual(row.seed_ranks, {1: 2, 5: 1})
        self.assertFalse(set(response.seed_movie_ids) & set(response.recommendations.movie_id))

    def test_repeated_calls_are_deterministic_and_content_outputs_are_not_mutated(self) -> None:
        before = self.content.results[1].copy(deep=True)
        first = self.recommender.recommend_from_favorites([1, 5], top_k=5)
        second = self.recommender.recommend_from_favorites([1, 5], top_k=5)
        self.assertEqual(first.recommendations.movie_id.tolist(), second.recommendations.movie_id.tolist())
        pd.testing.assert_frame_equal(before, self.content.results[1])

    def test_known_user_with_insufficient_history_needs_preferences_or_uses_them(self) -> None:
        no_seed = self.recommender.recommend_for_user(10)
        self.assertEqual(no_seed.status, "needs_preferences")
        self.assertIn("known user", no_seed.message)
        fallback = self.recommender.recommend_for_user(10, favorite_movie_ids=[1])
        self.assertEqual(fallback.status, "content_fallback_for_insufficient_collaborative_history")
        self.assertEqual(fallback.recommendation_source, "phase9_content_rank_aggregation")
        self.assertNotIn("predicted_rating", fallback.recommendations.columns)
        self.assertEqual(fallback.collaborative_evidence, "insufficient_for_user_recommendation")

    def test_no_rating_variation_and_no_candidate_evidence_are_distinguished(self) -> None:
        constant = self.recommender.recommend_for_user(11)
        self.assertEqual(constant.status, "needs_preferences")
        self.assertIn("no variation", constant.message)
        no_candidates = self.recommender.recommend_for_user(13)
        self.assertEqual(no_candidates.status, "needs_preferences")
        self.assertIn("no unseen candidates", no_candidates.message)

    def test_unknown_user_is_not_treated_as_known_insufficient_user(self) -> None:
        response = self.recommender.recommend_for_user(999, favorite_movie_ids=[1])
        self.assertEqual(response.status, "unknown_user")
        self.assertEqual(response.recommendation_source, "none")
        self.assertTrue(response.recommendations.empty)

    def test_sufficient_user_gets_phase10_predictions_without_score_blending(self) -> None:
        response = self.recommender.recommend_for_user(12)
        self.assertEqual(response.status, "collaborative_recommendations_available")
        self.assertEqual(response.recommendation_source, "phase10_collaborative")
        self.assertIn("predicted_rating", response.recommendations.columns)
        self.assertNotIn("mean_reciprocal_rank", response.recommendations.columns)

    def test_movie_without_collaborative_links_uses_content_and_reports_status(self) -> None:
        response = self.recommender.recommend_for_movie(5)
        self.assertEqual(response.status, "content_fallback_without_collaborative_evidence")
        self.assertEqual(response.collaborative_evidence, "no_qualifying_item_similarity")
        self.assertEqual(response.recommendation_source, "phase9_content")
        self.assertIn("genres", response.available_content_fields)
        self.assertIn("plot", response.available_content_fields)
        self.assertIn("hybrid_score", response.recommendations.columns)
        self.assertNotIn("predicted_rating", response.recommendations.columns)

    def test_movie_with_collaborative_links_is_still_labeled_content_only(self) -> None:
        response = self.recommender.recommend_for_movie(1)
        self.assertEqual(response.status, "content_recommendations_with_collaborative_evidence")
        self.assertEqual(response.collaborative_evidence, "item_similarity_available")
        self.assertEqual(response.recommendation_source, "phase9_content")
        self.assertIn("scores were combined", response.message)

    def test_unknown_movie_missing_plot_and_missing_all_content_are_distinct(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown movie_id: 999"):
            self.recommender.recommend_for_movie(999)
        no_plot = self.recommender.recommend_for_movie(4)
        self.assertEqual(no_plot.status, "no_content_candidates")
        self.assertEqual(no_plot.available_content_fields, ("genres",))
        empty = self.recommender.recommend_for_movie(6)
        self.assertEqual(empty.status, "no_content_metadata")
        self.assertEqual(empty.available_content_fields, ())
        self.assertIn("no usable", empty.message)
        self.assertTrue(pd.isna(self.recommender.movies.loc[6, "plot"]))
        self.assertTrue(pd.isna(self.recommender.movies.loc[6, "directors"]))


if __name__ == "__main__":
    unittest.main()
