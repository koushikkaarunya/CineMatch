import unittest

import pandas as pd

from scripts.recommend_cold_start import ColdStartResponse
from scripts.recommend_hybrid import HybridRecommender


def movies() -> pd.DataFrame:
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5, 6, 7],
        "title": [f"Movie {i}" for i in range(1, 8)],
    })


class FakeColdStart:
    def __init__(self, results=None) -> None:
        self.results = results or {}
        self.favorite_calls = []

    def recommend_from_favorites(self, seed_ids, top_k=10) -> ColdStartResponse:
        seeds = tuple(seed_ids)
        self.favorite_calls.append((seeds, top_k))
        if any(seed not in range(1, 8) for seed in seeds):
            raise KeyError(f"Unknown movie_id: {next(seed for seed in seeds if seed not in range(1, 8))}")
        if not seeds:
            return ColdStartResponse("needs_preferences", "Choose favorites", "none", pd.DataFrame(), ())
        frame = self.results.get(seeds, pd.DataFrame(columns=[
            "movie_id", "title", "mean_reciprocal_rank", "seed_explanations",
        ])).head(top_k).copy()
        status = "content_based_from_selected_movies" if len(frame) else "no_content_evidence"
        return ColdStartResponse(status, "content output", "phase9_content_rank_aggregation", frame, seeds)

    def recommend_for_movie(self, movie_id, top_k=10):
        if movie_id not in range(1, 8):
            raise KeyError(f"Unknown movie_id: {movie_id}")
        return ColdStartResponse(
            "content_candidates", "movie content", "phase9_content", self.results.get((movie_id,), pd.DataFrame()).head(top_k).copy(),
            (movie_id,), "no_qualifying_item_similarity",
        )


class FakeCollaborative:
    def __init__(self, outputs=None) -> None:
        self.outputs = outputs or {}
        self.calls = []

    def recommend(self, user_id, top_k=10) -> pd.DataFrame:
        self.calls.append((user_id, top_k))
        return self.outputs.get(user_id, pd.DataFrame(columns=[
            "movie_id", "title", "predicted_rating", "support_count", "evidence_weight",
        ])).head(top_k).copy()


def content_rows(ids_and_scores):
    return pd.DataFrame([
        {"movie_id": movie_id, "title": f"Movie {movie_id}",
         "mean_reciprocal_rank": score, "seed_explanations": {1: "plot/genre evidence"}}
        for movie_id, score in ids_and_scores
    ], columns=["movie_id", "title", "mean_reciprocal_rank", "seed_explanations"])


def collab_rows(ids_and_scores):
    return pd.DataFrame([
        {"movie_id": movie_id, "title": f"Movie {movie_id}", "predicted_rating": score,
         "support_count": 3, "evidence_weight": 0.4, "supporting_movies": "Movie 1"}
        for movie_id, score in ids_and_scores
    ], columns=["movie_id", "title", "predicted_rating", "support_count", "evidence_weight", "supporting_movies"])


class HybridRecommenderTests(unittest.TestCase):
    def make_model(self, interactions=None, content=None, collaborative=None, seed_cap=5):
        if interactions is None:
            interactions = pd.DataFrame([
                (10, 1, 5.0), (10, 2, 4.0), (10, 3, 2.0),
                (11, 1, 3.0), (11, 4, 2.0),
            ], columns=["user_id", "movie_id", "rating"])
        cold = FakeColdStart(content)
        cf = FakeCollaborative(collaborative)
        return HybridRecommender(movies(), interactions, cold, cf, positive_seed_cap=seed_cap), cold, cf

    def test_reciprocal_rank_and_validation(self):
        self.assertEqual(HybridRecommender.reciprocal_rank(2), 0.5)
        self.assertEqual(HybridRecommender.reciprocal_rank(None), 0.0)
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                HybridRecommender.reciprocal_rank(value)

    def test_fuses_both_sources_and_keeps_rating_separate(self):
        content = {(1, 2): content_rows([(4, 0.9), (5, 0.8)])}
        collaborative = {10: collab_rows([(5, 4.8), (6, 4.4)])}
        model, _, _ = self.make_model(content=content, collaborative=collaborative)
        response = model.recommend_for_user(10, top_k=5)
        result = response.recommendations.set_index("movie_id")
        self.assertEqual(response.status, "hybrid_recommendations")
        self.assertAlmostEqual(result.loc[5, "fusion_score"], 0.75)
        self.assertAlmostEqual(result.loc[4, "fusion_score"], 0.5)
        self.assertAlmostEqual(result.loc[6, "fusion_score"], 0.25)
        self.assertEqual(result.loc[5, "predicted_rating"], 4.8)
        self.assertNotEqual(result.loc[5, "fusion_score"], result.loc[5, "predicted_rating"])
        self.assertEqual(result.loc[5, "contributing_sources"], "collaborative, content")

    def test_seen_movies_filtered_and_seed_cap_is_deterministic(self):
        interactions = pd.DataFrame([
            (10, 1, 5.0), (10, 2, 4.5), (10, 3, 4.0), (10, 4, 2.0),
            (10, 5, 1.0), (10, 6, 3.0),
        ], columns=["user_id", "movie_id", "rating"])
        model, cold, _ = self.make_model(interactions, content={(1, 2, 3): content_rows([(4, .9), (7, .8)])}, seed_cap=3)
        response = model.recommend_for_user(10)
        self.assertEqual(response.positive_seed_movie_ids, (1, 2, 3))
        self.assertEqual(cold.favorite_calls[0][0], (1, 2, 3))
        self.assertEqual(response.recommendations.movie_id.tolist(), [7])
        self.assertTrue(set(interactions.movie_id) .isdisjoint(response.recommendations.movie_id))

    def test_new_user_delegates_to_phase11_and_requires_preferences(self):
        model, cold, _ = self.make_model(content={(1, 2): content_rows([(1, .99), (4, .9)])})
        response = model.recommend_for_new_user([1, 2], top_k=3)
        self.assertEqual(response.status, "content_based_from_selected_movies")
        self.assertEqual(cold.favorite_calls[-1], ((1, 2), 3))
        self.assertEqual(response.recommendations.movie_id.tolist(), [4])
        no_preferences = model.recommend_for_new_user([])
        self.assertEqual(no_preferences.status, "needs_preferences")
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            model.recommend_for_new_user([999])

    def test_unavailable_source_does_not_penalize_available_source(self):
        content = {(1, 2): content_rows([(4, .9), (5, .8)])}
        model, _, _ = self.make_model(content=content, collaborative={10: collab_rows([])})
        response = model.recommend_for_user(10)
        self.assertEqual(response.status, "content_fallback_for_insufficient_collaborative_evidence")
        self.assertAlmostEqual(response.recommendations.iloc[0].fusion_score, 1.0)
        self.assertAlmostEqual(response.recommendations.iloc[1].fusion_score, 0.5)

    def test_content_unavailable_leaves_collaborative_rankings_usable(self):
        model, _, _ = self.make_model(collaborative={10: collab_rows([(4, 4.7), (5, 4.2)])})
        response = model.recommend_for_user(10)
        self.assertEqual(response.status, "collaborative_only")
        self.assertEqual(response.recommendations.movie_id.tolist(), [4, 5])
        self.assertAlmostEqual(response.recommendations.iloc[0].fusion_score, 1.0)
        self.assertAlmostEqual(response.recommendations.iloc[1].fusion_score, 0.5)

    def test_known_user_without_eligible_sources_and_unknown_user(self):
        model, _, cf = self.make_model(collaborative={10: collab_rows([])})
        response = model.recommend_for_user(10)
        self.assertEqual(response.status, "no_eligible_evidence")
        self.assertEqual(model.recommend_for_user(999).status, "unknown_user")
        self.assertEqual(cf.calls[0][0], 10)

    def test_insufficient_history_uses_above_average_profile(self):
        interactions = pd.DataFrame([(30, 1, 5.0), (30, 2, 1.0)], columns=["user_id", "movie_id", "rating"])
        model, cold, cf = self.make_model(interactions, content={(1,): content_rows([(7, .8)])})
        response = model.recommend_for_user(30)
        self.assertEqual(response.status, "content_fallback_for_insufficient_collaborative_evidence")
        self.assertEqual(response.positive_seed_movie_ids, (1,))
        self.assertEqual(cf.calls[0][0], 30)
        self.assertEqual(cold.favorite_calls[0][0], (1,))

    def test_user_with_no_rating_variation_has_no_positive_seeds(self):
        interactions = pd.DataFrame([(30, 1, 4.0), (30, 2, 4.0)], columns=["user_id", "movie_id", "rating"])
        model, _, cf = self.make_model(interactions)
        response = model.recommend_for_user(30)
        self.assertEqual(response.status, "insufficient_history_and_no_positive_seeds")
        self.assertEqual(cf.calls, [])

    def test_movie_mode_delegates_to_phase11(self):
        model, _, _ = self.make_model()
        response = model.recommend_for_movie(1, top_k=2)
        self.assertEqual(response.recommendation_source, "phase9_content")
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            model.recommend_for_movie(999)

    def test_unique_results_top_k_ties_and_empty_candidates(self):
        content = {(1,): content_rows([(6, .9), (6, .8), (5, .7)])}
        collaborative = {10: collab_rows([(5, 4.5), (6, 4.5)])}
        model, _, _ = self.make_model(content=content, collaborative=collaborative)
        response = model.recommend_for_user(10, top_k=2)
        self.assertEqual(response.recommendations.movie_id.tolist(), [5, 6])
        self.assertTrue(response.recommendations.movie_id.is_unique)
        self.assertEqual(len(response.recommendations), 2)
        empty_model, _, _ = self.make_model(collaborative={10: collab_rows([])})
        self.assertTrue(empty_model.recommend_for_user(10).recommendations.empty)

    def test_invalid_top_k_and_interaction_rows_are_rejected(self):
        model, _, _ = self.make_model()
        with self.assertRaisesRegex(ValueError, "top_k"):
            model.recommend_for_user(10, 0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            model.recommend_for_new_user([], 0)
        broken = pd.DataFrame([(99, 9, 5.0)], columns=["user_id", "movie_id", "rating"])
        with self.assertRaisesRegex(ValueError, "absent from the supplied catalog"):
            self.make_model(broken)

    def test_rejects_collaborative_training_data_that_differs_from_supplied_history(self):
        model, cold, cf = self.make_model()
        cf.ratings = pd.DataFrame([(10, 7, 5.0)], columns=["user_id", "movie_id", "rating"])
        with self.assertRaisesRegex(ValueError, "same supplied ratings"):
            HybridRecommender(movies(), model.ratings, cold, cf)


if __name__ == "__main__":
    unittest.main()
