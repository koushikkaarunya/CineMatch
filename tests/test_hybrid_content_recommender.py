import json
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scripts.recommend_hybrid_content import (
    DEFAULT_WEIGHTS,
    HybridContentRecommender,
    _rank_percentiles,
    combine_rank_scores,
    parse_people,
    pair_metadata_similarity,
)
from scripts.phase9_diagnostics import chunk_token_ids, normalized_mean


def person(qid: str, name: str, role: str | None = None) -> str:
    item = {"wikidata_id": qid, "name": name}
    if role:
        item["role"] = role
    return json.dumps([item])


def sample_catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {"movie_id": 1, "title": "Space Rescue", "release_year": 2000,
         "genres": "Adventure|Sci-Fi", "plot_tags": "rescue, space",
         "plot": "A pilot rescues colonists in deep space.",
         "directors": person("Q1", "A Director"), "writers": pd.NA,
         "cast": person("Q4", "An Actor"), "selected_crew": pd.NA},
        {"movie_id": 2, "title": "Colony Mission", "release_year": 2001,
         "genres": "Adventure|Sci-Fi", "plot_tags": "rescue, colony",
         "plot": "A space pilot saves a colony from danger.",
         "directors": person("Q1", "A Director"), "writers": pd.NA,
         "cast": pd.NA, "selected_crew": person("Q8", "A Producer", "producer")},
        {"movie_id": 3, "title": "City Mystery", "release_year": 2002,
         "genres": "Crime|Thriller", "plot_tags": "murder, mystery",
         "plot": "A detective investigates a murder in the city.",
         "directors": person("Q7", "Another Director"), "writers": pd.NA,
         "cast": person("Q4", "An Actor"), "selected_crew": pd.NA},
        {"movie_id": 4, "title": "Rescue Without Plot", "release_year": 2003,
         "genres": "Adventure|Sci-Fi", "plot_tags": "rescue",
         "plot": pd.NA, "directors": person("Q1", "A Director"),
         "writers": pd.NA, "cast": pd.NA, "selected_crew": pd.NA},
        {"movie_id": 5, "title": "Metadata Missing", "release_year": 2004,
         "genres": "(no genres listed)", "plot_tags": pd.NA, "plot": pd.NA,
         "directors": pd.NA, "writers": pd.NA, "cast": pd.NA, "selected_crew": pd.NA},
    ])


class FakeTextModel:
    def __init__(self, movie_ids: list[int], vectors: list[list[float]]) -> None:
        self.movie_id_to_row = {movie_id: row for row, movie_id in enumerate(movie_ids)}
        self.catalog_ids = {1, 2, 3, 4, 5}
        values = np.asarray(vectors, dtype=np.float32)
        self.embeddings = values / np.linalg.norm(values, axis=1, keepdims=True)
        self.tfidf_matrix = self.embeddings


class HybridContentTests(unittest.TestCase):
    def setUp(self) -> None:
        # The text-model row order deliberately differs from the movie catalog order.
        ids = [2, 1, 3]
        vectors = [[0.8, 0.6], [1.0, 0.0], [0.0, 1.0]]
        self.tfidf = FakeTextModel(ids, vectors)
        self.tfidf.tfidf_matrix = csr_matrix(self.tfidf.tfidf_matrix)
        self.transformer = FakeTextModel(ids, [[0.75, 0.6614378], [1.0, 0.0], [0.0, 1.0]])
        self.recommender = HybridContentRecommender(
            sample_catalog(), tfidf_model=self.tfidf, transformer_model=self.transformer
        )

    def test_rank_fusion_normalizes_scales_and_penalizes_sparse_evidence(self) -> None:
        percentiles = _rank_percentiles({1: 0.001, 2: 0.99, 3: 0.5})
        self.assertLess(percentiles[1], percentiles[3])
        self.assertLess(percentiles[3], percentiles[2])
        score, coverage = combine_rank_scores(
            {"metadata": 0.8}, {"metadata"}, {"metadata": 0.4, "text": 0.6}, 1.0
        )
        self.assertAlmostEqual(score, 0.32)
        self.assertAlmostEqual(coverage, 0.4)

    def test_hybrid_ranks_supported_candidate_and_explains_real_evidence(self) -> None:
        results = self.recommender.recommend(1, top_k=5)
        self.assertEqual(results.iloc[0].movie_id, 2)
        self.assertNotIn(1, results.movie_id.tolist())
        self.assertTrue(results.movie_id.is_unique)
        explanation = results.iloc[0].explanation
        self.assertIn("shared genres: adventure, sci-fi", explanation)
        self.assertIn("shared director: A Director", explanation)
        self.assertIn("plot word overlap", explanation)
        self.assertGreaterEqual(results.iloc[0].positive_signals, 2)

    def test_metadata_signal_preserves_phase5_genre_tag_formula(self) -> None:
        scores = self.recommender._component_scores(0)
        self.assertAlmostEqual(scores["genre"][2], 1.0)
        self.assertAlmostEqual(scores["tags"][2], 1 / 3)
        self.assertAlmostEqual(scores["metadata"][2], 0.8)
        query = self.recommender.catalog.iloc[0]
        candidate = self.recommender.catalog.iloc[1].copy()
        candidate["tag_labels"] = set()
        self.assertAlmostEqual(pair_metadata_similarity(query, candidate), 1.0)

    def test_sparse_candidate_without_plots_can_be_recommended_from_metadata(self) -> None:
        results = self.recommender.recommend(4, top_k=5)
        self.assertIn(2, results.movie_id.tolist())
        row = results.loc[results.movie_id.eq(2)].iloc[0]
        self.assertTrue(pd.isna(row.tfidf_similarity))
        self.assertTrue(pd.isna(row.transformer_similarity))
        self.assertGreater(row.evidence_coverage, 0)

    def test_candidate_with_only_one_positive_signal_is_not_returned(self) -> None:
        # Movie 3 shares an actor but has no positive genre, tag, or plot match.
        results = self.recommender.recommend(1, top_k=10)
        self.assertNotIn(3, results.movie_id.tolist())

    def test_id_to_vector_mapping_is_used(self) -> None:
        scores = self.recommender._component_scores(0)
        self.assertAlmostEqual(scores["tfidf"][2], 0.8, places=5)
        self.assertAlmostEqual(scores["transformer"][2], 0.75, places=5)
        self.assertAlmostEqual(scores["tfidf"][3], 0.0, places=5)

    def test_ratings_column_is_ignored(self) -> None:
        original = self.recommender.recommend(1, top_k=5)
        movies = sample_catalog()
        movies["rating"] = [1, 5, 2, 4, 3]
        with_ratings = HybridContentRecommender(
            movies, tfidf_model=self.tfidf, transformer_model=self.transformer
        ).recommend(1, top_k=5)
        self.assertEqual(original.movie_id.tolist(), with_ratings.movie_id.tolist())
        np.testing.assert_allclose(original.hybrid_score, with_ratings.hybrid_score)

    def test_missing_data_unknown_id_invalid_top_k_and_empty_query(self) -> None:
        self.assertTrue(self.recommender.recommend(5).empty)
        with self.assertRaisesRegex(KeyError, "Unknown movie_id"):
            self.recommender.recommend(999)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "top_k"):
                self.recommender.recommend(1, invalid)
        with self.assertRaisesRegex(ValueError, "movie_id must be an integer"):
            self.recommender.recommend("1")

    def test_duplicate_ids_and_invalid_weights_are_rejected(self) -> None:
        duplicate = pd.concat([sample_catalog(), sample_catalog().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate movie_id"):
            HybridContentRecommender(duplicate, tfidf_model=self.tfidf, transformer_model=self.transformer)
        bad_weights = dict(DEFAULT_WEIGHTS)
        bad_weights["metadata"] = 0.5
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            HybridContentRecommender(
                sample_catalog(), tfidf_model=self.tfidf,
                transformer_model=self.transformer, weights=bad_weights,
            )

    def test_structured_identity_uses_qids_and_crew_roles(self) -> None:
        people, names = parse_people(person("Q1", "A Director"))
        self.assertEqual(people, {"Q1"})
        self.assertEqual(names["Q1"], "A Director")
        crew_ids, _ = parse_people(person("Q8", "A Producer", "producer"), crew=True)
        self.assertEqual(crew_ids, {"producer:Q8"})

    def test_chunk_windows_overlap_and_combined_vector_is_normalized(self) -> None:
        chunks = chunk_token_ids(list(range(10)), chunk_size=5, overlap=2)
        self.assertEqual(chunks, [list(range(5)), list(range(3, 8)), [6, 7, 8, 9]])
        combined = normalized_mean(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(combined, [2 ** -0.5, 2 ** -0.5], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_token_ids([1, 2], chunk_size=2, overlap=2)


if __name__ == "__main__":
    unittest.main()
