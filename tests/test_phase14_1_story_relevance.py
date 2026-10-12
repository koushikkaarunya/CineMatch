import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.phase14_1_story_relevance import (
    _label_column_for_source,
    _plot_length_diagnostics,
    ranking_metrics,
    rerank_retrieved_candidates,
    validate_pairs,
)


class StoryRelevanceValidationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            {"movie_id": 1, "title": "Query", "plot": "A real query plot."},
            {"movie_id": 2, "title": "Candidate", "plot": "A real candidate plot."},
            {"movie_id": 3, "title": "Missing", "plot": pd.NA},
        ])
        self.columns = {
            "pair_id": ["SR001", "SR002"],
            "query_movie_id": [1, 1],
            "query_title": ["Query", "Query"],
            "candidate_movie_id": [2, 3],
            "candidate_title": ["Candidate", "Missing"],
            "query_plot": ["A real query plot.", "A real query plot."],
            "candidate_plot": ["A real candidate plot.", ""],
            "proposed_relevance": [0, pd.NA],
            "approved_relevance": [pd.NA, pd.NA],
            "label_status": ["ai_proposed", "unjudgeable_missing_plot"],
            "label_provenance": ["ai_proposed", "not_applicable_missing_plot"],
            "proposed_justification": ["Different story.", "No candidate plot."],
            "reviewer_notes": ["", ""],
        }

    def test_valid_pairs_keep_missing_plot_unjudged(self):
        validate_pairs(pd.DataFrame(self.columns), self.catalog)

    def test_plot_must_match_catalog_exactly_so_text_is_not_fabricated(self):
        pairs = pd.DataFrame(self.columns)
        pairs.loc[0, "candidate_plot"] = "An invented description."
        with self.assertRaisesRegex(ValueError, "fabricated candidate plot"):
            validate_pairs(pairs, self.catalog)

    def test_self_pair_and_duplicate_pair_are_rejected(self):
        pairs = pd.DataFrame(self.columns)
        pairs.loc[0, "candidate_movie_id"] = 1
        with self.assertRaisesRegex(ValueError, "query as its own candidate"):
            validate_pairs(pairs, self.catalog)
        pairs = pd.DataFrame(self.columns)
        pairs.loc[1, "candidate_movie_id"] = 2
        with self.assertRaisesRegex(ValueError, "repeated query/candidate"):
            validate_pairs(pairs, self.catalog)

    def test_approved_label_requires_explicit_ai_assisted_provenance(self):
        pairs = pd.DataFrame(self.columns)
        pairs.loc[0, "approved_relevance"] = 2
        with self.assertRaisesRegex(ValueError, "inconsistent AI-proposed label provenance"):
            validate_pairs(pairs, self.catalog)
        pairs.loc[0, "label_status"] = "user_approved_ai_assisted"
        pairs.loc[0, "label_provenance"] = "user_approved_ai_assisted"
        validate_pairs(pairs, self.catalog)

    def test_legacy_human_source_is_rejected_and_approved_source_is_explicit(self):
        self.assertEqual(_label_column_for_source("user_approved_ai_assisted"), "approved_relevance")
        with self.assertRaisesRegex(ValueError, "independent human judgments are not present"):
            _label_column_for_source("human")

    def test_full_plot_token_count_does_not_emit_model_length_warning(self):
        class Tokenizer:
            model_max_length = 512

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def encode(self, text, *, add_special_tokens, verbose):
                self.assertions = (add_special_tokens, verbose)
                if verbose:
                    warnings.warn("sequence length warning")
                return list(range(1098))

        tokenizer = Tokenizer()
        auto_tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        transformer = SimpleNamespace(plot_movies=pd.DataFrame({"movie_id": [5], "plot": ["long plot"]}))
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=auto_tokenizer)}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _plot_length_diagnostics(transformer, 256)
        self.assertEqual(caught, [])
        self.assertEqual(tokenizer.assertions, (False, False))
        self.assertEqual(result["sample_wordpiece_counts"]["5"]["wordpieces_including_special_tokens"], 1100)
        self.assertTrue(result["sample_wordpiece_counts"]["5"]["exceeds_model_input_limit"])


class StoryRankingMetricTests(unittest.TestCase):
    def test_pairwise_accuracy_and_graded_ndcg_match_hand_calculation(self):
        frame = pd.DataFrame({
            "query_movie_id": [1, 1, 1],
            "candidate_movie_id": [2, 3, 4],
            "relevance": [2, 1, 0],
            "score": [0.9, 0.1, 0.4],
        })
        result = ranking_metrics(frame, "score", "relevance")
        self.assertEqual(result["judged_pairs"], 3)
        self.assertEqual(result["pairwise_accuracy_grade2_vs_grade0"], 1.0)
        self.assertEqual(result["relevance_grade_counts"], {"0": 1, "1": 1, "2": 1})
        self.assertLess(result["ndcg_at_5"], 1.0)
        self.assertEqual(result["clearly_unrelated_in_top5"], 1)

    def test_tied_relevant_and_unrelated_scores_receive_half_pairwise_credit(self):
        frame = pd.DataFrame({
            "query_movie_id": [1, 1], "candidate_movie_id": [2, 3],
            "relevance": [2, 0], "score": [0.5, 0.5],
        })
        result = ranking_metrics(frame, "score", "relevance")
        self.assertEqual(result["pairwise_accuracy_grade2_vs_grade0"], 0.5)


class ExperimentalRerankerTests(unittest.TestCase):
    def test_reranker_preserves_shortlist_excludes_query_and_deduplicates(self):
        retrieved = pd.DataFrame({
            "movie_id": [1, 2, 2, 3], "hybrid_score": [1.0, 0.8, 0.7, 0.6],
        })
        reranked = rerank_retrieved_candidates(
            retrieved, {2: 0.2, 3: 0.9}, query_movie_id=1, top_k=5,
        )
        self.assertEqual(reranked.movie_id.tolist(), [3, 2])
        self.assertEqual(set(reranked.movie_id), {2, 3})
        self.assertTrue(reranked.movie_id.is_unique)
        self.assertNotIn(1, reranked.movie_id.tolist())

    def test_missing_plot_scores_have_no_fabricated_similarity_and_sort_last(self):
        retrieved = pd.DataFrame({
            "movie_id": [2, 3], "hybrid_score": [0.9, 0.5],
        })
        reranked = rerank_retrieved_candidates(
            retrieved, {2: 0.1}, query_movie_id=1,
        )
        self.assertEqual(reranked.movie_id.tolist(), [2, 3])
        self.assertTrue(pd.isna(reranked.loc[reranked.movie_id.eq(3), "chunked_similarity"]).all())

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            rerank_retrieved_candidates(pd.DataFrame({"movie_id": [], "hybrid_score": []}), {},
                                        query_movie_id=1, top_k=0)


if __name__ == "__main__":
    unittest.main()
