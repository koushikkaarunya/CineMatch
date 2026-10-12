import unittest
import warnings

import numpy as np
import pandas as pd

from scripts.phase14_2_story_relevance import (
    _metric_summary,
    _token_chunks,
    chunk_pair_scores,
    likely_person_name_terms,
    name_discounted_score,
    validate_pairs,
)


class Phase142AuditDataTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            {"movie_id": 1, "title": "Query", "plot": "Alex Morgan (Actor One) follows a mystery."},
            {"movie_id": 2, "title": "Candidate", "plot": "Alex Morgan (Actor Two) solves a comedy."},
            {"movie_id": 3, "title": "Other", "plot": "A detective follows a mystery."},
        ])
        self.pairs = pd.DataFrame([{
            "pair_id": "P142-test", "split": "development", "query_movie_id": 1,
            "query_title": "Query", "candidate_movie_id": 2,
            "candidate_title": "Candidate", "query_plot": self.catalog.iloc[0]["plot"],
            "candidate_plot": self.catalog.iloc[1]["plot"], "provisional_relevance": 0,
            "suggestion_status": "ai_assisted_suggestion",
            "label_provenance": "ai_assisted_suggestion",
            "provisional_rationale": "Same character name, different central stories.",
        }])

    def test_pair_requires_exact_catalog_plots_and_ai_provenance(self):
        validate_pairs(self.pairs, self.catalog)
        altered = self.pairs.copy()
        altered.loc[0, "candidate_plot"] = "Invented text."
        with self.assertRaisesRegex(ValueError, "exact candidate_plot"):
            validate_pairs(altered, self.catalog)

    def test_query_group_cannot_cross_split(self):
        second = self.pairs.copy()
        second["pair_id"] = "P142-test-2"
        second["candidate_movie_id"] = 3
        second["candidate_title"] = "Other"
        second["candidate_plot"] = self.catalog.iloc[2]["plot"]
        second["split"] = "locked_audit"
        with self.assertRaisesRegex(ValueError, "both development and locked audit"):
            validate_pairs(pd.concat([self.pairs, second], ignore_index=True), self.catalog)

    def test_uncertain_pairs_must_stay_unlabeled(self):
        flagged = self.pairs.copy()
        flagged.loc[0, "provisional_relevance"] = np.nan
        flagged.loc[0, "suggestion_status"] = "needs_review"
        flagged.loc[0, "label_provenance"] = "ai_assisted_needs_review"
        validate_pairs(flagged, self.catalog)
        flagged.loc[0, "provisional_relevance"] = 0
        with self.assertRaisesRegex(ValueError, "must remain unlabeled"):
            validate_pairs(flagged, self.catalog)


class Phase142ExperimentalScoreTests(unittest.TestCase):
    def test_name_discount_only_reduces_detected_name_unigrams(self):
        class TinyTfidf:
            movie_id_to_row = {1: 0, 2: 1}

            class Matrix:
                def __getitem__(self, index):
                    return np.array([[0.6, 0.8]]) if index == 0 else np.array([[0.6, 0.8]])

            tfidf_matrix = Matrix()

            class Vectorizer:
                vocabulary_ = {"alex": 0, "mystery": 1}

            vectorizer = Vectorizer()

        class SparseRow:
            def __init__(self, values):
                self.values = values

            def multiply(self, other):
                return SparseRow(self.values * other.values)

            def __getitem__(self, key):
                _, columns = key
                return SparseMatrixColumn(self.values[columns])

            def sum(self):
                return self.values.sum()

        class SparseMatrix:
            def __init__(self):
                self.rows = [SparseRow(np.array([0.6, 0.8])), SparseRow(np.array([0.6, 0.8]))]

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    row, columns = index
                    return SparseMatrixColumn(self.rows[row].values[columns])
                return self.rows[index]

        class SparseMatrixColumn:
            def __init__(self, values):
                self.values = values

            def multiply(self, other):
                return SparseMatrixColumn(self.values * other.values)

            def sum(self):
                return self.values.sum()

        model = TinyTfidf()
        model.tfidf_matrix = SparseMatrix()
        adjusted, name_overlap, names = name_discounted_score(
            model, 1, 2,
            "Alex Morgan (Actor One) investigates a mystery.",
            "Alex Morgan (Actor Two) investigates a mystery.",
            discount=0.8,
        )
        self.assertLess(adjusted, 1.0)
        self.assertAlmostEqual(name_overlap, 0.36)
        self.assertIn("alex", names)

    def test_chunk_pair_aggregation_compares_max_and_top_two(self):
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        candidate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        best, top_two = chunk_pair_scores(query, candidate)
        self.assertAlmostEqual(best, 1.0)
        self.assertAlmostEqual(top_two, 0.5)

    def test_long_plot_tokens_are_windowed_below_model_limit_without_warning(self):
        class Tokenizer:
            def num_special_tokens_to_add(self, pair=False):
                return 2

            def encode(self, text, *, add_special_tokens, verbose):
                self.assert_settings = (add_special_tokens, verbose)
                if verbose:
                    warnings.warn("generic tokenizer length warning")
                return list(range(1098))

        tokenizer = Tokenizer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunks, count = _token_chunks(tokenizer, "long", model_limit=256)
        self.assertEqual(caught, [])
        self.assertEqual(tokenizer.assert_settings, (False, False))
        self.assertEqual(count, 1098)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) + 2 <= 256 for chunk in chunks))

    def test_metrics_ignore_unjudged_rows_instead_of_counting_them_as_negatives(self):
        frame = pd.DataFrame({
            "query_movie_id": [1, 1, 1], "candidate_movie_id": [2, 3, 4],
            "provisional_relevance": [2, 0, np.nan], "score": [0.8, 0.2, 100.0],
        })
        result = _metric_summary(frame, "score")
        self.assertEqual(result["labeled_pairs"], 2)
        self.assertEqual(result["relevance_counts"], {"0": 1, "1": 0, "2": 1})
        self.assertEqual(result["pairwise_comparisons"], 1)


if __name__ == "__main__":
    unittest.main()
