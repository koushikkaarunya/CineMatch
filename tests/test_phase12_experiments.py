import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.phase9_diagnostics import chunk_token_ids, normalized_mean, plot_fingerprint, load_or_build_chunked_embeddings
from scripts.phase12_experiments import (
    mask_diagnostic_names,
    reciprocal_rank_fusion,
)
from scripts.phase12_experiments import name_similarity_contribution
from scripts.recommend_tfidf import TfidfMovieRecommender


class Phase12ExperimentTests(unittest.TestCase):
    def test_name_ablation_is_narrow_and_preserves_other_plot_text(self):
        plot = "Robin Hood meets Robin; a robin lands nearby, and Maximus travels on."
        masked = mask_diagnostic_names(plot)
        self.assertNotIn("Robin", masked)
        self.assertIn("Hood", masked)
        self.assertIn("Maximus", masked)
        self.assertIn("a lands", masked)
        self.assertIn("meets", masked)
        self.assertTrue(pd.isna(mask_diagnostic_names(pd.NA)))
        self.assertEqual(mask_diagnostic_names("   "), "   ")

    def test_name_ablation_can_remove_a_genuinely_useful_name_signal(self):
        movies = pd.DataFrame([
            {"movie_id": 1, "title": "Query", "release_year": 2000,
             "plot": "Robin investigates a murder in London."},
            {"movie_id": 2, "title": "Same Character Story", "release_year": 2001,
             "plot": "Robin investigates a murder in London with a partner."},
            {"movie_id": 3, "title": "Same Name, Other Story", "release_year": 2002,
             "plot": "Robin discovers a hidden garden on Mars."},
        ])
        baseline = TfidfMovieRecommender(movies)
        name_ablation = movies.copy()
        name_ablation["plot"] = name_ablation["plot"].map(mask_diagnostic_names)
        modified = TfidfMovieRecommender(name_ablation)
        baseline_score = float(baseline.recommend(1, top_k=2).set_index("movie_id").loc[2, "similarity"])
        modified_score = float(modified.recommend(1, top_k=2).set_index("movie_id").loc[2, "similarity"])
        self.assertGreater(name_similarity_contribution(baseline, 1, 2), 0.0)
        self.assertGreater(baseline_score, modified_score)

    def test_rrf_combines_common_and_single_source_candidates_deterministically(self):
        fused = reciprocal_rank_fusion(
            {"cf": [8, 9], "content": [9, 10]},
            available_sources=("cf", "content"), top_k=3,
        )
        by_id = {row["movie_id"]: row for row in fused}
        self.assertEqual(fused[0]["movie_id"], 9)
        self.assertAlmostEqual(by_id[9]["fusion_score"], .75)
        self.assertAlmostEqual(by_id[8]["fusion_score"], .5)
        self.assertAlmostEqual(by_id[10]["fusion_score"], .25)

    def test_rrf_uses_only_available_sources_and_deduplicates_each_list(self):
        fused = reciprocal_rank_fusion(
            {"content": [4, 4, 5]}, available_sources=("content",), top_k=5,
        )
        self.assertEqual([(row["movie_id"], row["fusion_score"]) for row in fused], [(4, 1.0), (5, .5)])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({}, available_sources=())

    def test_chunk_windows_cover_all_tokens_with_expected_overlap(self):
        chunks = chunk_token_ids(list(range(10)), chunk_size=4, overlap=1)
        self.assertEqual(chunks, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]])
        self.assertEqual(chunk_token_ids([1, 2], chunk_size=4, overlap=1), [[1, 2]])
        for chunk_size, overlap in ((0, 0), (4, 4), (4, -1)):
            with self.assertRaises(ValueError):
                chunk_token_ids([1], chunk_size=chunk_size, overlap=overlap)

    def test_chunk_vectors_are_averaged_then_normalized(self):
        result = normalized_mean(np.array([[1, 0], [0, 1]], dtype=np.float32))
        np.testing.assert_allclose(result, [2 ** -.5, 2 ** -.5])
        with self.assertRaises(ValueError):
            normalized_mean(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            normalized_mean(np.array([]))

    def test_valid_chunk_artifact_is_reused_without_encoding(self):
        plots = pd.DataFrame({"movie_id": [1, 2], "plot": ["short plot", "another plot"]})
        base = np.zeros((2, 384), dtype=np.float32)
        base[:, 0] = 1.0
        transformer = SimpleNamespace(
            plot_movies=plots,
            embeddings=base,
            model_id="test/model",
            model_revision="revision",
        )

        class Tokenizer:
            def num_special_tokens_to_add(self, pair=False):
                return 2

        class Model:
            max_seq_length = 256

            def _first_module(self):
                return SimpleNamespace(tokenizer=Tokenizer())

            def encode(self, *args, **kwargs):
                raise AssertionError("A valid cached artifact should avoid re-encoding")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunked.npz"
            metadata = {
                "model_id": transformer.model_id,
                "model_revision": transformer.model_revision,
                "plot_fingerprint": plot_fingerprint(plots),
                "chunk_size": 224,
                "overlap": 48,
                "max_sequence_length": 256,
                "artifact_version": 1,
                "embedding_dimension": 384,
                "normalized": True,
            }
            np.savez_compressed(
                path,
                metadata=np.array(json.dumps(metadata)),
                movie_ids=plots.movie_id.to_numpy(dtype=np.int64),
                embeddings=base,
            )
            vectors, returned_metadata = load_or_build_chunked_embeddings(transformer, Model(), path=path)
        np.testing.assert_array_equal(vectors, base)
        self.assertTrue(returned_metadata["reused"])

    def test_long_plot_is_chunked_below_model_limit_without_length_warning(self):
        plots = pd.DataFrame({"movie_id": [5], "plot": ["long plot text"]})
        base = np.zeros((1, 384), dtype=np.float32)
        base[0, 0] = 1.0
        transformer = SimpleNamespace(
            plot_movies=plots,
            embeddings=base,
            model_id="test/model",
            model_revision="revision",
        )

        class Tokenizer:
            def __init__(self):
                self.decoded_lengths = {}
                self.encode_options = None

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def encode(self, text, *, add_special_tokens, verbose):
                self.encode_options = (add_special_tokens, verbose)
                self.assert_no_advisory = not verbose
                return list(range(600))

            def decode(self, tokens, skip_special_tokens=True):
                value = f"window-{len(tokens)}"
                self.decoded_lengths[value] = len(tokens)
                return value

        tokenizer = Tokenizer()

        class Model:
            max_seq_length = 256

            def _first_module(self):
                return SimpleNamespace(tokenizer=tokenizer)

            def encode(self, texts, **kwargs):
                self.encoded_texts = list(texts)
                return np.tile(np.eye(1, 384, 0, dtype=np.float32), (len(texts), 1))

        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            vectors, metadata = load_or_build_chunked_embeddings(
                transformer, model, path=Path(directory) / "long.npz",
            )

        self.assertEqual(tokenizer.encode_options, (False, False))
        self.assertTrue(tokenizer.assert_no_advisory)
        self.assertEqual(metadata["max_sequence_length"], 256)
        self.assertEqual(metadata["chunk_size"], 224)
        self.assertEqual(metadata["chunk_count"], len(model.encoded_texts))
        self.assertTrue(model.encoded_texts)
        self.assertNotIn("long plot text", model.encoded_texts)
        self.assertLessEqual(
            max(tokenizer.decoded_lengths[text] for text in model.encoded_texts) + 2,
            model.max_seq_length,
        )
        self.assertEqual(vectors.shape, (1, 384))
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
