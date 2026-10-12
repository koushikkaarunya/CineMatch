"""Targeted, baseline-preserving diagnostics for known Phase 6–7 weaknesses."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recommend_metadata import prepare_catalog, recommend_similar_movies
from scripts.recommend_tfidf import TfidfMovieRecommender, load_movies as load_clean_movies
from scripts.recommend_transformer import TransformerMovieRecommender, load_movies as load_transformer_movies
from scripts.recommend_hybrid_content import HybridContentRecommender, load_enriched_movies


PHASE9_REPORT = PROJECT_ROOT / "reports" / "phase9_hybrid_diagnostics.json"
ORIGINAL_EMBEDDINGS = PROJECT_ROOT / "data" / "models" / "all_minilm_l6_v2_plots.npz"
CHUNKED_EMBEDDINGS = PROJECT_ROOT / "data" / "models" / "all_minilm_l6_v2_plots_chunked.npz"
PHASE12_REPORT = PROJECT_ROOT / "reports" / "phase12_targeted_diagnostics.json"
KNOWN_CHARACTER_NAMES = ("Max", "Robin", "Savannah")
KNOWN_QUERY_IDS = (3, 4, 5)


def mask_diagnostic_names(plot: object, names: tuple[str, ...] = KNOWN_CHARACTER_NAMES) -> object:
    """Ablate only explicitly named diagnostic tokens; never used by production models."""
    if not isinstance(plot, str) or not plot.strip():
        return plot
    masked = plot
    for name in names:
        if not name or not name.strip():
            raise ValueError("names must contain non-empty strings")
        masked = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip()


def name_similarity_contribution(
    recommender: TfidfMovieRecommender,
    query_movie_id: int,
    candidate_movie_id: int,
    names: tuple[str, ...] = KNOWN_CHARACTER_NAMES,
) -> float:
    """Sum normalized TF-IDF cosine contributions from selected name unigrams."""
    query_row = recommender.movie_id_to_row[query_movie_id]
    candidate_row = recommender.movie_id_to_row[candidate_movie_id]
    query_vector = recommender.tfidf_matrix.getrow(query_row)
    candidate_vector = recommender.tfidf_matrix.getrow(candidate_row)
    query_values = dict(zip(query_vector.indices, query_vector.data))
    candidate_values = dict(zip(candidate_vector.indices, candidate_vector.data))
    contribution = 0.0
    for name in names:
        term_id = recommender.vectorizer.vocabulary_.get(name.lower())
        if term_id is not None:
            contribution += query_values.get(term_id, 0.0) * candidate_values.get(term_id, 0.0)
    return float(contribution)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]], *, available_sources: tuple[str, ...], top_k: int = 5
) -> list[dict[str, object]]:
    """Fuse ordinal positions; unavailable lists do not dilute available sources."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not available_sources or any(source not in ranked_lists for source in available_sources):
        raise ValueError("Every available source must have a ranked list")
    ranks: dict[int, dict[str, int]] = {}
    for source in available_sources:
        seen: set[int] = set()
        unique_ids = []
        for raw_id in ranked_lists[source]:
            movie_id = int(raw_id)
            if movie_id in seen:
                continue
            seen.add(movie_id)
            unique_ids.append(movie_id)
        for rank, movie_id in enumerate(unique_ids, start=1):
            ranks.setdefault(movie_id, {})[source] = rank
    output = []
    for movie_id, source_ranks in ranks.items():
        score = sum(1.0 / source_ranks[source] for source in available_sources if source in source_ranks)
        score /= len(available_sources)
        output.append({"movie_id": movie_id, "fusion_score": score, "source_ranks": source_ranks})
    return sorted(output, key=lambda row: (-row["fusion_score"], row["movie_id"]))[:top_k]


def _records(frame: pd.DataFrame, score_col: str) -> list[dict[str, object]]:
    return [
        {"movie_id": int(row.movie_id), "title": str(row.title), "score": float(getattr(row, score_col))}
        for row in frame.itertuples(index=False)
    ]


def _tfidf_ablation(clean_movies: pd.DataFrame, query_ids: tuple[int, ...]) -> list[dict[str, object]]:
    baseline = TfidfMovieRecommender(clean_movies)
    variant_movies = clean_movies.copy()
    variant_movies["plot"] = variant_movies["plot"].map(mask_diagnostic_names)
    variant = TfidfMovieRecommender(variant_movies)
    comparisons = []
    for query_id in query_ids:
        original = baseline.recommend(query_id, top_k=5)
        ablated = variant.recommend(query_id, top_k=5)
        name_contributions = []
        for candidate_id in original.movie_id.astype(int):
            contribution = name_similarity_contribution(baseline, query_id, candidate_id)
            similarity = float(original.loc[original.movie_id.eq(candidate_id), "similarity"].iloc[0])
            name_contributions.append({
                "movie_id": candidate_id,
                "name_unigram_cosine_contribution": contribution,
                "share_of_cosine": contribution / similarity if similarity > 0 else 0.0,
            })
        comparisons.append({
            "query_movie_id": query_id,
            "baseline_top5": _records(original, "similarity"),
            "name_masked_top5": _records(ablated, "similarity"),
            "baseline_name_contributions": name_contributions,
            "top5_overlap_count": len(set(original.movie_id) & set(ablated.movie_id)),
            "note": "Diagnostic ablation only: the Phase 6 recommender and its default behavior are unchanged.",
        })
    return comparisons


def _transformer_reranking(
    clean_movies: pd.DataFrame, enriched_movies: pd.DataFrame, query_ids: tuple[int, ...]
) -> list[dict[str, object]]:
    transformer = TransformerMovieRecommender(clean_movies, cache_path=ORIGINAL_EMBEDDINGS)
    lexical = TfidfMovieRecommender(clean_movies)
    phase9 = HybridContentRecommender(
        enriched_movies, tfidf_model=lexical, transformer_model=transformer
    )
    metadata_catalog = prepare_catalog(enriched_movies)
    reports = []
    for query_id in query_ids:
        semantic = transformer.recommend(query_id, top_k=20)
        metadata = recommend_similar_movies(metadata_catalog, query_id, top_k=20)
        phase9_results = phase9.recommend(query_id, top_k=5)
        fused = reciprocal_rank_fusion({
            "transformer": semantic.movie_id.astype(int).tolist(),
            "metadata": metadata.movie_id.astype(int).tolist(),
        }, available_sources=("transformer", "metadata"), top_k=5)
        title_by_id = dict(zip(clean_movies.movie_id.astype(int), clean_movies.title.astype(str)))
        reports.append({
            "query_movie_id": query_id,
            "phase7_transformer_top5": _records(semantic.head(5), "similarity"),
            "phase5_metadata_top5": _records(metadata.head(5), "similarity"),
            "phase9_existing_content_rerank_top5": _records(phase9_results, "hybrid_score"),
            "transformer_metadata_rrf_top5": [
                {**row, "title": title_by_id.get(row["movie_id"]),
                 "source_titles": {source: title_by_id.get(movie_id) for source, movie_id in row["source_ranks"].items()}}
                for row in fused
            ],
            "top5_overlap_with_phase7": len(set(semantic.head(5).movie_id) & {row["movie_id"] for row in fused}),
            "note": "Diagnostic RRF alternative; it does not replace Phase 7 or change Phase 9 weights.",
        })
    return reports


def _chunked_comparisons(query_ids: tuple[int, ...]) -> dict[str, object]:
    with np.load(ORIGINAL_EMBEDDINGS, allow_pickle=False) as artifact:
        original_metadata = json.loads(str(artifact["metadata"].item()))
        original_ids = artifact["movie_ids"].astype(np.int64)
        original_vectors = artifact["embeddings"].astype(np.float32)
    with np.load(CHUNKED_EMBEDDINGS, allow_pickle=False) as artifact:
        chunk_metadata = json.loads(str(artifact["metadata"].item()))
        chunk_ids = artifact["movie_ids"].astype(np.int64)
        chunk_vectors = artifact["embeddings"].astype(np.float32)
    required_equal = ("model_id", "model_revision", "plot_fingerprint")
    if any(original_metadata.get(key) != chunk_metadata.get(key) for key in required_equal):
        raise ValueError("Original and chunked embedding artifacts do not describe the same model and plots")
    if not np.array_equal(original_ids, chunk_ids) or original_vectors.shape != chunk_vectors.shape:
        raise ValueError("Original and chunked embedding IDs or vector shapes do not match")
    if not np.isfinite(chunk_vectors).all() or not np.allclose(np.linalg.norm(chunk_vectors, axis=1), 1, atol=1e-3):
        raise ValueError("Chunked embedding artifact has invalid vectors")

    clean_movies = load_transformer_movies()
    titles = dict(zip(clean_movies.movie_id.astype(int), clean_movies.title.astype(str)))
    id_to_row = {int(movie_id): row for row, movie_id in enumerate(original_ids)}
    comparisons = []
    for query_id in query_ids:
        if query_id not in id_to_row:
            continue
        row = id_to_row[query_id]
        base_scores = original_vectors @ original_vectors[row]
        chunk_scores = chunk_vectors @ chunk_vectors[row]
        base_scores[row] = -1
        chunk_scores[row] = -1
        base_rows = np.argsort(-base_scores, kind="stable")[:5]
        chunk_rows = np.argsort(-chunk_scores, kind="stable")[:5]
        base_ids = [int(original_ids[index]) for index in base_rows]
        new_ids = [int(chunk_ids[index]) for index in chunk_rows]
        comparisons.append({
            "query_movie_id": query_id,
            "title": titles.get(query_id),
            "plot_vector_unchanged": bool(np.array_equal(original_vectors[row], chunk_vectors[row])),
            "phase7_single_input_top5": [
                {"movie_id": movie_id, "title": titles.get(movie_id), "cosine": float(base_scores[id_to_row[movie_id]])}
                for movie_id in base_ids
            ],
            "chunked_top5": [
                {"movie_id": movie_id, "title": titles.get(movie_id), "cosine": float(chunk_scores[id_to_row[movie_id]])}
                for movie_id in new_ids
            ],
            "top5_overlap_count": len(set(base_ids) & set(new_ids)),
        })
    return {
        "model_id": chunk_metadata["model_id"],
        "model_revision": chunk_metadata["model_revision"],
        "cache_validated_and_reused": True,
        "max_sequence_length": chunk_metadata["max_sequence_length"],
        "chunk_size_wordpieces": chunk_metadata["chunk_size"],
        "overlap_wordpieces": chunk_metadata["overlap"],
        "long_plot_count": chunk_metadata["long_plot_count"],
        "chunks_encoded": chunk_metadata["chunk_count"],
        "short_plot_vectors_reused": chunk_metadata["short_plot_vectors_reused_from_phase7"],
        "representative_comparisons": comparisons,
    }


def run_diagnostics() -> dict[str, object]:
    """Compare unchanged baselines with narrow diagnostic alternatives."""
    clean_movies = load_clean_movies()
    enriched_movies = load_enriched_movies()
    previous = json.loads(PHASE9_REPORT.read_text())
    long_ids = tuple(
        int(row["movie_id"])
        for row in previous["chunked_plot_experiment"]["representative_long_plots"]
    )
    with np.load(ORIGINAL_EMBEDDINGS, allow_pickle=False) as original_artifact, \
         np.load(CHUNKED_EMBEDDINGS, allow_pickle=False) as chunked_artifact:
        short_id = next(
            int(movie_id) for movie_id, original, chunked in zip(
                original_artifact["movie_ids"], original_artifact["embeddings"],
                chunked_artifact["embeddings"],
            ) if np.array_equal(original, chunked)
        )
    chunked = _chunked_comparisons((short_id, *long_ids))
    results = {
        "phase": "Phase 12 targeted diagnostics",
        "scope_note": "Qualitative, baseline-preserving experiments; not held-out evaluation.",
        "tfidf_character_name_ablation": _tfidf_ablation(clean_movies, KNOWN_QUERY_IDS),
        "transformer_metadata_reranking": _transformer_reranking(clean_movies, enriched_movies, KNOWN_QUERY_IDS),
        "long_plot_chunking": chunked,
        "chosen_short_plot_id": short_id,
        "baseline_preservation": {
            "phase6_tfidf_default_unchanged": True,
            "phase7_transformer_unchanged": True,
            "phase9_weights_unchanged": True,
            "phase10_algorithm_unchanged": True,
        },
    }
    PHASE12_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE12_REPORT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return results


def main() -> None:
    report = run_diagnostics()
    print(json.dumps({
        "report_path": str(PHASE12_REPORT),
        "character_name_queries": len(report["tfidf_character_name_ablation"]),
        "transformer_queries": len(report["transformer_metadata_reranking"]),
        "chunked_plot_cases": len(report["long_plot_chunking"]["representative_comparisons"]),
        "chunk_cache_reused": report["long_plot_chunking"]["cache_validated_and_reused"],
    }, indent=2))


if __name__ == "__main__":
    main()
