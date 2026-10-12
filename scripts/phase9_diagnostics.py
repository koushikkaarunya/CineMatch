"""Run repeatable Phase 9 comparisons and the separate chunked-plot experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Let this file work both as `python scripts/...py` and `python -m scripts...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recommend_hybrid_content import (
    DEFAULT_WEIGHTS,
    CHUNKED_EMBEDDINGS_PATH,
    HybridContentRecommender,
    load_enriched_movies,
)
from scripts.recommend_metadata import load_catalog as load_metadata_catalog
from scripts.recommend_metadata import recommend_similar_movies
from scripts.recommend_tfidf import TfidfMovieRecommender, load_movies as load_tfidf_movies
from scripts.recommend_transformer import (
    EMBEDDINGS_PATH,
    MODEL_ID,
    MODEL_REVISION,
    TransformerMovieRecommender,
    load_sentence_model,
    plot_fingerprint,
    usable_plot_mask,
)


REPORT_PATH = PROJECT_ROOT / "reports" / "phase9_hybrid_diagnostics.json"
KNOWN_FAILURE_QUERIES = {3: "Grumpier Old Men (1995)", 4: "Waiting to Exhale (1995)",
                         5: "Father of the Bride Part II (1995)"}
DIAGNOSTIC_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def chunk_token_ids(token_ids: list[int], chunk_size: int = 224, overlap: int = 48) -> list[list[int]]:
    """Split token IDs into overlapping windows, validating the window settings."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    if len(token_ids) <= chunk_size:
        return [token_ids]
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(token_ids):
        chunks.append(token_ids[start:start + chunk_size])
        if start + chunk_size >= len(token_ids):
            break
        start += step
    return chunks


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    """Average chunk vectors and re-normalize the result for cosine comparison."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("vectors must be a non-empty two-dimensional array")
    mean_vector = values.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("mean chunk vector cannot be normalized")
    return mean_vector / norm


def _tokenizer_from_model(model):
    module = model._first_module()
    return module.tokenizer


def load_or_build_chunked_embeddings(
    transformer: TransformerMovieRecommender,
    model,
    *,
    path: Path = CHUNKED_EMBEDDINGS_PATH,
    chunk_size: int = 224,
    overlap: int = 48,
) -> tuple[np.ndarray, dict]:
    """Reuse valid experimental vectors or chunk only plots exceeding Phase 7's limit."""
    plot_movies = transformer.plot_movies
    tokenizer = _tokenizer_from_model(model)
    max_sequence_length = int(model.max_seq_length)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    safe_chunk_size = min(chunk_size, max_sequence_length - special_tokens)
    fingerprint = plot_fingerprint(plot_movies)
    expected = {
        "model_id": transformer.model_id,
        "model_revision": transformer.model_revision,
        "plot_fingerprint": fingerprint,
        "chunk_size": safe_chunk_size,
        "overlap": overlap,
        "max_sequence_length": max_sequence_length,
    }
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as artifact:
                metadata = json.loads(str(artifact["metadata"].item()))
                ids = artifact["movie_ids"].astype(np.int64)
                vectors = artifact["embeddings"].astype(np.float32)
            if (all(metadata.get(key) == value for key, value in expected.items())
                    and np.array_equal(ids, plot_movies["movie_id"].to_numpy(dtype=np.int64))
                    and vectors.shape == transformer.embeddings.shape
                    and np.isfinite(vectors).all()
                    and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)):
                metadata["reused"] = True
                return vectors, metadata
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    # These IDs are immediately split into safe windows before any model call.
    # Keep the full token sequence but suppress the tokenizer's generic
    # overlength warning, which describes a later model-forward risk rather
    # than this intentional chunking step.
    tokenized_plots = [tokenizer.encode(
        str(plot), add_special_tokens=False, verbose=False,
    )
                       for plot in plot_movies["plot"]]
    token_limit = max_sequence_length - special_tokens
    long_positions = [position for position, tokens in enumerate(tokenized_plots)
                      if len(tokens) > token_limit]
    chunk_texts: list[str] = []
    chunk_ranges: dict[int, tuple[int, int]] = {}
    for position in long_positions:
        windows = chunk_token_ids(tokenized_plots[position], safe_chunk_size, overlap)
        first = len(chunk_texts)
        chunk_texts.extend(tokenizer.decode(window, skip_special_tokens=True) for window in windows)
        chunk_ranges[position] = (first, len(chunk_texts))

    vectors = transformer.embeddings.copy()
    if chunk_texts:
        chunk_vectors = model.encode(
            chunk_texts, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        for position, (start, end) in chunk_ranges.items():
            vectors[position] = normalized_mean(chunk_vectors[start:end])

    metadata = {
        **expected,
        "artifact_version": 1,
        "embedding_dimension": int(vectors.shape[1]),
        "normalized": True,
        "long_plot_count": len(long_positions),
        "chunk_count": len(chunk_texts),
        "short_plot_vectors_reused_from_phase7": len(plot_movies) - len(long_positions),
        "reused": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary, metadata=np.array(json.dumps(metadata)),
        movie_ids=plot_movies["movie_id"].to_numpy(dtype=np.int64), embeddings=vectors,
    )
    temporary.replace(path)
    return vectors, metadata


def _result_records(results: pd.DataFrame, score_column: str) -> list[dict]:
    records = []
    for row in results.to_dict(orient="records"):
        item = {}
        for key, value in row.items():
            if pd.isna(value):
                item[key] = None
            elif isinstance(value, np.generic):
                item[key] = value.item()
            else:
                item[key] = value
        records.append(item)
    return records


def _query_title(catalog: pd.DataFrame, movie_id: int) -> str:
    return str(catalog.loc[catalog.movie_id.eq(movie_id), "title"].iloc[0])


def _broad_query_ids(catalog: pd.DataFrame) -> dict[int, list[str]]:
    """Choose first plot-bearing catalog movie for each listed genre, by movie_id."""
    queries: dict[int, list[str]] = {}
    has_plot = usable_plot_mask(catalog)
    for genre in DIAGNOSTIC_GENRES:
        matching = catalog.loc[
            has_plot & catalog["genres"].fillna("").str.split("|").map(
                lambda labels: any(label.casefold() == genre.casefold() for label in labels)
            )
        ].sort_values("movie_id")
        if not matching.empty:
            movie_id = int(matching.iloc[0]["movie_id"])
            queries.setdefault(movie_id, []).append(genre)
    return queries


def _run_baselines(movie_id: int, metadata_catalog: pd.DataFrame,
                   tfidf: TfidfMovieRecommender,
                   transformer: TransformerMovieRecommender,
                   hybrid: HybridContentRecommender) -> dict:
    query_has_plot = movie_id in tfidf.movie_id_to_row
    result = {
        "movie_id": movie_id,
        "title": _query_title(metadata_catalog, movie_id),
        "usable_plot": query_has_plot,
        "phase5_metadata": _result_records(
            recommend_similar_movies(metadata_catalog, movie_id, top_k=5), "similarity"
        ),
    }
    if query_has_plot:
        result["phase6_tfidf"] = _result_records(tfidf.recommend(movie_id, top_k=5), "similarity")
        result["phase7_transformer"] = _result_records(
            transformer.recommend(movie_id, top_k=5), "similarity"
        )
    else:
        result["phase6_tfidf"] = {"unavailable": "query has no usable plot"}
        result["phase7_transformer"] = {"unavailable": "query has no usable plot"}
    result["phase9_hybrid"] = _result_records(hybrid.recommend(movie_id, top_k=5), "hybrid_score")
    return result


def run_diagnostics() -> dict:
    clean_movies = load_tfidf_movies()
    enriched = load_enriched_movies()
    metadata_catalog = load_metadata_catalog()
    plot_count = int(usable_plot_mask(enriched).sum())
    if len(clean_movies) != len(enriched) or plot_count != 3434:
        # Report actual catalog state instead of assuming the Phase 8 snapshot remains current.
        pass

    tfidf = TfidfMovieRecommender(clean_movies)
    transformer = TransformerMovieRecommender(clean_movies, cache_path=EMBEDDINGS_PATH)
    hybrid = HybridContentRecommender(
        enriched, tfidf_model=tfidf, transformer_model=transformer, weights=DEFAULT_WEIGHTS
    )

    known = {}
    for movie_id, expected_title in KNOWN_FAILURE_QUERIES.items():
        actual_title = _query_title(metadata_catalog, movie_id)
        known[movie_id] = {
            "expected_title": expected_title,
            "title_matches_expected": actual_title == expected_title,
            **_run_baselines(movie_id, metadata_catalog, tfidf, transformer, hybrid),
        }

    broad_queries = _broad_query_ids(enriched)
    broad_results = {
        str(movie_id): {
            "selected_genres": genres,
            **_run_baselines(movie_id, metadata_catalog, tfidf, transformer, hybrid),
        }
        for movie_id, genres in broad_queries.items()
    }

    missing_plot_rows = enriched.loc[
        ~usable_plot_mask(enriched)
        & enriched["genres"].fillna("").ne("(no genres listed)")
        & (enriched["directors"].notna() | enriched["writers"].notna()
           | enriched["cast"].notna() | enriched["selected_crew"].notna())
    ].sort_values("movie_id")
    no_plot_query = None
    if not missing_plot_rows.empty:
        movie_id = int(missing_plot_rows.iloc[0]["movie_id"])
        no_plot_query = _run_baselines(movie_id, metadata_catalog, tfidf, transformer, hybrid)

    model = load_sentence_model(transformer.model_id, transformer.model_revision)
    chunk_vectors, chunk_metadata = load_or_build_chunked_embeddings(transformer, model)
    tokenizer = _tokenizer_from_model(model)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    token_lengths = [len(tokenizer.encode(str(plot), add_special_tokens=False))
                     for plot in transformer.plot_movies["plot"]]
    long_rows = pd.DataFrame({
        "movie_id": transformer.plot_movies["movie_id"].to_numpy(),
        "title": transformer.plot_movies["title"].to_numpy(),
        "token_count": token_lengths,
        "position": np.arange(len(token_lengths)),
    }).sort_values(["token_count", "movie_id"], ascending=[False, True], kind="mergesort")
    long_examples = []
    for row in long_rows.head(5).itertuples(index=False):
        position = int(row.position)
        baseline_scores = cosine_similarity(
            transformer.embeddings[position:position + 1], transformer.embeddings
        )[0]
        chunk_scores = cosine_similarity(chunk_vectors[position:position + 1], chunk_vectors)[0]
        baseline_scores[position] = -1
        chunk_scores[position] = -1
        baseline_top = np.argsort(-baseline_scores, kind="stable")[:5]
        chunk_top = np.argsort(-chunk_scores, kind="stable")[:5]
        ids = transformer.plot_movies["movie_id"].to_numpy()
        titles = transformer.plot_movies["title"].astype(str).to_numpy()
        baseline_ids = [int(ids[index]) for index in baseline_top]
        chunk_ids = [int(ids[index]) for index in chunk_top]
        shared = set(baseline_ids) & set(chunk_ids)
        long_examples.append({
            "movie_id": int(row.movie_id),
            "title": str(row.title),
            "wordpiece_count": int(row.token_count),
            "exceeds_phase7_limit": int(row.token_count) + special_tokens > int(model.max_seq_length),
            "phase7_single_input_top5": [
                {"movie_id": int(ids[i]), "title": str(titles[i]), "cosine": float(baseline_scores[i])}
                for i in baseline_top
            ],
            "chunked_experiment_top5": [
                {"movie_id": int(ids[i]), "title": str(titles[i]), "cosine": float(chunk_scores[i])}
                for i in chunk_top
            ],
            "top5_overlap_count": len(shared),
            "top5_overlap_fraction": len(shared) / 5,
        })

    report = {
        "phase": "Phase 9: Hybrid Content-Based Recommendation",
        "catalog": {
            "movie_count": int(len(enriched)),
            "usable_plot_count": plot_count,
            "missing_plot_count": int(len(enriched) - plot_count),
            "director_count": int(enriched.directors.notna().sum()),
            "writer_count": int(enriched.writers.notna().sum()),
            "cast_count": int(enriched["cast"].notna().sum()),
            "selected_crew_count": int(enriched.selected_crew.notna().sum()),
        },
        "weights": DEFAULT_WEIGHTS,
        "combination": (
            "The Phase 5 metadata component preserves its 0.7 genre/0.3 tag Jaccard formula "
            "when both fields are present and renormalizes over pairwise-present fields when "
            "one is missing. Per-component scores become within-query percentile ranks; configured weights "
            "combine available ranks; final rank score is multiplied by available/query-signal "
            "weight coverage. Candidates require at least two positive component similarities."
        ),
        "evaluation_note": "Qualitative diagnostic only; no held-out rating evaluation or weight tuning.",
        "known_failure_comparisons": known,
        "broader_genre_diagnostics": {
            "selection_rule": "Lowest movie_id with usable plot for each predefined genre list; overlaps deduplicated.",
            "genre_list": DIAGNOSTIC_GENRES,
            "query_count": len(broad_results),
            "results": broad_results,
        },
        "no_plot_metadata_query": no_plot_query,
        "chunked_plot_experiment": {
            "model_id": transformer.model_id,
            "model_revision": transformer.model_revision,
            "phase7_max_sequence_length": int(model.max_seq_length),
            "chunk_size_wordpieces": chunk_metadata["chunk_size"],
            "overlap_wordpieces": chunk_metadata["overlap"],
            "long_plot_count": chunk_metadata["long_plot_count"],
            "chunks_encoded": chunk_metadata["chunk_count"],
            "short_plot_vectors_reused": chunk_metadata["short_plot_vectors_reused_from_phase7"],
            "cache_reused": chunk_metadata["reused"],
            "representative_long_plots": long_examples,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_diagnostics()
    print(json.dumps({
        "report_path": str(REPORT_PATH),
        "known_query_count": len(result["known_failure_comparisons"]),
        "broad_query_count": result["broader_genre_diagnostics"]["query_count"],
        "chunked_long_plots": result["chunked_plot_experiment"]["long_plot_count"],
        "chunked_overlap_counts": [
            row["top5_overlap_count"]
            for row in result["chunked_plot_experiment"]["representative_long_plots"]
        ],
    }, indent=2))
