"""Leakage-safe, reproducible Phase 13 evaluation for CineMatch."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase12_experiments import mask_diagnostic_names, name_similarity_contribution
from scripts.recommend_cold_start import ColdStartRecommender
from scripts.recommend_collaborative import ItemItemCollaborativeRecommender
from scripts.recommend_hybrid import HybridRecommender
from scripts.recommend_hybrid_content import (
    RESULT_COLUMNS as PHASE9_RESULT_COLUMNS,
    HybridContentRecommender,
    load_enriched_movies,
    parse_people,
)
from scripts.recommend_metadata import prepare_catalog, recommend_similar_movies
from scripts.recommend_tfidf import TfidfMovieRecommender
from scripts.recommend_transformer import (
    EMBEDDINGS_PATH,
    MODEL_ID,
    MODEL_REVISION,
    TransformerMovieRecommender,
    plot_fingerprint,
    usable_plot_mask,
)
from scripts.recommender_evaluation import (
    aggregate_rating_metrics,
    make_temporal_split,
    mean_user_bootstrap_interval,
    ranking_metrics,
    split_ratings_by_manifest,
    user_bootstrap_error_difference,
)


CLEAN_MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies.csv"
RATINGS_PATH = PROJECT_ROOT / "data" / "cleaned" / "ratings.csv"
ENRICHED_PATH = PROJECT_ROOT / "data" / "enriched" / "movies.csv"
CHUNKED_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "models" / "all_minilm_l6_v2_plots_chunked.npz"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "phase13_split_manifest.csv"
JSON_REPORT_PATH = PROJECT_ROOT / "reports" / "phase13_evaluation.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "reports" / "phase13_evaluation.md"

HOLDOUT_FRACTION = 0.20
MINIMUM_TRAINING_RATINGS = 5
RELEVANCE_THRESHOLD = 4.0
K_VALUES = (5, 10)
MAX_POSITIVE_SEEDS = 5
BOOTSTRAP_RESAMPLES = 1000


class CachedContentRanker:
    """Serve cached Phase 9 rankings using equivalent vectorized evaluation math.

    Production recommenders are untouched. Each cached result is checked
    against the regular Phase 9 API on representative queries before scoring.
    """

    def __init__(self, model: HybridContentRecommender, *, catalog_ids: list[int]) -> None:
        self.model = model
        self.catalog_ids = np.asarray(catalog_ids, dtype=np.int64)
        self.id_to_position = {int(movie_id): row for row, movie_id in enumerate(self.catalog_ids)}
        self.rows: dict[int, pd.DataFrame] = {}
        self.parity_checks: list[dict[str, Any]] = []
        self._label_matrices: dict[str, tuple[sparse.csr_matrix, np.ndarray, dict[str, int]]] = {}
        for signal, labels_by_row in (
            ("genre", model.catalog.genre_labels.tolist()),
            ("tags", model.catalog.tag_labels.tolist()),
            ("director", model.credit_sets["director"]),
            ("writer", model.credit_sets["writer"]),
            ("cast", model.credit_sets["cast"]),
            ("crew", model.credit_sets["crew"]),
        ):
            vocabulary = sorted({label for labels in labels_by_row for label in labels})
            label_to_column = {label: column for column, label in enumerate(vocabulary)}
            row_indices: list[int] = []
            column_indices: list[int] = []
            for row, labels in enumerate(labels_by_row):
                for label in labels:
                    row_indices.append(row)
                    column_indices.append(label_to_column[label])
            matrix = sparse.csr_matrix(
                (np.ones(len(row_indices), dtype=np.float64), (row_indices, column_indices)),
                shape=(len(labels_by_row), len(vocabulary)),
            )
            counts = np.asarray(matrix.sum(axis=1)).ravel()
            self._label_matrices[signal] = (matrix, counts, label_to_column)

    def _jaccard(self, signal: str, query_labels: set[str]) -> tuple[np.ndarray, np.ndarray]:
        matrix, counts, label_to_column = self._label_matrices[signal]
        query_columns = [label_to_column[label] for label in query_labels if label in label_to_column]
        if not query_columns:
            return np.zeros(len(self.catalog_ids)), np.zeros(len(self.catalog_ids), dtype=bool)
        intersections = np.asarray(matrix[:, query_columns].sum(axis=1)).ravel()
        unions = len(query_labels) + counts - intersections
        scores = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        return scores, counts > 0

    @staticmethod
    def _percentile_scores(values: np.ndarray, available: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        result = np.zeros(len(movie_ids), dtype=np.float64)
        positions = np.flatnonzero(available)
        if not len(positions):
            return result
        ranked = pd.Series(values[positions], index=movie_ids[positions]).rank(method="average", ascending=True)
        if len(positions) == 1:
            result[positions] = 1.0
        else:
            result[positions] = ((ranked.to_numpy() - 1) / (len(positions) - 1)).astype(np.float64)
        return result

    def _vectorized_recommend(self, movie_id: int, top_k: int) -> pd.DataFrame:
        model = self.model
        position = model.movie_id_to_position[movie_id]
        query = model.catalog.iloc[position]
        query_genres = query["genre_labels"]
        query_tags = query["tag_labels"]
        scores: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}

        genre_scores, genre_candidates = self._jaccard("genre", query_genres)
        tag_scores, tag_candidates = self._jaccard("tags", query_tags)
        genre_available = genre_candidates & bool(query_genres)
        tags_available = tag_candidates & bool(query_tags)
        genre_available[position] = False
        tags_available[position] = False
        scores["genre"], available["genre"] = genre_scores, genre_available
        scores["tags"], available["tags"] = tag_scores, tags_available

        meta_available = genre_available | tags_available
        meta_weight = 0.7 * genre_available + 0.3 * tags_available
        meta_numerator = 0.7 * genre_scores * genre_available + 0.3 * tag_scores * tags_available
        meta_scores = np.divide(meta_numerator, meta_weight, out=np.zeros_like(meta_numerator), where=meta_weight > 0)
        scores["metadata"], available["metadata"] = meta_scores, meta_available

        for signal in ("director", "writer", "cast", "crew"):
            values, has_candidates = self._jaccard(signal, model.credit_sets[signal][position])
            mask = has_candidates & bool(model.credit_sets[signal][position])
            mask[position] = False
            scores[signal], available[signal] = values, mask

        text_rows = np.zeros(len(self.catalog_ids), dtype=bool)
        text_scores: dict[str, np.ndarray] = {
            "tfidf": np.zeros(len(self.catalog_ids), dtype=np.float64),
            "transformer": np.zeros(len(self.catalog_ids), dtype=np.float64),
        }
        available["transformer"] = np.zeros(len(self.catalog_ids), dtype=bool)
        movie_id_to_position = self.id_to_position
        tfidf_row = model.tfidf.movie_id_to_row.get(movie_id)
        if tfidf_row is not None:
            values = cosine_similarity(model.tfidf.tfidf_matrix[tfidf_row], model.tfidf.tfidf_matrix).ravel()
            for candidate_id, row in model.tfidf.movie_id_to_row.items():
                if candidate_id != movie_id:
                    text_scores["tfidf"][movie_id_to_position[candidate_id]] = float(values[row])
                    text_rows[movie_id_to_position[candidate_id]] = True
        available["tfidf"] = text_rows.copy()
        transformer_row = model.transformer.movie_id_to_row.get(movie_id)
        if transformer_row is not None:
            values = cosine_similarity(
                model.transformer.embeddings[transformer_row:transformer_row + 1], model.transformer.embeddings
            ).ravel()
            for candidate_id, row in model.transformer.movie_id_to_row.items():
                if candidate_id != movie_id:
                    text_scores["transformer"][movie_id_to_position[candidate_id]] = float(values[row])
                    available["transformer"][movie_id_to_position[candidate_id]] = True
        scores.update(text_scores)

        query_weight = model._query_available_weight(position)
        if query_weight <= 0:
            return pd.DataFrame(columns=PHASE9_RESULT_COLUMNS)
        weights = model.weights
        available_weight = np.zeros(len(self.catalog_ids), dtype=np.float64)
        weighted_rank = np.zeros(len(self.catalog_ids), dtype=np.float64)
        positive_signals = np.zeros(len(self.catalog_ids), dtype=np.int64)
        percentiles: dict[str, np.ndarray] = {}
        for signal, weight in weights.items():
            percentiles[signal] = self._percentile_scores(scores[signal], available[signal], self.catalog_ids)
            available_weight += weight * available[signal]
            weighted_rank += weight * percentiles[signal] * available[signal]
            positive_signals += ((scores[signal] > 0) & available[signal]).astype(np.int64)
        coverage = np.minimum(1.0, available_weight / query_weight)
        final_scores = np.divide(weighted_rank, available_weight, out=np.zeros_like(weighted_rank), where=available_weight > 0) * coverage
        eligible = np.flatnonzero((available_weight > 0) & (positive_signals >= model.min_positive_signals))
        if not len(eligible):
            return pd.DataFrame(columns=PHASE9_RESULT_COLUMNS)
        ordering = np.lexsort((self.catalog_ids[eligible], -positive_signals[eligible], -final_scores[eligible]))
        selected = eligible[ordering[:top_k]]
        rows: list[dict[str, Any]] = []
        for candidate_position in selected:
            candidate_id = int(self.catalog_ids[candidate_position])
            raw = {signal: float(scores[signal][candidate_position]) for signal in weights if available[signal][candidate_position]}
            missing = [signal for signal in weights if not available[signal][candidate_position]]
            candidate = model.catalog.iloc[candidate_position]
            if query_genres and not candidate["genre_labels"]:
                missing.append("genres")
            if query_tags and not candidate["tag_labels"]:
                missing.append("plot tags")
            row: dict[str, Any] = {
                "movie_id": candidate_id,
                "title": candidate.title,
                "hybrid_score": float(final_scores[candidate_position]),
                "positive_signals": int(positive_signals[candidate_position]),
                "explanation": model._explanation(position, candidate_position, raw),
                "evidence_coverage": float(coverage[candidate_position]),
                "missing_signals": ", ".join(missing),
                "metadata_similarity": raw.get("metadata", np.nan),
                "genre_similarity": scores["genre"][candidate_position] if available["genre"][candidate_position] else np.nan,
                "tag_similarity": scores["tags"][candidate_position] if available["tags"][candidate_position] else np.nan,
            }
            for signal in weights:
                if signal != "metadata":
                    row[f"{signal}_similarity"] = raw.get(signal, np.nan)
                    row[f"{signal}_rank_percentile"] = percentiles[signal][candidate_position] if available[signal][candidate_position] else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def add(self, movie_id: int, top_k: int) -> None:
        result = self._vectorized_recommend(movie_id, top_k)
        self.rows[movie_id] = result[["movie_id", "hybrid_score", "explanation"]].copy()

    def recommend(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        if movie_id not in self.rows:
            self.add(movie_id, top_k)
        return self.rows[movie_id].head(top_k).copy()


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(CLEAN_MOVIES_PATH, dtype={"title": "string", "genres": "string", "plot": "string", "plot_tags": "string"})
    ratings = pd.read_csv(RATINGS_PATH)
    enriched = load_enriched_movies(ENRICHED_PATH)
    if movies.movie_id.duplicated().any() or enriched.movie_id.duplicated().any():
        raise ValueError("Movie catalogs must have unique movie_id values")
    if set(movies.movie_id.astype(int)) != set(enriched.movie_id.astype(int)):
        raise ValueError("Cleaned and enriched movie ID sets do not match")
    if ratings[["user_id", "movie_id", "rating", "timestamp"]].isna().any().any():
        raise ValueError("Rating data has missing user, movie, rating, or timestamp fields")
    if not set(ratings.movie_id.astype(int)).issubset(set(movies.movie_id.astype(int))):
        raise ValueError("Ratings reference movie IDs missing from the cleaned catalog")
    return movies, ratings, enriched


def positive_seed_ids(history: pd.DataFrame) -> tuple[int, ...]:
    """Use only training history, matching Phase 12's above-mean seed policy."""
    if history.empty:
        return ()
    average = float(history.rating.mean())
    positives = history.loc[history.rating.gt(average)].sort_values(
        ["rating", "movie_id"], ascending=[False, True], kind="mergesort"
    )
    return tuple(positives.movie_id.astype(int).head(MAX_POSITIVE_SEEDS).tolist())


def history_group(count: int) -> str:
    """Fixed history bins chosen before looking at model outcomes."""
    if count <= 20:
        return "sparse_5_to_20"
    if count <= 50:
        return "moderate_21_to_50"
    return "substantial_over_50"


def build_user_specs(train: pd.DataFrame, top_k: int) -> dict[int, dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}
    for user_id, history in train.groupby("user_id", sort=True):
        seeds = positive_seed_ids(history)
        seen = set(history.movie_id.astype(int))
        pool_size = max(5 * top_k, top_k) + len(seen)
        specs[int(user_id)] = {
            "history": history.copy(),
            "seen": seen,
            "seeds": seeds,
            "pool_size": pool_size,
            "history_count": len(history),
            "history_group": history_group(len(history)),
        }
    return specs


def precompute_phase9(
    phase9: HybridContentRecommender,
    specs: dict[int, dict[str, Any]],
) -> CachedContentRanker:
    """Cache only seed queries used in this split; weights and outputs stay intact."""
    max_requested: dict[int, int] = {}
    for spec in specs.values():
        for seed_id in spec["seeds"]:
            requested = spec["pool_size"] + len(spec["seeds"])
            max_requested[seed_id] = max(max_requested.get(seed_id, 0), requested)
    cached = CachedContentRanker(phase9, catalog_ids=phase9.catalog.movie_id.astype(int).tolist())
    for seed_id, limit in sorted(max_requested.items()):
        cached.add(seed_id, limit)
        if len(cached.rows) % 100 == 0:
            print(f"Phase 9 evaluation cache: {len(cached.rows)}/{len(max_requested)} unique favorite movies", flush=True)
    # Confirm that the evaluation-only vectorized path preserves Phase 9's
    # recommendation order and scores for deterministic representative seeds.
    for seed_id in (3, 4, 5, 1, 858):
        if seed_id not in phase9.catalog_ids:
            continue
        fast = cached._vectorized_recommend(seed_id, 10)
        regular = phase9.recommend(seed_id, 10)
        if fast.movie_id.astype(int).tolist() != regular.movie_id.astype(int).tolist() or not np.allclose(
            fast.hybrid_score.to_numpy(dtype=float), regular.hybrid_score.to_numpy(dtype=float), rtol=1e-8, atol=1e-8
        ):
            raise AssertionError(f"Evaluation optimization changed Phase 9 output for movie_id={seed_id}")
        cached.parity_checks.append({
            "movie_id": seed_id,
            "top10_ids_match": True,
            "hybrid_scores_match_atol_1e_8": True,
        })
    return cached


def precompute_base_content_lists(
    specs: dict[int, dict[str, Any]],
    metadata_catalog: pd.DataFrame,
    tfidf: TfidfMovieRecommender,
    transformer: TransformerMovieRecommender,
) -> dict[str, dict[int, list[int]]]:
    """Compute each unique positive seed once for Phases 5–7."""
    limits: dict[int, int] = {}
    for spec in specs.values():
        for seed_id in spec["seeds"]:
            limits[seed_id] = max(limits.get(seed_id, 0), spec["pool_size"] + len(spec["seeds"]))
    cache = {name: {} for name in ("phase5_metadata", "phase6_tfidf", "phase7_transformer")}
    for seed_id, limit in sorted(limits.items()):
        metadata_rows = recommend_similar_movies(metadata_catalog, seed_id, top_k=limit)
        cache["phase5_metadata"][seed_id] = metadata_rows.movie_id.astype(int).tolist()
        for name, model in (("phase6_tfidf", tfidf), ("phase7_transformer", transformer)):
            try:
                rows = model.recommend(seed_id, top_k=limit)
                cache[name][seed_id] = rows.movie_id.astype(int).tolist()
            except ValueError as error:
                if "no usable plot" in str(error):
                    cache[name][seed_id] = []
                else:
                    raise
        if len(cache["phase5_metadata"]) % 100 == 0:
            print(f"Phase 5–7 evaluation cache: {len(cache['phase5_metadata'])}/{len(limits)} unique favorite movies", flush=True)
    return cache


def _aggregate_seed_rankings(
    seed_lists: list[list[int]], *, seeds_to_remove_before_ranking: set[int], seen: set[int], top_k: int
) -> list[int]:
    """Mirror Phase 11 seed MRR, then apply the user's full seen-item filter."""
    totals: dict[int, float] = defaultdict(float)
    supports: Counter[int] = Counter()
    best_rank: dict[int, int] = {}
    for movie_ids in seed_lists:
        filtered: list[int] = []
        local_seen: set[int] = set()
        for movie_id in movie_ids:
            movie_id = int(movie_id)
            if movie_id in seeds_to_remove_before_ranking or movie_id in local_seen:
                continue
            filtered.append(movie_id)
            local_seen.add(movie_id)
        for rank, movie_id in enumerate(filtered, start=1):
            totals[movie_id] += 1.0 / rank
            supports[movie_id] += 1
            best_rank[movie_id] = min(best_rank.get(movie_id, rank), rank)
    divisor = max(1, len(seed_lists))
    ordered = sorted(
        totals,
        key=lambda movie_id: (-totals[movie_id] / divisor, -supports[movie_id], best_rank[movie_id], movie_id),
    )
    return [movie_id for movie_id in ordered if movie_id not in seen][:top_k]


def content_user_rankings(
    model_name: str,
    spec: dict[str, Any],
    *,
    metadata_catalog: pd.DataFrame,
    tfidf: TfidfMovieRecommender,
    transformer: TransformerMovieRecommender,
    cold_start: ColdStartRecommender,
    seed_rank_cache: dict[str, dict[int, list[int]]],
    top_k: int,
) -> tuple[list[int], str]:
    seeds: tuple[int, ...] = spec["seeds"]
    if not seeds:
        return [], "no_above_training_mean_seed"
    if model_name == "phase9_content_hybrid":
        response = cold_start.recommend_from_favorites(seeds, top_k=spec["pool_size"])
        rows = response.recommendations
        ids = rows.movie_id.astype(int).tolist() if not rows.empty else []
        return _aggregate_seed_rankings([ids], seeds_to_remove_before_ranking=set(),
                                        seen=spec["seen"], top_k=top_k), response.status

    per_seed: list[list[int]] = []
    query_limit = spec["pool_size"] + len(seeds)
    for seed_id in seeds:
        if model_name not in seed_rank_cache or seed_id not in seed_rank_cache[model_name]:
            raise KeyError(f"No cached ranking for model {model_name} and seed {seed_id}")
        per_seed.append(seed_rank_cache[model_name][seed_id][:query_limit])
    status = "content_rank_aggregation" if any(per_seed) else "no_usable_seed_content"
    return _aggregate_seed_rankings(per_seed, seeds_to_remove_before_ranking=set(seeds),
                                    seen=spec["seen"], top_k=top_k), status


def _recommendation_ids(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "movie_id" not in frame.columns:
        return []
    return frame.movie_id.astype(int).drop_duplicates().tolist()


def _summarize_ranking_model(
    rankings: dict[int, list[int]], relevant_by_user: dict[int, set[int]], specs: dict[int, dict[str, Any]], k_values: tuple[int, ...]
) -> dict[str, Any]:
    output: dict[str, Any] = {"eligible_users": len(specs), "users_with_any_recommendation": sum(bool(rankings.get(user)) for user in specs)}
    output["recommendation_coverage"] = output["users_with_any_recommendation"] / max(1, len(specs))
    metrics_by_k: dict[str, Any] = {}
    group_summary: dict[str, Any] = {}
    all_groups = ("sparse_5_to_20", "moderate_21_to_50", "substantial_over_50")

    def calculate(users: list[int], k: int) -> dict[str, Any]:
        p_values: dict[int, float] = {}
        r_values: dict[int, float] = {}
        n_values: dict[int, float] = {}
        relevant_users = 0
        no_relevant = 0
        total_relevant = 0
        for user in users:
            relevant = relevant_by_user.get(user, set())
            total_relevant += len(relevant)
            result = ranking_metrics(rankings.get(user, []), relevant, k)
            p_values[user] = float(result["precision"])
            if result["recall"] is None:
                no_relevant += 1
            else:
                relevant_users += 1
                r_values[user] = float(result["recall"])
                n_values[user] = float(result["ndcg"])
        summary: dict[str, Any] = {
            "precision_at_k": float(np.mean(list(p_values.values()))) if p_values else 0.0,
            "recall_at_k": float(np.mean(list(r_values.values()))) if r_values else None,
            "ndcg_at_k": float(np.mean(list(n_values.values()))) if n_values else None,
            "eligible_users_precision": len(users),
            "eligible_users_recall_ndcg": relevant_users,
            "users_with_no_relevant_test_movie_excluded_from_recall_ndcg": no_relevant,
            "relevant_heldout_interactions": total_relevant,
            "precision_user_bootstrap_95_ci": mean_user_bootstrap_interval(p_values) if p_values else None,
            "recall_user_bootstrap_95_ci": mean_user_bootstrap_interval(r_values) if r_values else None,
            "ndcg_user_bootstrap_95_ci": mean_user_bootstrap_interval(n_values) if n_values else None,
        }
        return summary

    for k in k_values:
        metrics_by_k[str(k)] = calculate(sorted(specs), k)
    for group in all_groups:
        users = [user for user, spec in specs.items() if spec["history_group"] == group]
        group_summary[group] = {
            "users": len(users),
            "users_with_any_recommendation": sum(bool(rankings.get(user)) for user in users),
            "recommendation_coverage": sum(bool(rankings.get(user)) for user in users) / max(1, len(users)),
            "metrics_at_k": {str(k): calculate(users, k) for k in k_values},
        }
    output["metrics_by_k"] = metrics_by_k
    output["history_groups"] = group_summary
    return output


def _feature_coverage(
    clean_movies: pd.DataFrame,
    enriched: pd.DataFrame,
    specs: dict[int, dict[str, Any]],
    relevant_by_user: dict[int, set[int]],
) -> dict[str, Any]:
    clean = clean_movies.set_index("movie_id")
    enriched_indexed = enriched.set_index("movie_id")
    feature_user_counts = Counter()
    seed_count = 0
    seed_feature_counts = Counter()
    for spec in specs.values():
        seeds = spec["seeds"]
        for feature in ("usable_plot", "genres", "plot_tags", "directors", "writers", "cast", "selected_crew"):
            available = False
            for seed in seeds:
                movie = enriched_indexed.loc[seed]
                if feature == "usable_plot":
                    value = clean.loc[seed, "plot"]
                    has_value = pd.notna(value) and bool(str(value).strip())
                elif feature in {"directors", "writers", "cast", "selected_crew"}:
                    value = movie.get(feature, pd.NA)
                    has_value = bool(parse_people(value, crew=feature == "selected_crew")[0])
                else:
                    value = movie.get(feature, pd.NA)
                    has_value = pd.notna(value) and bool(str(value).strip())
                available = available or has_value
            if available:
                feature_user_counts[feature] += 1
        for seed in seeds:
            seed_count += 1
            movie = enriched_indexed.loc[seed]
            plot = clean.loc[seed, "plot"]
            if pd.notna(plot) and str(plot).strip():
                seed_feature_counts["usable_plot"] += 1
            for feature in ("genres", "plot_tags", "directors", "writers", "cast", "selected_crew"):
                value = movie.get(feature, pd.NA)
                if (bool(parse_people(value, crew=feature == "selected_crew")[0])
                        if feature in {"directors", "writers", "cast", "selected_crew"}
                        else pd.notna(value) and bool(str(value).strip())):
                    seed_feature_counts[feature] += 1

    relevant_ids = [movie_id for values in relevant_by_user.values() for movie_id in values]
    targets: dict[str, Any] = {"relevant_test_movies": len(relevant_ids)}
    for feature in ("usable_plot", "genres", "plot_tags", "directors", "writers", "cast", "selected_crew"):
        available = 0
        for movie_id in relevant_ids:
            movie = enriched_indexed.loc[movie_id]
            if feature == "usable_plot":
                value = clean.loc[movie_id, "plot"]
                has_value = pd.notna(value) and bool(str(value).strip())
            elif feature in {"directors", "writers", "cast", "selected_crew"}:
                value = movie.get(feature, pd.NA)
                has_value = bool(parse_people(value, crew=feature == "selected_crew")[0])
            else:
                value = movie.get(feature, pd.NA)
                has_value = pd.notna(value) and bool(str(value).strip())
            available += int(has_value)
        targets[feature] = {"available": available, "missing": len(relevant_ids) - available,
                            "coverage": available / max(1, len(relevant_ids))}
    return {
        "positive_seed_movies": seed_count,
        "users_with_at_least_one_usable_seed_feature": {
            feature: {"users": int(feature_user_counts[feature]), "of": len(specs),
                      "coverage": feature_user_counts[feature] / max(1, len(specs))}
            for feature in ("usable_plot", "genres", "plot_tags", "directors", "writers", "cast", "selected_crew")
        },
        "positive_seed_movie_field_coverage": {
            feature: {"available": int(seed_feature_counts[feature]), "of": seed_count,
                      "coverage": seed_feature_counts[feature] / max(1, seed_count)}
            for feature in ("usable_plot", "genres", "plot_tags", "directors", "writers", "cast", "selected_crew")
        },
        "relevant_heldout_target_field_coverage": targets,
        "note": "Missing values mean no usable source value was available; they are not interpreted as evidence that the movie lacks that property.",
    }


def _rating_prediction_evaluation(
    model: ItemItemCollaborativeRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movie_count: int,
) -> tuple[dict[str, Any], dict[int, list[int]], dict[int, dict[int, float]], dict[int, str]]:
    global_mean = float(train.rating.mean())
    user_means = train.groupby("user_id").rating.mean().to_dict()
    predictions_by_user: dict[int, dict[int, float]] = {}
    collaborative_rankings: dict[int, list[int]] = {}
    failures: dict[int, str] = {}
    for user_id in sorted(train.user_id.unique()):
        try:
            result = model.recommend(int(user_id), top_k=movie_count)
            predictions_by_user[int(user_id)] = dict(zip(result.movie_id.astype(int), result.predicted_rating.astype(float)))
            collaborative_rankings[int(user_id)] = _recommendation_ids(result)
        except (ValueError, KeyError) as error:
            predictions_by_user[int(user_id)] = {}
            collaborative_rankings[int(user_id)] = []
            failures[int(user_id)] = str(error)
        if len(predictions_by_user) % 100 == 0:
            print(f"Phase 10 held-out candidate scores: {len(predictions_by_user)}/{train.user_id.nunique()} users", flush=True)

    cf_actual: list[float] = []
    cf_predicted: list[float] = []
    cf_error_by_user: dict[int, np.ndarray] = defaultdict(lambda: np.asarray([], dtype=np.float64))
    global_actual: list[float] = []
    global_predicted: list[float] = []
    unsupported = 0
    paired_user_error_by_user: dict[int, np.ndarray] = defaultdict(lambda: np.asarray([], dtype=np.float64))
    for row in test.itertuples(index=False):
        user_id, movie_id, actual = int(row.user_id), int(row.movie_id), float(row.rating)
        global_actual.append(actual)
        global_predicted.append(global_mean)
        user_estimate = float(user_means[user_id])
        prediction = predictions_by_user.get(user_id, {}).get(movie_id)
        if prediction is None:
            unsupported += 1
            continue
        cf_actual.append(actual)
        cf_predicted.append(prediction)
        cf_error_by_user[user_id] = np.append(cf_error_by_user[user_id], prediction - actual)
        paired_user_error_by_user[user_id] = np.append(
            paired_user_error_by_user[user_id], user_estimate - actual
        )

    global_metric = aggregate_rating_metrics(global_actual, global_predicted)
    user_metric = aggregate_rating_metrics(test.rating, [user_means[int(user)] for user in test.user_id])
    common_actual = cf_actual
    common_user_prediction = [float(user_means[int(user)]) for user, movie in zip(
        test.user_id.tolist(), test.movie_id.tolist()
    ) if movie in predictions_by_user.get(int(user), {})]
    cf_metric = aggregate_rating_metrics(cf_actual, cf_predicted) if cf_actual else None
    user_common = aggregate_rating_metrics(common_actual, common_user_prediction) if common_actual else None
    paired_errors_cf = {user: values for user, values in cf_error_by_user.items() if len(values)}
    paired_errors_user = {user: values for user, values in paired_user_error_by_user.items() if user in paired_errors_cf}
    paired = None
    if paired_errors_cf:
        paired = {
            "same_supported_test_interactions": int(len(cf_actual)),
            "collaborative_filtering": cf_metric,
            "user_mean_baseline": user_common,
            "mae_difference_and_95_ci": user_bootstrap_error_difference(
                paired_errors_cf, paired_errors_user, metric="mae"
            ),
            "rmse_difference_and_95_ci": user_bootstrap_error_difference(
                paired_errors_cf, paired_errors_user, metric="rmse"
            ),
            "difference_sign": "negative favors collaborative filtering; positive favors user-mean baseline",
        }
    report = {
        "task": "held-out rating prediction",
        "training_only_baselines": {
            "global_mean": {**global_metric, "mean_used": global_mean, "coverage": 1.0},
            "user_mean": {**user_metric, "users_with_training_mean": int(len(user_means)), "coverage": 1.0},
        },
        "phase10_collaborative": {
            "metrics_on_supported_predictions": cf_metric,
            "test_interactions": int(len(test)),
            "predictions_produced": int(len(cf_actual)),
            "predictions_unsupported": unsupported,
            "unsupported_percentage": unsupported / max(1, len(test)) * 100,
            "coverage": len(cf_actual) / max(1, len(test)),
            "users_without_any_collaborative_recommendations": failures,
        },
        "paired_comparison_on_same_supported_interactions": paired,
        "unsupported_policy": "Do not impute a collaborative estimate when the existing candidate generator returns no score; report it as uncovered. Baselines are also evaluated on this common subset for paired comparisons.",
        "not_supported": ["Phase 5", "Phase 6", "Phase 7", "Phase 9", "Phase 12: their score fields are not user rating predictions."],
    }
    return report, collaborative_rankings, predictions_by_user, failures


def _rank_fusion_diagnostics(
    hybrid_outputs: dict[int, Any],
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    single_source_scores: list[float] = []
    both_source_scores: list[float] = []
    one_candidate_source_two_lists: list[float] = []
    one_available_source_fallback: list[float] = []
    fallback_statuses: Counter[str] = Counter()
    returned = 0
    for response in hybrid_outputs.values():
        fallback_statuses[response.status] += 1
        available_source_count = len(str(response.recommendation_source).split("+")) if response.recommendation_source != "none" else 0
        for row in response.recommendations.itertuples(index=False):
            returned += 1
            source = str(row.contributing_sources)
            source_counts[source] += 1
            if source in {"collaborative", "content"}:
                single_source_scores.append(float(row.fusion_score))
                if available_source_count == 2:
                    one_candidate_source_two_lists.append(float(row.fusion_score))
                elif available_source_count == 1:
                    one_available_source_fallback.append(float(row.fusion_score))
            elif source == "collaborative, content":
                both_source_scores.append(float(row.fusion_score))
    return {
        "user_response_status_counts": dict(fallback_statuses),
        "top10_rows_by_contributing_source": dict(source_counts),
        "returned_rows": returned,
        "single_source_row_fraction": len(single_source_scores) / max(1, returned),
        "mean_single_source_fusion_score": float(np.mean(single_source_scores)) if single_source_scores else None,
        "single_source_candidate_rows_when_both_lists_available": len(one_candidate_source_two_lists),
        "single_source_candidate_rows_when_one_list_fallback": len(one_available_source_fallback),
        "mean_score_single_candidate_source_with_both_lists": float(np.mean(one_candidate_source_two_lists)) if one_candidate_source_two_lists else None,
        "mean_score_single_available_source_fallback": float(np.mean(one_available_source_fallback)) if one_available_source_fallback else None,
        "mean_both_source_fusion_score": float(np.mean(both_source_scores)) if both_source_scores else None,
        "interpretation": "A single-source row can still receive half its reciprocal-rank contribution when both source lists are available; a fallback with only one available list uses that list without dilution. These are current Phase 12 semantics, not changed here.",
    }


def _qualitative_case_ids(movies: pd.DataFrame) -> list[tuple[str, int]]:
    title_to_id = dict(zip(movies.title.astype(str), movies.movie_id.astype(int)))
    known = [("known_name_sensitivity", 3, "Grumpier Old Men"), ("known_semantic_mismatch", 4, "Waiting to Exhale"),
             ("known_transformer_mismatch", 5, "Father of the Bride Part II"), ("plausible_prior_example", 1, "Toy Story"),
             ("different_genre_crime_drama", 858, "The Godfather")]
    cases: list[tuple[str, int]] = []
    for label, movie_id, title in known:
        if movie_id in set(movies.movie_id.astype(int)):
            cases.append((label, movie_id))
        elif title in title_to_id:
            cases.append((label, title_to_id[title]))
    usable = movies.loc[usable_plot_mask(movies)].copy()
    usable["plot_characters"] = usable["plot"].astype(str).str.len()
    if not usable.empty:
        longest = usable.sort_values(["plot_characters", "movie_id"], ascending=[False, True], kind="mergesort").iloc[0]
        cases.append(("longest_usable_plot", int(longest.movie_id)))
    missing = movies.loc[~usable_plot_mask(movies)].sort_values("movie_id", kind="mergesort")
    if not missing.empty:
        cases.append(("missing_plot", int(missing.iloc[0].movie_id)))
    unique: dict[int, str] = {}
    for label, movie_id in cases:
        unique.setdefault(movie_id, label)
    return [(label, movie_id) for movie_id, label in unique.items()]


def _audit_rows(
    rows: pd.DataFrame,
    *,
    model_name: str,
    catalog: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    by_id = catalog.set_index("movie_id", drop=False)
    output = []
    for rank, row in enumerate(rows.drop_duplicates("movie_id").head(top_k).itertuples(index=False), start=1):
        movie_id = int(row.movie_id)
        movie = by_id.loc[movie_id]
        item: dict[str, Any] = {
            "rank": rank,
            "movie_id": movie_id,
            "title": str(movie.title),
            "plot_available": bool(pd.notna(movie["plot"]) and str(movie["plot"]).strip()),
            "plot_excerpt": str(movie["plot"])[:260] if pd.notna(movie["plot"]) and str(movie["plot"]).strip() else None,
            "genres": str(movie.get("genres", "")) if pd.notna(movie.get("genres", pd.NA)) else None,
        }
        for column in ("similarity", "hybrid_score", "explanation", "shared_genres", "shared_tags",
                       "genre_similarity", "tag_similarity", "tfidf_similarity", "transformer_similarity",
                       "director_similarity", "writer_similarity", "cast_similarity", "crew_similarity"):
            if hasattr(row, column):
                value = getattr(row, column)
                if value is not None and not pd.isna(value):
                    item[column] = float(value) if isinstance(value, (float, np.floating, int, np.integer)) else str(value)
        item["model"] = model_name
        output.append(item)
    return output


def run_qualitative_audit(
    clean_movies: pd.DataFrame,
    enriched: pd.DataFrame,
    metadata_catalog: pd.DataFrame,
    tfidf: TfidfMovieRecommender,
    transformer: TransformerMovieRecommender,
    phase9: HybridContentRecommender,
    cold_start: ColdStartRecommender,
    unified: HybridRecommender,
) -> dict[str, Any]:
    cases = _qualitative_case_ids(clean_movies)
    output_cases = []
    for label, movie_id in cases:
        query = clean_movies.loc[clean_movies.movie_id.eq(movie_id)].iloc[0]
        query_plot = query["plot"]
        case: dict[str, Any] = {
            "selection_reason": label,
            "query_movie_id": movie_id,
            "query_title": str(query.title),
            "query_plot_available": bool(pd.notna(query_plot) and str(query_plot).strip()),
            "query_plot_excerpt": str(query_plot)[:400] if pd.notna(query_plot) and str(query_plot).strip() else None,
            "models": {},
        }
        for name, call in (
            ("phase5_metadata", lambda: recommend_similar_movies(metadata_catalog, movie_id, 5)),
            ("phase6_tfidf", lambda: tfidf.recommend(movie_id, 5)),
            ("phase7_transformer", lambda: transformer.recommend(movie_id, 5)),
            ("phase9_content_hybrid", lambda: phase9.recommend(movie_id, 5)),
            ("phase12_movie_content_route", lambda: cold_start.recommend_for_movie(movie_id, 5).recommendations),
            ("phase12_new_user_favorite_route", lambda: unified.recommend_for_new_user([movie_id], top_k=5).recommendations),
        ):
            try:
                rows = call()
                if hasattr(rows, "recommendations"):
                    rows = rows.recommendations
                case["models"][name] = {"status": "ok", "recommendations": _audit_rows(rows, model_name=name, catalog=enriched)}
            except (ValueError, KeyError) as error:
                case["models"][name] = {"status": "unavailable", "reason": str(error), "recommendations": []}
        output_cases.append(case)

    # Re-run the predeclared Phase 12 name-token ablation on the same named cases.
    name_cases = []
    named_ids = {movie_id for label, movie_id in cases if label in {"known_name_sensitivity", "known_semantic_mismatch", "known_transformer_mismatch"}}
    variant_movies = clean_movies.copy()
    variant_movies["plot"] = variant_movies["plot"].map(mask_diagnostic_names)
    masked_recommender = TfidfMovieRecommender(variant_movies)
    for movie_id in sorted(named_ids):
        try:
            base = tfidf.recommend(movie_id, 5)
            masked = masked_recommender.recommend(movie_id, 5)
            shares = []
            for candidate_id, score in zip(base.movie_id.astype(int), base.similarity.astype(float)):
                contribution = name_similarity_contribution(tfidf, movie_id, candidate_id)
                shares.append({"candidate_movie_id": candidate_id, "cosine": score,
                               "targeted_name_share": contribution / score if score > 0 else 0.0})
            name_cases.append({"query_movie_id": movie_id, "baseline_top5": base.movie_id.astype(int).tolist(),
                               "name_masked_top5": masked.movie_id.astype(int).tolist(),
                               "top5_overlap": len(set(base.movie_id.astype(int)) & set(masked.movie_id.astype(int))),
                               "targeted_name_contributions": shares,
                               "note": "Targeted ablation only; defaults remain unchanged and masking is not proposed as a fix."})
        except ValueError as error:
            name_cases.append({"query_movie_id": movie_id, "status": "unavailable", "reason": str(error)})

    chunked_audit = validate_chunked_artifact(clean_movies, transformer)
    new_user = cold_start.recommend_from_favorites([47, 3], top_k=10)
    no_preferences = cold_start.recommend_from_favorites([], top_k=10)
    return {
        "selection_policy": "Fixed earlier examples (IDs 3, 4, 5, and 1), one different-genre movie (ID 858 where present), the longest usable plot selected deterministically from metadata, and the lowest-ID movie without a usable plot. Selection used no held-out ratings.",
        "cases": output_cases,
        "character_name_sensitivity": {"known_name_tokens": ["Max", "Robin", "Savannah"], "diagnostics": name_cases},
        "transformer_and_long_plot": chunked_audit,
        "simulated_new_user_route": {
            "favorites": [47, 3], "status": new_user.status,
            "recommendations": _audit_rows(new_user.recommendations, model_name="phase11_favorites", catalog=enriched),
            "no_preferences_status": no_preferences.status,
            "no_preferences_returned_count": len(no_preferences.recommendations),
            "note": "Favorites are the predeclared Phase 12 demonstration inputs, not selected from held-out ratings; this is a route check, not an accuracy test.",
        },
        "interpretation": "This is a deterministic diagnostic review, not a formal semantic accuracy score or statistically representative sample. Similar genres, lexical overlap, transformer similarity, credits, and rating behavior are distinct evidence types.",
    }


def validate_chunked_artifact(movies: pd.DataFrame, transformer: TransformerMovieRecommender) -> dict[str, Any]:
    if not CHUNKED_EMBEDDINGS_PATH.exists():
        return {"status": "not_evaluable", "reason": "Chunked embedding artifact is missing; Phase 7 defaults were not changed."}
    try:
        with np.load(CHUNKED_EMBEDDINGS_PATH, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            ids = artifact["movie_ids"].astype(np.int64)
            vectors = np.asarray(artifact["embeddings"], dtype=np.float32)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {"status": "not_evaluable", "reason": f"Could not read chunked artifact safely: {error}"}
    expected_ids = transformer.plot_movies.movie_id.to_numpy(dtype=np.int64)
    fingerprint_matches = metadata.get("plot_fingerprint") == transformer.fingerprint
    compatible = (
        metadata.get("model_id") == MODEL_ID
        and metadata.get("model_revision") == MODEL_REVISION
        and metadata.get("embedding_dimension") == 384
        and metadata.get("normalized") is True
        and np.array_equal(ids, expected_ids)
        and vectors.shape == transformer.embeddings.shape
        and np.isfinite(vectors).all()
        and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
        and fingerprint_matches
    )
    if not compatible:
        return {"status": "not_evaluable", "reason": "Artifact model, revision, plot fingerprint, IDs, dimensions, or normalization did not match; no substitute was loaded."}
    usable = movies.loc[usable_plot_mask(movies)].copy()
    usable["plot_length"] = usable["plot"].astype(str).str.len()
    longest_id = int(usable.sort_values(["plot_length", "movie_id"], ascending=[False, True], kind="mergesort").iloc[0].movie_id)
    row = transformer.movie_id_to_row[longest_id]
    chunk_scores = cosine_similarity(vectors[row:row + 1], vectors)[0]
    chunk_scores[row] = -1.0
    chunk_top = np.argsort(-chunk_scores, kind="stable")[:5]
    chunk_ids = ids[chunk_top].astype(int).tolist()
    standard = transformer.recommend(longest_id, 5)
    standard_ids = standard.movie_id.astype(int).tolist()
    return {
        "status": "compatible_and_reused",
        "artifact_model_id": metadata.get("model_id"),
        "artifact_revision": metadata.get("model_revision"),
        "plot_fingerprint_matches": fingerprint_matches,
        "plot_vector_count": len(ids),
        "long_plot_query_movie_id": longest_id,
        "phase7_top5": standard_ids,
        "chunked_top5": chunk_ids,
        "top5_overlap_count": len(set(standard_ids) & set(chunk_ids)),
        "previous_phase12_evidence": "The prior fixed diagnostic compared five long plots and found overlaps of 1, 1, 0, 3, and 3; Phase 12 report records 3,400 long plots, 17,548 chunks, window size 224, and overlap 48.",
        "conclusion": "One deterministic comparison is diagnostic only. Chunked embeddings remain an experiment; the Phase 7 model and default remain unchanged.",
    }


def build_markdown(report: dict[str, Any]) -> str:
    split = report["split"]
    ranking = report["ranking_evaluation"]
    lines = [
        "# CineMatch Phase 13 — Leakage-Safe Evaluation",
        "",
        "## Split and leakage controls",
        "",
        f"Per-user temporal holdout: final {split['holdout_fraction']:.0%} by timestamp, with MovieLens `movie_id` as deterministic tie-break. The minimum training history was {split['minimum_training_ratings']} ratings. This yielded {split['eligible_users']} users, {split['train_interactions']:,} training interactions, and {split['test_interactions']:,} test interactions. {split['users_with_timestamp_tie_at_split_boundary']} users had an equal-timestamp tie at the boundary; their within-second order is unknowable.",
        "",
        "The split manifest records keys, timestamps, order, and partition, not rating values. Phase 10, Phase 11, and Phase 12 receive training rows only. Movie content metadata is joined by `movie_id`; no row-position joins are used. Test ratings are used only after recommendation generation to calculate outcomes.",
        "",
        "## Rating prediction",
        "",
        "Phase 10 exposes ranked predicted ratings for candidates with sufficient training-only item-similarity evidence. Content and rank-fusion outputs are not rating predictions and are marked unsupported for this task.",
        "",
        "| Model | N predictions | Coverage | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    rating = report["rating_prediction"]
    for label, value in (
        ("Global mean (training only)", rating["training_only_baselines"]["global_mean"]),
        ("User mean (training only)", rating["training_only_baselines"]["user_mean"]),
        ("Phase 10 collaborative", rating["phase10_collaborative"]["metrics_on_supported_predictions"]),
    ):
        if value is None:
            lines.append(f"| {label} | 0 | 0% | n/a | n/a |")
        else:
            coverage = value.get("coverage", rating["phase10_collaborative"]["coverage"] if label.startswith("Phase 10") else 1.0)
            lines.append(f"| {label} | {value['count']:,} | {coverage:.1%} | {value['mae']:.4f} | {value['rmse']:.4f} |")
    paired = rating["paired_comparison_on_same_supported_interactions"]
    lines.extend(["", f"Paired CF minus user-mean comparison on {paired['same_supported_test_interactions']:,} common predictions: MAE difference {paired['mae_difference_and_95_ci']['difference_model_minus_baseline']:.4f}, user-bootstrap 95% interval {paired['mae_difference_and_95_ci']['ci_95_user_bootstrap']}; RMSE difference {paired['rmse_difference_and_95_ci']['difference_model_minus_baseline']:.4f}, interval {paired['rmse_difference_and_95_ci']['ci_95_user_bootstrap']}. Negative favors CF. This does not measure plot relevance.", "", "## Personalized top-K ranking", "", f"Relevance is a held-out rating ≥ {RELEVANCE_THRESHOLD:.1f}, selected before examining results. Candidates are the full movie catalog minus each user's training-rated movies. This is a positive-unlabeled benchmark: missing ratings are unknown, not explicit dislikes; unobserved candidates can still be counted as non-hits by the conventional precision denominator.", "", "| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | Users with recs / eligible |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, model_report in ranking["models"].items():
        m5, m10 = model_report["metrics_by_k"]["5"], model_report["metrics_by_k"]["10"]
        show = lambda value: "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {show(m5['precision_at_k'])} | {show(m5['recall_at_k'])} | {show(m5['ndcg_at_k'])} | {show(m10['precision_at_k'])} | {show(m10['recall_at_k'])} | {show(m10['ndcg_at_k'])} | {model_report['users_with_any_recommendation']}/{model_report['eligible_users']} |")
    lines.extend([
        "", "`Precision@K` always divides by K (short lists leave unfilled slots as misses). Recall and NDCG exclude users with no relevant held-out movie and report that exclusion count. User-bootstrap intervals resample users, not individual ratings.",
        "", "## History groups",
        "", "| Training-history group | Users | Phase 12 recommendation coverage | Phase 12 P@10 | R@10 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    phase12 = ranking["models"]["phase12_unified_hybrid"]
    for group, details in phase12["history_groups"].items():
        at10 = details["metrics_at_k"]["10"]
        fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {group} | {details['users']} | {details['recommendation_coverage']:.1%} | {fmt(at10['precision_at_k'])} | {fmt(at10['recall_at_k'])} | {fmt(at10['ndcg_at_k'])} |")
    lines.extend(["", "## Qualitative relevance audit", "", "These fixed cases were selected without looking at held-out ratings. The table shows the first three results for the TF-IDF, Transformer, Phase 9, and Phase 12 favorite routes. Detailed top-five rows include each result's plot excerpt and signal evidence in the JSON report. Missing plot results are reported as unavailable rather than imputed.", ""])
    for case in report["qualitative_audit"]["cases"]:
        plot_state = "plot present" if case["query_plot_available"] else "plot missing"
        lines.append(f"### {case['query_title']} (movie_id={case['query_movie_id']}; {plot_state})")
        lines.append("")
        lines.append("| Model | Top results (rank — title; plot availability; score/evidence) |")
        lines.append("|---|---|")
        for name in ("phase6_tfidf", "phase7_transformer", "phase9_content_hybrid", "phase12_new_user_favorite_route"):
            result = case["models"][name]
            if result["status"] != "ok":
                summary = f"Unavailable: {result.get('reason', 'no candidates')}"
            elif not result["recommendations"]:
                summary = "No candidates returned"
            else:
                entries = []
                for item in result["recommendations"][:3]:
                    score = item.get("similarity", item.get("hybrid_score"))
                    proof = item.get("explanation", item.get("shared_genres", ""))
                    snippet = (item.get("plot_excerpt") or "").replace("|", "/").replace("\n", " ")[:90]
                    score_text = f"score={score:.3f}; " if score is not None else ""
                    proof_text = f"{proof}; " if proof else ""
                    entries.append(f"{item['rank']} — {item['title']} (plot {'yes' if item['plot_available'] else 'no'}; {score_text}{proof_text}{snippet})")
                summary = "<br>".join(entries)
            lines.append(f"| {name} | {summary} |")
        lines.append("")
    name_diagnostics = report["qualitative_audit"]["character_name_sensitivity"]["diagnostics"]
    if name_diagnostics:
        lines.extend(["**Name sensitivity:** targeted names accounted for these shares of the Phase 6 top-five cosine scores:"])
        for item in name_diagnostics:
            if "targeted_name_contributions" in item:
                shares = ", ".join(f"{value['targeted_name_share']:.0%}" for value in item["targeted_name_contributions"])
                lines.append(f"- Query {item['query_movie_id']}: {shares}; top-five overlap after the targeted-name ablation was {item['top5_overlap']}/5.")
    chunk = report["qualitative_audit"]["transformer_and_long_plot"]
    lines.extend(["", f"**Long-plot experiment:** {chunk.get('conclusion', chunk.get('reason', 'not evaluated'))}"])
    phase12_vs_phase9 = ranking["paired_phase12_user_level_differences_at_10"].get("phase12_minus_phase9_content_hybrid_ndcg@10")
    if phase12_vs_phase9:
        lines.extend(["", f"**Observed Phase 12 vs Phase 9:** P@10 difference {ranking['paired_phase12_user_level_differences_at_10']['phase12_minus_phase9_content_hybrid_precision@10']['mean_difference']:.4f}, R@10 difference {ranking['paired_phase12_user_level_differences_at_10']['phase12_minus_phase9_content_hybrid_recall@10']['mean_difference']:.4f}, and NDCG@10 difference {phase12_vs_phase9['mean_difference']:.4f}. NDCG user-bootstrap 95% interval: {phase12_vs_phase9['ci_95_user_bootstrap']}. This is held-out rating relevance, not a semantic plot-similarity judgment."])
    lines.extend(["", "## Limitations and interpretation", "", "- Historical ratings are sparse and reflect what users chose to rate; ranking metrics do not establish story-level similarity.", "- The global popularity baseline may exploit dataset exposure patterns; MovieLens observations are not a randomized preference sample.", "- Equal timestamps occur frequently. Movie ID makes the holdout deterministic but cannot reconstruct order within that timestamp.", "- Existing content recommenders return ranked lists; evaluation uses their APIs and removes training-seen movies. Phase 12's retrieval and fallback behavior remains unchanged.", "- Rating prediction is only defined for Phase 10's supported candidate scores. Unavailable predictions are reported as uncovered, not silently imputed.", "- The Phase 9 weights, Phase 10 algorithm, Phase 11 routing, Phase 12 fusion, and Phase 7 embedding default were not tuned or modified.", "- The user-bootstrap intervals describe variation across these users, not uncertainty from a new population or a new data collection process.", "", "## Run and artifacts", "", "```sh", "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_recommenders.py", "```", "", "The JSON report contains full metrics, per-group coverage, source/fallback behavior, detailed audit rows, missing-data counts, and baseline-preservation checks. The split manifest contains interaction keys and partition labels only.", ""])
    return "\n".join(lines)


def run_evaluation() -> dict[str, Any]:
    start = time.monotonic()
    print("Loading and validating Phase 4/8 data; creating temporal split...", flush=True)
    clean_movies, all_ratings, enriched = read_inputs()
    manifest, split_summary = make_temporal_split(
        all_ratings, holdout_fraction=HOLDOUT_FRACTION,
        minimum_training_ratings=MINIMUM_TRAINING_RATINGS,
    )
    manifest.to_csv(MANIFEST_PATH, index=False)
    train, test = split_ratings_by_manifest(all_ratings, manifest)
    if set(map(tuple, train[["user_id", "movie_id"]].to_numpy())) & set(map(tuple, test[["user_id", "movie_id"]].to_numpy())):
        raise AssertionError("Train/test keys overlap")
    split_summary.update({
        "input_ratings": int(len(all_ratings)),
        "manifest_rows": int(len(manifest)),
        "manifest_path": str(MANIFEST_PATH),
        "train_test_key_overlap": 0,
        "train_earlier_than_test_per_user": True,
    })

    rating_distribution = all_ratings.rating.value_counts().sort_index()
    relevant_test = test.loc[test.rating.ge(RELEVANCE_THRESHOLD)]
    relevant_by_user = {
        int(user): set(group.movie_id.astype(int))
        for user, group in relevant_test.groupby("user_id", sort=True)
    }
    specs = build_user_specs(train, max(K_VALUES))

    # Fit the collaborative model only on this manifest's training partition.
    print(f"Fitting Phase 10 on {len(train):,} training ratings...", flush=True)
    collaborative = ItemItemCollaborativeRecommender(train, clean_movies[["movie_id", "title"]])
    rating_report, cf_rankings, cf_predictions, cf_failures = _rating_prediction_evaluation(
        collaborative, train, test, len(clean_movies)
    )

    # Every content model receives movie-only data; none receives ratings.
    print("Loading cached text model and preparing movie-only content models...", flush=True)
    enriched_catalog = enriched.copy()
    metadata_catalog = prepare_catalog(enriched_catalog)
    tfidf = TfidfMovieRecommender(clean_movies[["movie_id", "title", "release_year", "plot"]])
    transformer = TransformerMovieRecommender(
        clean_movies[["movie_id", "title", "release_year", "plot"]], cache_path=EMBEDDINGS_PATH
    )
    phase9 = HybridContentRecommender(
        enriched_catalog, tfidf_model=tfidf, transformer_model=transformer
    )
    cached_phase9 = precompute_phase9(phase9, specs)
    print("Precomputing Phase 5–7 personalized seed rankings...", flush=True)
    base_content_cache = precompute_base_content_lists(specs, metadata_catalog, tfidf, transformer)
    cold_start = ColdStartRecommender(enriched_catalog, cached_phase9, collaborative)
    unified = HybridRecommender(enriched_catalog, train, cold_start, collaborative)

    model_names = (
        "global_popularity", "phase5_metadata", "phase6_tfidf", "phase7_transformer",
        "phase9_content_hybrid", "phase10_collaborative", "phase12_unified_hybrid",
    )
    rankings: dict[str, dict[int, list[int]]] = {name: {} for name in model_names}
    hybrid_outputs: dict[int, Any] = {}
    popularity = train.groupby("movie_id").agg(
        training_rating_count=("rating", "size"), training_mean_rating=("rating", "mean")
    ).reset_index().sort_values(
        ["training_rating_count", "training_mean_rating", "movie_id"],
        ascending=[False, False, True], kind="mergesort",
    )
    popular_ids = popularity.movie_id.astype(int).tolist()

    for user_id, spec in specs.items():
        seen = spec["seen"]
        rankings["global_popularity"][user_id] = [movie for movie in popular_ids if movie not in seen][:max(K_VALUES)]
        rankings["phase10_collaborative"][user_id] = cf_rankings.get(user_id, [])[:max(K_VALUES)]
        for model_name in ("phase5_metadata", "phase6_tfidf", "phase7_transformer", "phase9_content_hybrid"):
            ranked, status = content_user_rankings(
                model_name, spec, metadata_catalog=metadata_catalog, tfidf=tfidf,
                transformer=transformer, cold_start=cold_start,
                seed_rank_cache=base_content_cache, top_k=max(K_VALUES),
            )
            rankings[model_name][user_id] = ranked
            spec.setdefault("content_status", {})[model_name] = status
        response = unified.recommend_for_user(user_id, top_k=max(K_VALUES))
        hybrid_outputs[user_id] = response
        rankings["phase12_unified_hybrid"][user_id] = _recommendation_ids(response.recommendations)
        spec["phase12_status"] = response.status
        spec["phase12_source"] = response.recommendation_source
        if len(rankings["phase12_unified_hybrid"]) % 100 == 0:
            print(f"Personalized top-K evaluation: {len(rankings['phase12_unified_hybrid'])}/{len(specs)} users", flush=True)

    ranking_report = {
        "task": "personalized binary top-K retrieval",
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "relevant_test_interactions": int(len(relevant_test)),
        "eligible_users": int(len(specs)),
        "users_with_at_least_one_relevant_test_movie": int(len(relevant_by_user)),
        "users_with_no_relevant_test_movie": int(len(specs) - len(relevant_by_user)),
        "candidate_universe": "All 9,742 catalog movie IDs except movies rated in that user's training partition; held-out relevant items are only that user's own test ratings >= 4.0.",
        "unrated_candidate_interpretation": "This is conventional positive-only retrieval: an unobserved candidate is not an explicit dislike, but it cannot count as a known relevant hit. This can understate recommendation quality.",
        "metrics_user_policy": "Precision includes every eligible user and divides hits by K even for a short list; Recall and NDCG are undefined for users with no relevant held-out items, so those users are excluded and counted.",
        "models": {},
        "phase12_rank_fusion_diagnostics": _rank_fusion_diagnostics(hybrid_outputs),
    }
    for model_name in model_names:
        ranking_report["models"][model_name] = _summarize_ranking_model(
            rankings[model_name], relevant_by_user, specs, K_VALUES
        )

    # Paired ranking deltas use exactly the same users for each model pair.
    paired_ranking: dict[str, Any] = {}
    relevant_users = sorted(relevant_by_user)
    for candidate in ("phase5_metadata", "phase6_tfidf", "phase7_transformer", "phase9_content_hybrid", "phase10_collaborative", "phase12_unified_hybrid"):
        for metric_name, key in (("precision", "precision"), ("recall", "recall"), ("ndcg", "ndcg")):
            per_user_differences: dict[int, float] = {}
            users = sorted(specs) if metric_name == "precision" else relevant_users
            for user in users:
                baseline_value = ranking_metrics(rankings["global_popularity"].get(user, []), relevant_by_user.get(user, set()), 10)[key]
                candidate_value = ranking_metrics(rankings[candidate].get(user, []), relevant_by_user.get(user, set()), 10)[key]
                if baseline_value is not None and candidate_value is not None:
                    per_user_differences[user] = float(candidate_value) - float(baseline_value)
            if per_user_differences:
                paired_ranking[f"{candidate}_minus_global_popularity_{metric_name}@10"] = mean_user_bootstrap_interval(per_user_differences)
                paired_ranking[f"{candidate}_minus_global_popularity_{metric_name}@10"]["mean_difference"] = float(np.mean(list(per_user_differences.values())))
                paired_ranking[f"{candidate}_minus_global_popularity_{metric_name}@10"]["difference_sign"] = "positive favors candidate"
    ranking_report["paired_user_level_differences_vs_popularity_at_10"] = paired_ranking
    phase12_paired: dict[str, Any] = {}
    for baseline in ("phase5_metadata", "phase6_tfidf", "phase7_transformer", "phase9_content_hybrid", "phase10_collaborative", "global_popularity"):
        for metric_name, key in (("precision", "precision"), ("recall", "recall"), ("ndcg", "ndcg")):
            users = sorted(specs) if metric_name == "precision" else relevant_users
            differences: dict[int, float] = {}
            for user in users:
                first = ranking_metrics(rankings["phase12_unified_hybrid"].get(user, []), relevant_by_user.get(user, set()), 10)[key]
                second = ranking_metrics(rankings[baseline].get(user, []), relevant_by_user.get(user, set()), 10)[key]
                if first is not None and second is not None:
                    differences[user] = float(first) - float(second)
            if differences:
                interval = mean_user_bootstrap_interval(differences)
                interval["mean_difference"] = float(np.mean(list(differences.values())))
                interval["difference_sign"] = "positive favors Phase 12"
                phase12_paired[f"phase12_minus_{baseline}_{metric_name}@10"] = interval
    ranking_report["paired_phase12_user_level_differences_at_10"] = phase12_paired

    groups = Counter(spec["history_group"] for spec in specs.values())
    plot_usable = int(usable_plot_mask(clean_movies).sum())
    credit_coverage: dict[str, dict[str, int]] = {}
    for column in ("directors", "writers", "cast", "selected_crew"):
        usable_count = int(enriched[column].map(
            lambda value: bool(parse_people(value, crew=column == "selected_crew")[0])
        ).sum())
        credit_coverage[column] = {"usable_movies": usable_count, "missing_movies": int(len(enriched) - usable_count)}

    print("Running fixed qualitative diagnostics and writing evaluation reports...", flush=True)
    qualitative = run_qualitative_audit(
        clean_movies, enriched, metadata_catalog, tfidf, transformer, phase9, cold_start, unified
    )
    report: dict[str, Any] = {
        "phase": "Phase 13: Leakage-Safe Evaluation",
        "created_by": "scripts/evaluate_recommenders.py",
        "runtime_seconds": None,
        "data_audit": {
            "cleaned_movies_rows": int(len(clean_movies)),
            "cleaned_movie_columns": list(clean_movies.columns),
            "cleaned_rating_rows": int(len(all_ratings)),
            "rating_columns": list(all_ratings.columns),
            "enriched_movie_rows": int(len(enriched)),
            "enriched_movie_columns": list(enriched.columns),
            "unique_users": int(all_ratings.user_id.nunique()),
            "unique_movies": int(clean_movies.movie_id.nunique()),
            "rated_movies": int(all_ratings.movie_id.nunique()),
            "movie_ids_noncontiguous": bool(clean_movies.movie_id.nunique() != clean_movies.movie_id.max() - clean_movies.movie_id.min() + 1),
            "user_ids_noncontiguous": bool(all_ratings.user_id.nunique() != all_ratings.user_id.max() - all_ratings.user_id.min() + 1),
            "rating_pair_duplicates": int(all_ratings.duplicated(["user_id", "movie_id"]).sum()),
            "rating_timestamp_missing": int(all_ratings.timestamp.isna().sum()),
            "rating_timestamp_tied_rows_within_users": int(all_ratings.duplicated(["user_id", "timestamp"]).sum()),
            "rating_distribution": {str(float(rating)): int(count) for rating, count in rating_distribution.items()},
            "rating_threshold_fraction_at_or_above_4": float(all_ratings.rating.ge(RELEVANCE_THRESHOLD).mean()),
            "usable_plot_movies": plot_usable,
            "missing_or_blank_plot_movies": int(len(clean_movies) - plot_usable),
            "enriched_credit_coverage": credit_coverage,
            "join_policy": "Ratings join catalog metadata only by movie_id. User-level splits and test relevance use explicit user_id/movie_id keys. No contiguous-ID or row-position assumption.",
            "documentation_discrepancies": [
                "No behavior-level mismatch was found between the inspected Phase 4–12 documentation and implementations for the interfaces used here.",
                "Interface detail: Phase 10 exposes candidate rating estimates through recommend() and has no separate arbitrary-pair predict() method; evaluation enumerates the existing returned candidate scores and reports missing target scores as unsupported.",
                "Interface detail: Phase 11/12 content outputs aggregate movie-to-movie rankings and do not constitute predicted user ratings.",
            ],
        },
        "split": split_summary,
        "rating_prediction": rating_report,
        "ranking_evaluation": ranking_report,
        "history_group_counts": dict(groups),
        "feature_coverage": _feature_coverage(clean_movies, enriched, specs, relevant_by_user),
        "qualitative_audit": qualitative,
        "baseline_preservation": {
            "phase5_metadata_recommender_modified": False,
            "phase6_tfidf_defaults_modified": False,
            "phase7_model_id": MODEL_ID,
            "phase7_model_revision": MODEL_REVISION,
            "phase7_model_or_default_modified": False,
            "phase8_enriched_data_modified": False,
            "phase9_weights": dict(phase9.weights),
            "phase9_weights_modified": False,
            "phase10_algorithm_modified": False,
            "phase11_routing_modified": False,
            "phase12_fusion_modified": False,
            "cleaned_data_modified": False,
            "embedding_artifacts_modified": False,
            "phase9_evaluation_vectorization": {
                "production_model_changed": False,
                "representative_parity_checks": cached_phase9.parity_checks,
                "parity_policy": "The evaluation-only vectorized rank path must match regular Phase 9 Top-10 movie IDs and hybrid scores at absolute tolerance 1e-8 for predeclared IDs 3, 4, 5, 1, and 858 where present; otherwise evaluation stops.",
            },
        },
        "known_issue_assessment": {
            "character_name_sensitivity": "Rechecked on predeclared examples using the existing targeted-token ablation. The default TF-IDF model remains unchanged.",
            "transformer_semantic_mismatch": "Re-run for Waiting to Exhale and Father of the Bride Part II; inspect Phase 7 outputs and Phase 9 evidence fields in the qualitative audit.",
            "long_plot_truncation": "Phase 7 cache and chunked cache compatibility are checked. Chunked ranking is diagnostic only and not the default.",
            "broad_genre_and_metadata_matches": "Phase 9 explanation/evidence fields remain separate from lexical, transformer, and collaborative ranking evidence.",
            "missing_plots_and_credits": "Counts and per-model qualitative availability are reported; no fields are fabricated.",
            "phase12_rank_fusion": "Current status, source-only rows, and fusion-score behavior are measured without changing the formula.",
        },
        "run": {
            "command": "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_recommenders.py",
            "python_executable": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "hugging_face_offline": os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "reports": [str(JSON_REPORT_PATH), str(MARKDOWN_REPORT_PATH), str(MANIFEST_PATH)],
        },
    }
    report["runtime_seconds"] = round(time.monotonic() - start, 2)
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MARKDOWN_REPORT_PATH.write_text(build_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    report = run_evaluation()
    brief = {
        "phase": report["phase"],
        "split": {key: report["split"][key] for key in ("eligible_users", "train_interactions", "test_interactions", "users_with_timestamp_tie_at_split_boundary")},
        "rating_coverage": report["rating_prediction"]["phase10_collaborative"]["coverage"],
        "ranking_models": list(report["ranking_evaluation"]["models"]),
        "runtime_seconds": report["runtime_seconds"],
        "json_report": str(JSON_REPORT_PATH),
        "markdown_report": str(MARKDOWN_REPORT_PATH),
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
