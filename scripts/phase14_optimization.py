"""Leakage-safe Phase 14 validation experiments for CineMatch.

This script keeps the Phase 13 test partition closed. It creates a nested
temporal validation split from Phase 13 training rows, then compares existing
recommendation baselines with a small, predeclared fusion variant and two
diagnostic content variants. It does not change production defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_recommenders import (
    CHUNKED_EMBEDDINGS_PATH,
    EMBEDDINGS_PATH,
    CachedContentRanker,
    _recommendation_ids,
    _summarize_ranking_model,
    build_user_specs,
    content_user_rankings,
    history_group,
    positive_seed_ids,
)
from scripts.phase12_experiments import mask_diagnostic_names
from scripts.recommend_cold_start import ColdStartRecommender
from scripts.recommend_collaborative import ItemItemCollaborativeRecommender
from scripts.recommend_hybrid import HybridRecommender
from scripts.recommend_hybrid_content import HybridContentRecommender, load_enriched_movies, parse_people
from scripts.recommend_metadata import prepare_catalog, recommend_similar_movies, split_labels
from scripts.recommend_transformer import (
    EXPECTED_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    TransformerMovieRecommender,
    load_movies as load_text_movies,
    plot_fingerprint,
)
from scripts.recommend_tfidf import TfidfMovieRecommender
from scripts.recommender_evaluation import (
    make_temporal_split,
    mean_user_bootstrap_interval,
    ranking_metrics,
    split_ratings_by_manifest,
    user_bootstrap_error_difference,
)


ROOT = PROJECT_ROOT
CLEAN_MOVIES = ROOT / "data" / "cleaned" / "movies.csv"
RATINGS = ROOT / "data" / "cleaned" / "ratings.csv"
ENRICHED = ROOT / "data" / "enriched" / "movies.csv"
PHASE13_MANIFEST = ROOT / "reports" / "phase13_split_manifest.csv"
PHASE13_JSON = ROOT / "reports" / "phase13_evaluation.json"
VALIDATION_MANIFEST = ROOT / "reports" / "phase14_validation_manifest.csv"
JSON_REPORT = ROOT / "reports" / "phase14_optimization.json"
MARKDOWN_REPORT = ROOT / "reports" / "phase14_optimization.md"

HOLDOUT_FRACTION = 0.20
MINIMUM_TRAINING_RATINGS = 5
RELEVANCE_THRESHOLD = 4.0
K_VALUES = (5, 10)
BOOTSTRAP_SEED = 14
BOOTSTRAP_RESAMPLES = 1000
PROPOSED_WEIGHTS = {"collaborative": 0.25, "content": 0.75}
FIXED_AUDIT_IDS = (3, 4, 5, 1445, 1, 858)
CONTENT_RANKING_VARIANTS = ("phase9_content_hybrid",)


def _read_validation_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Derive Phase 14 data strictly from rows marked train in Phase 13."""
    phase13_report = json.loads(PHASE13_JSON.read_text())
    saved_manifest = pd.read_csv(PHASE13_MANIFEST)
    if not {"user_id", "movie_id", "timestamp", "partition"}.issubset(saved_manifest.columns):
        raise ValueError("Phase 13 manifest schema is not the expected key/timestamp/partition schema")
    phase13_train_keys = saved_manifest.loc[
        saved_manifest.partition.eq("train"), ["user_id", "movie_id", "timestamp"]
    ]
    phase13_test_keys = saved_manifest.loc[
        saved_manifest.partition.eq("test"), ["user_id", "movie_id", "timestamp"]
    ]
    all_ratings = pd.read_csv(RATINGS)
    # The join is restricted to Phase 13 training keys before rating values are
    # used. Phase 13 test ratings never enter the Phase 14 split or models.
    phase13_train = all_ratings.merge(
        phase13_train_keys,
        on=["user_id", "movie_id", "timestamp"],
        how="inner",
        validate="one_to_one",
    )
    if len(phase13_train) != int(phase13_report["split"]["train_interactions"]):
        raise ValueError("Phase 13 training rows do not match the saved report")

    validation_manifest, validation_summary = make_temporal_split(
        phase13_train,
        holdout_fraction=HOLDOUT_FRACTION,
        minimum_training_ratings=MINIMUM_TRAINING_RATINGS,
    )
    validation_keys = set(map(tuple, validation_manifest[["user_id", "movie_id"]].to_numpy()))
    test_keys = set(map(tuple, phase13_test_keys[["user_id", "movie_id"]].to_numpy()))
    train_keys = set(map(tuple, phase13_train_keys[["user_id", "movie_id"]].to_numpy()))
    if validation_keys & test_keys:
        raise AssertionError("Phase 14 validation overlaps the Phase 13 test partition")
    if not validation_keys.issubset(train_keys):
        raise AssertionError("Phase 14 manifest contains an interaction outside Phase 13 training")

    train, validation = split_ratings_by_manifest(phase13_train, validation_manifest)
    if set(map(tuple, train[["user_id", "movie_id"]].to_numpy())) & set(
        map(tuple, validation[["user_id", "movie_id"]].to_numpy())
    ):
        raise AssertionError("Phase 14 training and validation interactions overlap")
    summary = {
        **validation_summary,
        "phase13_train_interactions": int(len(phase13_train)),
        "phase13_test_interactions": int(len(phase13_test_keys)),
        "phase14_train_interactions": int(len(train)),
        "phase14_validation_interactions": int(len(validation)),
        "phase14_validation_users": int(validation.user_id.nunique()),
        "phase13_test_overlap": 0,
        "all_phase14_keys_within_phase13_train": True,
        "saved_manifest_partition_labels": ["train", "validation"],
        "validation_rule": "Per-user final 20% within Phase 13 training rows ordered by timestamp then movie_id; minimum Phase 14 training history 5.",
    }
    VALIDATION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    saved_manifest = validation_manifest.copy()
    saved_manifest["partition"] = saved_manifest["partition"].replace({"test": "validation"})
    saved_manifest.to_csv(VALIDATION_MANIFEST, index=False)
    return train, validation, validation_manifest, summary


def _load_validated_transformer_artifacts(
    text_movies: pd.DataFrame,
) -> tuple[TransformerMovieRecommender, TransformerMovieRecommender, dict[str, Any]]:
    """Load both pinned local caches after checking identity, alignment and vectors."""
    plot_rows = text_movies.loc[
        text_movies["plot"].notna() & text_movies["plot"].astype("string").str.strip().ne("")
    ].copy().reset_index(drop=True)
    expected_ids = plot_rows.movie_id.astype("int64").to_numpy()
    expected_fingerprint = plot_fingerprint(plot_rows)

    def load_artifact(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            ids = artifact["movie_ids"].astype("int64")
            vectors = artifact["embeddings"].astype("float32")
        if not np.array_equal(ids, expected_ids):
            raise ValueError(f"Embedding movie IDs do not align with source plots: {path.name}")
        if vectors.shape != (len(expected_ids), EXPECTED_DIMENSION):
            raise ValueError(f"Unexpected embedding dimensions in {path.name}: {vectors.shape}")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError(f"Embeddings are not finite, normalized vectors: {path.name}")
        if metadata.get("model_id") != MODEL_ID or metadata.get("model_revision") != MODEL_REVISION:
            raise ValueError(f"Embedding model or pinned revision mismatch in {path.name}")
        if metadata.get("plot_fingerprint") != expected_fingerprint:
            raise ValueError(f"Embedding plot fingerprint mismatch in {path.name}")
        if metadata.get("embedding_dimension") != EXPECTED_DIMENSION or metadata.get("normalized") is not True:
            raise ValueError(f"Embedding metadata mismatch in {path.name}")
        return metadata, ids, vectors

    original_meta, original_ids, original_vectors = load_artifact(EMBEDDINGS_PATH)
    chunk_meta, chunk_ids, chunk_vectors = load_artifact(CHUNKED_EMBEDDINGS_PATH)
    if not np.array_equal(original_ids, chunk_ids):
        raise ValueError("Original and chunked cache movie IDs differ")
    if chunk_meta.get("chunk_size") != 224 or chunk_meta.get("overlap") != 48:
        raise ValueError("Chunked embedding cache does not match the documented window settings")

    original = TransformerMovieRecommender(text_movies, embeddings=original_vectors)
    chunked = TransformerMovieRecommender(text_movies, embeddings=chunk_vectors)
    storage = {
        "original_cache_bytes": EMBEDDINGS_PATH.stat().st_size,
        "chunked_cache_bytes": CHUNKED_EMBEDDINGS_PATH.stat().st_size,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "plot_fingerprint_match": True,
        "movie_id_alignment": True,
        "embedding_dimension": EXPECTED_DIMENSION,
        "plot_vectors": int(len(expected_ids)),
        "chunk_size_wordpieces": 224,
        "overlap_wordpieces": 48,
        "long_plot_count": int(chunk_meta["long_plot_count"]),
        "chunks_encoded": int(chunk_meta["chunk_count"]),
        "short_plot_vectors_reused": int(chunk_meta["short_plot_vectors_reused_from_phase7"]),
        "cache_metadata_original": original_meta,
        "cache_metadata_chunked": chunk_meta,
    }
    return original, chunked, storage


def _make_phase9_ranking_cache(
    model: HybridContentRecommender,
    specs: dict[int, dict[str, Any]],
    *,
    label: str,
) -> CachedContentRanker:
    """Cache each validation positive-seed query once for a content variant."""
    limits: dict[int, int] = {}
    for spec in specs.values():
        for seed_id in spec["seeds"]:
            limits[seed_id] = max(limits.get(seed_id, 0), spec["pool_size"] + len(spec["seeds"]))
    cached = CachedContentRanker(model, catalog_ids=model.catalog.movie_id.astype(int).tolist())
    for count, (seed_id, limit) in enumerate(sorted(limits.items()), start=1):
        cached.add(seed_id, limit)
        if count % 100 == 0:
            print(f"{label} seed-rank cache: {count}/{len(limits)}", flush=True)
    for seed_id in (3, 4, 5, 1, 858):
        if seed_id not in model.catalog_ids:
            continue
        fast = cached._vectorized_recommend(seed_id, 10)
        regular = model.recommend(seed_id, 10)
        if fast.movie_id.astype(int).tolist() != regular.movie_id.astype(int).tolist() or not np.allclose(
            fast.hybrid_score.to_numpy(dtype=float), regular.hybrid_score.to_numpy(dtype=float), atol=1e-8, rtol=1e-8
        ):
            raise AssertionError(f"{label} rank cache disagrees with its normal recommender API for {seed_id}")
    return cached


def _weighted_rank_fusion(
    collaborative_ids: list[int],
    content_ids: list[int],
    *,
    weights: dict[str, float],
    top_k: int,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Fuse ordinal ranks, renormalizing only when an entire source is absent."""
    available = {
        "collaborative": bool(collaborative_ids),
        "content": bool(content_ids),
    }
    denominator = sum(weights[name] for name, exists in available.items() if exists)
    if denominator <= 0:
        return [], {}
    source_ranks: dict[int, dict[str, int]] = {}
    for name, movie_ids in (("collaborative", collaborative_ids), ("content", content_ids)):
        seen: set[int] = set()
        rank = 0
        for movie_id in movie_ids:
            movie_id = int(movie_id)
            if movie_id in seen:
                continue
            seen.add(movie_id)
            rank += 1
            source_ranks.setdefault(movie_id, {})[name] = rank
    scores: dict[int, float] = {}
    for movie_id, ranks in source_ranks.items():
        scores[movie_id] = sum(weights[name] / rank for name, rank in ranks.items()) / denominator
    ordered = sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))[:top_k]
    details = {
        movie_id: {
            "fusion_score": scores[movie_id],
            "source_ranks": source_ranks[movie_id],
            "contributing_sources": sorted(source_ranks[movie_id]),
        }
        for movie_id in ordered
    }
    return ordered, details


def _metric_values(
    rankings: dict[int, list[int]], relevant: dict[int, set[int]], users: list[int], k: int
) -> dict[int, dict[str, float | None]]:
    return {user: ranking_metrics(rankings.get(user, []), relevant.get(user, set()), k) for user in users}


def _paired_differences(
    rankings: dict[str, dict[int, list[int]]],
    relevant: dict[int, set[int]],
    *,
    baselines: tuple[str, ...],
    k: int = 10,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    users_all = sorted(rankings["global_popularity"])
    users_relevant = [u for u in users_all if relevant.get(u)]
    for candidate, baseline in (("phase12_weighted_content_75", "phase9_content_hybrid"),
                                ("phase12_weighted_content_75", "phase12_unified_hybrid"),
                                ("phase12_unified_hybrid", "phase9_content_hybrid"),
                                ("phase9_name_masked_diagnostic", "phase9_content_hybrid"),
                                ("phase9_chunked_diagnostic", "phase9_content_hybrid"),
                                ("phase12_weighted_content_75", "global_popularity")):
        if baseline not in baselines:
            continue
        for metric in ("precision", "recall", "ndcg"):
            users = users_all if metric == "precision" else users_relevant
            diffs: dict[int, float] = {}
            for user in users:
                a = ranking_metrics(rankings[candidate].get(user, []), relevant.get(user, set()), k)[metric]
                b = ranking_metrics(rankings[baseline].get(user, []), relevant.get(user, set()), k)[metric]
                if a is not None and b is not None:
                    diffs[user] = float(a) - float(b)
            if diffs:
                interval = mean_user_bootstrap_interval(diffs, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
                interval["mean_difference"] = float(np.mean(list(diffs.values())))
                interval["difference_sign"] = "positive favors first model"
                output[f"{candidate}_minus_{baseline}_{metric}@{k}"] = interval
    return output


def _source_diagnostics(
    source_rows: dict[int, dict[str, Any]],
    phase12: dict[int, list[int]],
    relevant: dict[int, set[int]],
    specs: dict[int, dict[str, Any]],
    phase12_outputs: dict[int, Any],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    overlap_sizes: list[int] = []
    jaccards: list[float] = []
    collab_hits = content_hits = phase12_hits = displaced_content_hits = cf_only_hits = 0
    users_with_score_ties = boundary_ties = rank1_content_retained = rank1_cf_retained = 0
    repeated_tie_candidates = 0
    supports: list[float] = []
    evidence: list[float] = []
    phase12_evidence: dict[str, list[float]] = {
        "content_mrr": [], "content_seed_support": [], "cf_support": [], "cf_evidence_weight": []
    }
    source_counts: Counter[str] = Counter()
    group_counts: dict[str, Counter[str]] = {}
    for user, row in source_rows.items():
        collab = row["collaborative"][:top_k]
        content = row["content"][:top_k]
        collab_set, content_set = set(collab), set(content)
        union = collab_set | content_set
        overlap = collab_set & content_set
        overlap_sizes.append(len(overlap))
        jaccards.append(len(overlap) / len(union) if union else 0.0)
        counters["collaborative_available"] += bool(row["collaborative"])
        counters["content_available"] += bool(row["content"])
        counters["both_available"] += bool(row["collaborative"] and row["content"])
        counters["collaborative_empty"] += not bool(row["collaborative"])
        counters["content_empty"] += not bool(row["content"])
        counters["both_empty"] += not bool(row["collaborative"] or row["content"])
        counters["no_positive_seeds"] += not bool(specs[user]["seeds"])
        group = specs[user]["history_group"]
        group_counts.setdefault(group, Counter())["users"] += 1
        group_counts[group]["both_available"] += bool(row["collaborative"] and row["content"])
        group_counts[group]["both_empty"] += not bool(row["collaborative"] or row["content"])
        heldout = relevant.get(user, set())
        collab_hits += len(collab_set & heldout)
        content_hits += len(content_set & heldout)
        phase12_set = set(phase12.get(user, [])[:top_k])
        phase12_hits += len(phase12_set & heldout)
        displaced_content_hits += len((content_set & heldout) - phase12_set)
        cf_only_hits += len((collab_set - content_set) & heldout)
        supports.extend(row["support_count"])
        evidence.extend(row["evidence_weight"])
        # Inspect equal-fusion score ties over the full returned source union.
        full_ids, all_details = _weighted_rank_fusion(
            row["collaborative"], row["content"],
            weights={"collaborative": 0.5, "content": 0.5},
            top_k=len(row["collaborative"]) + len(row["content"]),
        )
        selected = set(full_ids[:top_k])
        if row["content"] and row["content"][0] in selected:
            rank1_content_retained += 1
        if row["collaborative"] and row["collaborative"][0] in selected:
            rank1_cf_retained += 1
        visible_scores = [all_details[mid]["fusion_score"] for mid in full_ids[:top_k]]
        tie_values = Counter(visible_scores)
        repeated_tie_candidates += sum(count for count in tie_values.values() if count > 1)
        users_with_score_ties += any(count > 1 for count in tie_values.values())
        if len(full_ids) > top_k and all_details[full_ids[top_k - 1]]["fusion_score"] == all_details[full_ids[top_k]]["fusion_score"]:
            boundary_ties += 1

        response = phase12_outputs[user]
        for rec in response.recommendations.itertuples(index=False):
            source_counts[str(rec.contributing_sources)] += 1
            if pd.notna(rec.content_mean_reciprocal_rank):
                phase12_evidence["content_mrr"].append(float(rec.content_mean_reciprocal_rank))
                explanation = rec.content_explanation
                if isinstance(explanation, dict):
                    phase12_evidence["content_seed_support"].append(float(len(explanation)))
            if pd.notna(rec.support_count):
                phase12_evidence["cf_support"].append(float(rec.support_count))
            if pd.notna(rec.evidence_weight):
                phase12_evidence["cf_evidence_weight"].append(float(rec.evidence_weight))
    users = len(source_rows)
    return {
        "users": users,
        "source_availability_and_empty_counts": dict(counters),
        "top10_overlap": {
            "mean_shared_candidates": float(np.mean(overlap_sizes)) if overlap_sizes else 0.0,
            "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
            "users_with_any_overlap": int(sum(value > 0 for value in overlap_sizes)),
        },
        "equal_fusion_rank_and_tie_diagnostics": {
            "users_with_at_least_one_equal_score_tie_within_top10": int(users_with_score_ties),
            "tied_candidate_rows_within_top10": int(repeated_tie_candidates),
            "users_with_equal_score_at_top10_boundary": int(boundary_ties),
            "users_where_content_rank1_survived_top10_fusion": int(rank1_content_retained),
            "users_where_collaborative_rank1_survived_top10_fusion": int(rank1_cf_retained),
            "tie_break": "The unchanged Phase 12 implementation sorts equal fusion scores by ascending movie_id.",
        },
        "phase12_top10_source_rows": dict(source_counts),
        "phase12_returned_candidate_evidence": {
            key: {"count": len(values), "mean": float(np.mean(values)) if values else None,
                  "median": float(np.median(values)) if values else None}
            for key, values in phase12_evidence.items()
        },
        "source_availability_by_history_group": {key: dict(value) for key, value in group_counts.items()},
        "relevant_heldout_movies_in_top10_lists": {
            "collaborative": collab_hits,
            "content": content_hits,
            "phase12_equal_fusion": phase12_hits,
            "collaborative_only_candidate_hits": cf_only_hits,
            "relevant_content_candidates_displaced_from_phase12_top10": displaced_content_hits,
        },
        "collaborative_top10_evidence": {
            "candidate_rows_with_support": len(supports),
            "mean_support_count": float(np.mean(supports)) if supports else None,
            "median_support_count": float(np.median(supports)) if supports else None,
            "mean_evidence_weight": float(np.mean(evidence)) if evidence else None,
        },
    }


def _feature_coverage_diagnostics(
    clean: pd.DataFrame,
    enriched: pd.DataFrame,
    specs: dict[int, dict[str, Any]],
    relevant: dict[int, set[int]],
    rankings: dict[str, dict[int, list[int]]],
) -> dict[str, Any]:
    """Measure source availability without treating missing as a negative fact."""
    clean_by_id = clean.set_index("movie_id")
    enriched_by_id = enriched.set_index("movie_id")

    def available(movie_id: int, field: str) -> bool:
        if field == "usable_plot":
            value = clean_by_id.loc[movie_id, "plot"]
            return pd.notna(value) and bool(str(value).strip())
        if field == "genres":
            return bool(split_labels(enriched_by_id.loc[movie_id, "genres"], "|"))
        if field == "plot_tags":
            return bool(split_labels(enriched_by_id.loc[movie_id, "plot_tags"], ","))
        crew = field == "selected_crew"
        value = enriched_by_id.loc[movie_id, field]
        return bool(parse_people(value, crew=crew)[0])

    fields = ("usable_plot", "genres", "plot_tags", "directors", "writers", "cast", "selected_crew")
    catalog_coverage = {}
    for field in fields:
        count = sum(available(int(movie_id), field) for movie_id in clean.movie_id)
        catalog_coverage[field] = {"available_movies": int(count), "missing_or_unretrieved": int(len(clean) - count), "coverage": count / max(1, len(clean))}

    seed_feature_groups: dict[str, dict[int, list[int]]] = {
        "usable_plot": {1: [], 0: []},
        "any_usable_credit": {1: [], 0: []},
    }
    for user, spec in specs.items():
        has_plot = any(available(seed, "usable_plot") for seed in spec["seeds"])
        has_credit = any(
            available(seed, field)
            for seed in spec["seeds"]
            for field in ("directors", "writers", "cast", "selected_crew")
        )
        seed_feature_groups["usable_plot"][int(has_plot)].append(user)
        seed_feature_groups["any_usable_credit"][int(has_credit)].append(user)

    seed_feature_metrics: dict[str, Any] = {}
    for feature, groups in seed_feature_groups.items():
        available_label = "at_least_one_seed_with_usable_plot" if feature == "usable_plot" else "at_least_one_seed_with_usable_credit"
        missing_label = "no_positive_seed_with_usable_plot" if feature == "usable_plot" else "no_positive_seed_with_usable_credit"
        seed_feature_metrics[feature] = {}
        for has_value, users in groups.items():
            label = available_label if has_value else missing_label
            seed_feature_metrics[feature][label] = {"users": len(users), "models": {}}
            for model in ("phase9_content_hybrid", "phase10_collaborative", "phase12_unified_hybrid", "phase12_weighted_content_75"):
                sub = _summarize_ranking_model(
                    rankings[model], relevant, {user: specs[user] for user in users}, (10,)
                ) if users else {"users_with_any_recommendation": 0, "recommendation_coverage": 0.0, "metrics_by_k": {"10": {"precision_at_k": None, "recall_at_k": None, "ndcg_at_k": None}}}
                metric = sub["metrics_by_k"]["10"]
                seed_feature_metrics[feature][label]["models"][model] = {
                    "users_with_recommendations": sub["users_with_any_recommendation"],
                    "coverage": sub["recommendation_coverage"],
                    "precision_at_10": metric["precision_at_k"],
                    "recall_at_10": metric["recall_at_k"],
                    "ndcg_at_10": metric["ndcg_at_k"],
                }

    relevant_interactions = [movie_id for movies in relevant.values() for movie_id in movies]
    target_coverage = {}
    for field in fields:
        count = sum(available(movie_id, field) for movie_id in relevant_interactions)
        target_coverage[field] = {
            "available_relevant_interactions": int(count),
            "relevant_interactions": len(relevant_interactions),
            "coverage": count / max(1, len(relevant_interactions)),
        }
    # Conditional Recall/NDCG show whether held-out positive targets with a
    # source value are retrieved at different rates than targets without one.
    # These are descriptive slices, not causal estimates or semantic labels.
    target_quality: dict[str, Any] = {}
    quality_models = ("phase9_content_hybrid", "phase10_collaborative", "phase12_unified_hybrid", "phase12_weighted_content_75")
    for field in fields:
        target_quality[field] = {}
        for has_value in (True, False):
            label = "source_value_available" if has_value else "source_value_missing_or_unretrieved"
            subset_by_user: dict[int, set[int]] = {}
            interaction_count = 0
            for user, movie_ids in relevant.items():
                subset = {movie_id for movie_id in movie_ids if available(movie_id, field) == has_value}
                if subset:
                    subset_by_user[int(user)] = subset
                    interaction_count += len(subset)
            target_quality[field][label] = {
                "relevant_interactions": interaction_count,
                "users_with_relevant_interactions": len(subset_by_user),
                "models": {},
            }
            for model in quality_models:
                recalls: list[float] = []
                ndcgs: list[float] = []
                for user, subset in subset_by_user.items():
                    metrics = ranking_metrics(rankings[model].get(user, []), subset, 10)
                    recalls.append(float(metrics["recall"]))
                    ndcgs.append(float(metrics["ndcg"]))
                target_quality[field][label]["models"][model] = {
                    "mean_user_recall_at_10": float(np.mean(recalls)) if recalls else None,
                    "mean_user_ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else None,
                }
    return {
        "catalog_field_coverage": catalog_coverage,
        "positive_seed_feature_groups": seed_feature_metrics,
        "relevant_validation_target_field_coverage": target_coverage,
        "conditional_target_quality": target_quality,
        "interpretation": "Missing plots/credits indicate unavailable source data; they are not treated as evidence the movie has no plot or credits. User metrics are descriptive, not causal estimates of the effect of a field.",
    }


def _rating_validation(
    collaborative: ItemItemCollaborativeRecommender,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movie_count: int,
) -> tuple[dict[str, Any], dict[int, pd.DataFrame]]:
    """Evaluate Phase 10 scores and training-only means on validation events."""
    global_mean = float(train.rating.mean())
    user_means = train.groupby("user_id").rating.mean().to_dict()
    predictions: dict[int, dict[int, float]] = {}
    full_rankings: dict[int, pd.DataFrame] = {}
    for user in sorted(train.user_id.unique()):
        try:
            frame = collaborative.recommend(int(user), top_k=movie_count)
        except (KeyError, ValueError):
            frame = pd.DataFrame(columns=["movie_id", "predicted_rating", "support_count", "evidence_weight"])
        full_rankings[int(user)] = frame
        predictions[int(user)] = dict(zip(frame.movie_id.astype(int), frame.predicted_rating.astype(float)))
    actual: list[float] = []
    predicted: list[float] = []
    paired_actual: list[float] = []
    paired_cf: list[float] = []
    paired_user: list[float] = []
    by_user_cf: dict[int, list[float]] = {}
    by_user_base: dict[int, list[float]] = {}
    for row in validation.itertuples(index=False):
        user, movie, rating = int(row.user_id), int(row.movie_id), float(row.rating)
        score = predictions.get(user, {}).get(movie)
        if score is None:
            continue
        actual.append(rating)
        predicted.append(score)
        paired_actual.append(rating)
        paired_cf.append(score)
        paired_user.append(float(user_means[user]))
        by_user_cf.setdefault(user, []).append(score - rating)
        by_user_base.setdefault(user, []).append(float(user_means[user]) - rating)
    all_actual = validation.rating.astype(float).to_numpy()
    global_pred = np.full(len(validation), global_mean)
    user_pred = validation.user_id.map(user_means).astype(float).to_numpy()
    supported = len(actual)
    if supported:
        cf_mae = float(mean_absolute_error(actual, predicted))
        cf_rmse = float(math.sqrt(mean_squared_error(actual, predicted)))
        base_mae = float(mean_absolute_error(paired_actual, paired_user))
        base_rmse = float(math.sqrt(mean_squared_error(paired_actual, paired_user)))
        mae_ci = user_bootstrap_error_difference(
            {u: np.asarray(v) for u, v in by_user_cf.items()},
            {u: np.asarray(v) for u, v in by_user_base.items()}, metric="mae",
            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        )
        rmse_ci = user_bootstrap_error_difference(
            {u: np.asarray(v) for u, v in by_user_cf.items()},
            {u: np.asarray(v) for u, v in by_user_base.items()}, metric="rmse",
            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        )
    else:
        cf_mae = cf_rmse = base_mae = base_rmse = None
        mae_ci = rmse_ci = None
    result = {
        "validation_interactions": int(len(validation)),
        "training_only_baselines": {
            "global_mean": {"mae": float(mean_absolute_error(all_actual, global_pred)), "rmse": float(math.sqrt(mean_squared_error(all_actual, global_pred)))},
            "user_mean": {"mae": float(mean_absolute_error(all_actual, user_pred)), "rmse": float(math.sqrt(mean_squared_error(all_actual, user_pred)))},
        },
        "phase10_collaborative": {
            "supported_predictions": supported,
            "unsupported_predictions": int(len(validation) - supported),
            "coverage": supported / max(1, len(validation)),
            "mae": cf_mae,
            "rmse": cf_rmse,
        },
        "paired_on_supported_interactions": {
            "count": supported,
            "phase10_mae": cf_mae,
            "phase10_rmse": cf_rmse,
            "user_mean_mae": base_mae,
            "user_mean_rmse": base_rmse,
            "mae_difference_and_95_ci": mae_ci,
            "rmse_difference_and_95_ci": rmse_ci,
            "difference_sign": "negative Phase10-minus-user-mean favors Phase 10",
        },
        "prediction_api_note": "The existing Phase 10 interface returns ranked supported candidates; no arbitrary-pair prediction method was added.",
    }
    return result, full_rankings


def _audit_movie_recommendations(
    clean: pd.DataFrame,
    phase9: HybridContentRecommender,
    masked_phase9: HybridContentRecommender,
    chunked_phase9: HybridContentRecommender,
    tfidf: TfidfMovieRecommender,
    transformer: TransformerMovieRecommender,
) -> list[dict[str, Any]]:
    title_by_id = dict(zip(clean.movie_id.astype(int), clean.title.astype(str)))
    plot_by_id = dict(zip(clean.movie_id.astype(int), clean["plot"]))
    row_by_id = {int(row.movie_id): row for row in clean.itertuples(index=False)}
    reports: list[dict[str, Any]] = []
    for movie_id in FIXED_AUDIT_IDS:
        if movie_id not in title_by_id:
            continue
        query = row_by_id[movie_id]
        query_plot = getattr(query, "plot")
        record: dict[str, Any] = {
            "query_movie_id": movie_id,
            "query_title": title_by_id[movie_id],
            "query_plot_available": bool(pd.notna(query_plot) and str(query_plot).strip()),
            "query_plot_excerpt": str(query_plot)[:350] if pd.notna(query_plot) else None,
            "query_genres": str(getattr(query, "genres", "")),
            "phase6_tfidf": [],
            "phase7_transformer": [],
            "phase9_baseline": [],
            "phase9_name_masked_diagnostic": [],
            "phase9_chunked_diagnostic": [],
        }
        for name, model in (("phase6_tfidf", tfidf), ("phase7_transformer", transformer)):
            try:
                rows = model.recommend(movie_id, top_k=5)
                record[name] = [
                    {"rank": i, "movie_id": int(r.movie_id), "title": str(r.title),
                     "score": float(r.similarity), "plot_excerpt": str(plot_by_id.get(int(r.movie_id), ""))[:250],
                     "evidence_type": "plot lexical similarity" if name == "phase6_tfidf" else "plot embedding cosine"}
                    for i, r in enumerate(rows.itertuples(index=False), start=1)
                ]
            except ValueError as error:
                record[name] = {"status": "unavailable", "reason": str(error)}
        for name, model in (("phase9_baseline", phase9),
                            ("phase9_name_masked_diagnostic", masked_phase9),
                            ("phase9_chunked_diagnostic", chunked_phase9)):
            rows = model.recommend(movie_id, top_k=5)
            record[name] = [
                {
                    "rank": i,
                    "movie_id": int(r.movie_id),
                    "title": str(r.title),
                    "score": float(r.hybrid_score),
                    "plot_excerpt": str(plot_by_id.get(int(r.movie_id), ""))[:250],
                    "genres": str(row_by_id[int(r.movie_id)].genres),
                    "evidence_type": "separate actual Phase 9 fields: genres/tags, TF-IDF, Transformer, director, writer, cast, crew",
                    "explanation": str(getattr(r, "explanation", "")),
                    "tfidf_similarity": float(r.tfidf_similarity) if pd.notna(r.tfidf_similarity) else None,
                    "transformer_similarity": float(r.transformer_similarity) if pd.notna(r.transformer_similarity) else None,
                    "director_similarity": float(r.director_similarity) if pd.notna(r.director_similarity) else None,
                    "writer_similarity": float(r.writer_similarity) if pd.notna(r.writer_similarity) else None,
                    "cast_similarity": float(r.cast_similarity) if pd.notna(r.cast_similarity) else None,
                    "crew_similarity": float(r.crew_similarity) if pd.notna(r.crew_similarity) else None,
                }
                for i, r in enumerate(rows.itertuples(index=False), start=1)
            ]
        reports.append(record)
    return reports


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# CineMatch Phase 14 — Evidence-Based Optimization",
        "",
        "## Validation boundary",
        "",
        f"Nested temporal holdout from Phase 13 training only: {report['validation_split']['phase14_train_interactions']:,} train and {report['validation_split']['phase14_validation_interactions']:,} validation interactions across {report['validation_split']['phase14_validation_users']} users. Phase 13 test overlap: {report['validation_split']['phase13_test_overlap']}. Boundary ties: {report['validation_split']['users_with_timestamp_tie_at_split_boundary']}. The Phase 13 test report and manifest were not used for tuning.",
        "",
        "## Ranking validation",
        "",
        "Relevance is a validation rating ≥4.0. All models use the same validation users and candidate universe (catalog excluding each user's Phase 14 training-rated movies). Recall/NDCG omit users with no relevant validation item; precision includes every eligible user.",
        "",
        "| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | Users with recs / eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["ranking_models"].items():
        m5, m10 = values["metrics_by_k"]["5"], values["metrics_by_k"]["10"]
        lines.append(
            f"| {name} | {m5['precision_at_k']:.4f} | {m5['recall_at_k']:.4f} | {m5['ndcg_at_k']:.4f} | {m10['precision_at_k']:.4f} | {m10['recall_at_k']:.4f} | {m10['ndcg_at_k']:.4f} | {values['users_with_any_recommendation']}/{values['eligible_users']} ({values['recommendation_coverage']:.1%}) |"
        )
    lines.extend(["", "## Phase 10 rating prediction", ""])
    rating = report["rating_prediction"]
    lines.append("| Model | N | Coverage | MAE | RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, values in rating["training_only_baselines"].items():
        lines.append(f"| {name} | {rating['validation_interactions']} | 100.0% | {values['mae']:.4f} | {values['rmse']:.4f} |")
    cf = rating["phase10_collaborative"]
    lines.append(f"| phase10_collaborative | {cf['supported_predictions']} | {cf['coverage']:.1%} | {cf['mae']:.4f} | {cf['rmse']:.4f} |")
    lines.extend(["", "## Paired differences at K=10", "", "Differences are first model minus baseline; intervals bootstrap users.", ""])
    for key, item in report["paired_differences_at_10"].items():
        lines.append(f"- `{key}`: {item['mean_difference']:.4f}, 95% interval [{item['ci_95_user_bootstrap'][0]:.4f}, {item['ci_95_user_bootstrap'][1]:.4f}] (n={item['users']}).")
    lines.extend(["", "## Sparse history and source diagnostics", ""])
    sparse = report["ranking_models"]["phase12_weighted_content_75"]["history_groups"]["sparse_5_to_20"]
    m = sparse["metrics_at_k"]["10"]
    lines.append(f"Sparse users: {sparse['users']}; weighted-fusion coverage {sparse['recommendation_coverage']:.1%}; P@10 {m['precision_at_k']:.4f}; R@10 {m['recall_at_k']:.4f}; NDCG@10 {m['ndcg_at_k']:.4f}.")
    diag = report["phase12_source_diagnostics"]
    source_counts = diag["source_availability_and_empty_counts"]
    relevant_hits = diag["relevant_heldout_movies_in_top10_lists"]
    lines.append(f"Both sources returned a list for {source_counts['both_available']}/{diag['users']} users; both were empty for {source_counts['both_empty']}. Their top-10 lists shared {diag['top10_overlap']['mean_shared_candidates']:.2f} movies on average (mean Jaccard {diag['top10_overlap']['mean_jaccard']:.4f}; overlap for {diag['top10_overlap']['users_with_any_overlap']} users). Relevant validation hits: content {relevant_hits['content']}, collaborative {relevant_hits['collaborative']}, equal fusion {relevant_hits['phase12_equal_fusion']}; equal fusion omitted {relevant_hits['relevant_content_candidates_displaced_from_phase12_top10']} relevant items that were in content's top 10.")
    tie = diag["equal_fusion_rank_and_tie_diagnostics"]
    lines.append(f"Equal-fusion scores tied within top 10 for {tie['users_with_at_least_one_equal_score_tie_within_top10']} users; {tie['users_with_equal_score_at_top10_boundary']} had a tie at rank 10/11. The existing ascending movie_id tie-break is preserved. Phase 12 top-10 source rows: {diag['phase12_top10_source_rows']}.")
    pop = report["popularity_diagnostics"]
    lines.append(f"Popularity ordering used training counts/means and covered all {pop['catalog_ids_in_popularity_order']} catalog IDs; {pop['catalog_ids_without_training_ratings']} had no training ratings and were placed last. {pop['top100_movies_with_training_mean_at_least_4']}/{pop['top100_movies_with_training_mean_available']} top-100 titles had training mean ≥4, compared with {pop['all_movies_with_training_mean_at_least_4']}/{pop['catalog_movies_with_any_training_rating']} rated catalog titles overall. Exposure effects remain unresolved.")
    feature = report["missing_feature_diagnostics"]
    plot = feature["catalog_field_coverage"]["usable_plot"]
    lines.append(f"Usable plots were present for {plot['available_movies']}/{plot['available_movies'] + plot['missing_or_unretrieved']} catalog titles; the JSON report also breaks plot/genre/tag/credit coverage down for positive training seeds, relevant validation items, and users with/without a plot-bearing seed. Missing credit values mean unavailable/unretrieved, not confirmed absence.")
    lines.extend(["", "## Diagnostic variant comparisons", ""])
    for candidate in ("phase9_name_masked_diagnostic", "phase9_chunked_diagnostic"):
        m10 = report["ranking_models"][candidate]["metrics_by_k"]["10"]
        lines.append(f"- **{candidate}**: P@10 {m10['precision_at_k']:.4f}, R@10 {m10['recall_at_k']:.4f}, NDCG@10 {m10['ndcg_at_k']:.4f}, coverage {report['ranking_models'][candidate]['recommendation_coverage']:.1%}.")
        key = f"{candidate}_minus_phase9_content_hybrid_ndcg@10"
        if key in report["paired_differences_at_10"]:
            item = report["paired_differences_at_10"][key]
            lines.append(f"  Paired NDCG@10 difference vs Phase 9: {item['mean_difference']:.4f}, 95% interval [{item['ci_95_user_bootstrap'][0]:.4f}, {item['ci_95_user_bootstrap'][1]:.4f}].")
    chunk = report["content_variants"]["chunked_transformer"]
    cache = chunk["cache_validation"]
    timing = chunk["ranking_cache_runtime_seconds"]
    lines.append(f"Chunked Transformer validation reused the pinned cache ({cache['chunks_encoded']:,} windows for {cache['long_plot_count']:,} long plots; 224 wordpieces with 48 overlap). Original/chunked artifact sizes: {cache['original_cache_bytes'] / 1_000_000:.2f}/{cache['chunked_cache_bytes'] / 1_000_000:.2f} MB. Phase 9 seed-ranking time: {timing['phase9_baseline']:.2f}s baseline vs {timing['phase9_chunked_diagnostic']:.2f}s chunked; total Phase 14 run: {report['runtime_seconds']:.2f}s. Embedding generation time was not measured in this run because the cache already existed.")
    lines.extend(["", "## Qualitative audit", "", "The JSON report contains full deterministic query rows and evidence. These are diagnostics, not human semantic relevance judgments.", ""])
    for case in report["qualitative_audit"]:
        lines.append(f"- **{case['query_title']}** (movie_id {case['query_movie_id']}, plot available: {case['query_plot_available']}): Phase 9 top result: {case['phase9_baseline'][0]['title'] if case['phase9_baseline'] else 'none'}; name-masked diagnostic: {case['phase9_name_masked_diagnostic'][0]['title'] if case['phase9_name_masked_diagnostic'] else 'none'}; chunked diagnostic: {case['phase9_chunked_diagnostic'][0]['title'] if case['phase9_chunked_diagnostic'] else 'none'}.")
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- No semantic ground-truth labels or human judgments are available; rating relevance is not story similarity.",
        "- MovieLens is positive-unlabeled: an unobserved candidate is not necessarily disliked.",
        "- Name masking is a diagnostic using previously identified names, not a general entity-resolution solution and not a production change.",
        "- Chunked vectors use a verified local cache; the stored artifact size and validation cost are recorded in JSON. One validation split cannot establish general improvement.",
        "- Popularity counts, means, and ordering are fitted only on Phase 14 training ratings. Exposure bias cannot be distinguished from broad appeal in this dataset.",
        "- No Phase 13 test outcomes were used to choose the experimental fusion or content diagnostic variants. Production defaults remain unchanged.",
        "",
        "## Reproducibility",
        "",
        "Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_optimization.py`",
        "",
        "The Phase 13 test report and manifest are preserved. Phase 14 writes only its own report and validation manifest.",
        "",
    ])
    MARKDOWN_REPORT.write_text("\n".join(lines))


def run_experiments() -> dict[str, Any]:
    start = time.monotonic()
    print("Creating Phase 14 validation split from Phase 13 training rows...", flush=True)
    train, validation, validation_manifest, split_summary = _read_validation_data()

    clean = pd.read_csv(CLEAN_MOVIES, dtype={"title": "string", "genres": "string", "plot": "string", "plot_tags": "string"})
    enriched = load_enriched_movies(ENRICHED)
    if set(clean.movie_id.astype(int)) != set(enriched.movie_id.astype(int)):
        raise ValueError("Clean and enriched movie catalogs have different movie IDs")
    text_movies = load_text_movies(CLEAN_MOVIES)
    original_transformer, chunked_transformer, transformer_storage = _load_validated_transformer_artifacts(text_movies)
    baseline_tfidf = TfidfMovieRecommender(text_movies)
    baseline_phase9 = HybridContentRecommender(
        enriched, tfidf_model=baseline_tfidf, transformer_model=original_transformer
    )

    masked_movies = text_movies.copy()
    masked_movies["plot"] = masked_movies["plot"].map(mask_diagnostic_names)
    masked_tfidf = TfidfMovieRecommender(masked_movies)
    masked_phase9 = HybridContentRecommender(
        enriched, tfidf_model=masked_tfidf, transformer_model=original_transformer
    )
    chunked_phase9 = HybridContentRecommender(
        enriched, tfidf_model=baseline_tfidf, transformer_model=chunked_transformer
    )

    specs = build_user_specs(train, top_k=max(K_VALUES))
    history_counts = Counter(spec["history_group"] for spec in specs.values())
    print(f"Preparing Phase 9 baseline rankings for {len(specs)} validation users...", flush=True)
    cache_timings: dict[str, float] = {}
    cache_start = time.monotonic()
    phase9_cache = _make_phase9_ranking_cache(baseline_phase9, specs, label="Phase 9")
    cache_timings["phase9_baseline"] = round(time.monotonic() - cache_start, 2)
    cache_start = time.monotonic()
    masked_cache = _make_phase9_ranking_cache(masked_phase9, specs, label="Name-mask diagnostic")
    cache_timings["phase9_name_masked_diagnostic"] = round(time.monotonic() - cache_start, 2)
    cache_start = time.monotonic()
    chunked_cache = _make_phase9_ranking_cache(chunked_phase9, specs, label="Chunked diagnostic")
    cache_timings["phase9_chunked_diagnostic"] = round(time.monotonic() - cache_start, 2)
    cold = ColdStartRecommender(enriched, phase9_cache, None)  # Phase 11 content route does not consume CF here.
    masked_cold = ColdStartRecommender(enriched, masked_cache, None)
    chunked_cold = ColdStartRecommender(enriched, chunked_cache, None)

    collaborative = ItemItemCollaborativeRecommender(train, clean[["movie_id", "title"]])
    metadata_catalog = prepare_catalog(enriched)
    tfidf = baseline_tfidf
    popularity_stats = train.groupby("movie_id").agg(
        training_rating_count=("rating", "size"), training_mean_rating=("rating", "mean")
    ).reset_index()
    # Give every catalog title a place in the same candidate universe. A movie
    # without Phase 14 training ratings has zero popularity evidence and sorts
    # after rated movies; it is not silently removed from evaluation.
    popularity = clean[["movie_id", "title"]].merge(
        popularity_stats, on="movie_id", how="left", validate="one_to_one"
    )
    popularity["training_rating_count"] = popularity["training_rating_count"].fillna(0).astype("int64")
    popularity["training_mean_rating"] = popularity["training_mean_rating"].fillna(0.0)
    popularity = popularity.sort_values(
        ["training_rating_count", "training_mean_rating", "movie_id"],
        ascending=[False, False, True], kind="mergesort",
    )
    popular_ids = popularity.movie_id.astype(int).tolist()
    movie_count = len(clean)
    rating_report, cf_full_results = _rating_validation(collaborative, train, validation, movie_count)
    unified = HybridRecommender(enriched, train, cold, collaborative)

    model_names = (
        "global_popularity", "phase9_content_hybrid", "phase12_content_only_route",
        "phase10_collaborative", "phase12_unified_hybrid", "phase12_weighted_content_75",
        "phase9_name_masked_diagnostic", "phase9_chunked_diagnostic",
    )
    rankings: dict[str, dict[int, list[int]]] = {name: {} for name in model_names}
    sources: dict[int, dict[str, Any]] = {}
    phase12_outputs: dict[int, Any] = {}
    weighted_details: dict[int, dict[int, dict[str, Any]]] = {}
    phase12_statuses: Counter[str] = Counter()
    source_rows: dict[int, dict[str, Any]] = {}
    cf_support_by_movie: dict[int, tuple[int, float]] = {}

    print("Ranking validation users and building controlled fusion comparison...", flush=True)
    for user, spec in specs.items():
        seen = spec["seen"]
        source_k = max(5 * max(K_VALUES), max(K_VALUES)) + len(seen)
        p9_ids, p9_status = content_user_rankings(
            "phase9_content_hybrid", spec, metadata_catalog=metadata_catalog,
            tfidf=tfidf, transformer=original_transformer, cold_start=cold,
            seed_rank_cache={}, top_k=source_k,
        )
        # Phase 12's source list is Phase 11's favorite route, limited to the
        # same source pool after the same seen-item filter.
        p9_ids = p9_ids[:source_k]
        try:
            cf_frame = cf_full_results[user].head(source_k).copy()
        except KeyError:
            cf_frame = pd.DataFrame(columns=["movie_id", "support_count", "evidence_weight", "predicted_rating"])
        cf_ids = cf_frame.movie_id.astype(int).drop_duplicates().tolist()
        collab_history = train.loc[train.user_id.eq(user), "rating"]
        enough_history = len(collab_history) >= 2 and not np.allclose(collab_history.to_numpy(), collab_history.mean())
        if not enough_history:
            cf_ids = []
        for row in cf_frame.itertuples(index=False):
            cf_support_by_movie[int(row.movie_id)] = (int(row.support_count), float(row.evidence_weight))

        equal_ids, _ = _weighted_rank_fusion(
            cf_ids, p9_ids, weights={"collaborative": 0.5, "content": 0.5}, top_k=max(K_VALUES)
        )
        weighted_ids, detail = _weighted_rank_fusion(
            cf_ids, p9_ids, weights=PROPOSED_WEIGHTS, top_k=max(K_VALUES)
        )
        response = unified.recommend_for_user(user, top_k=max(K_VALUES))
        phase12_outputs[user] = response
        phase12_statuses[response.status] += 1
        api_ids = _recommendation_ids(response.recommendations)
        if api_ids != equal_ids:
            raise AssertionError(f"Phase 14 equal-fusion mirror differs from Phase 12 API for user {user}")

        rankings["global_popularity"][user] = [mid for mid in popular_ids if mid not in seen][:max(K_VALUES)]
        rankings["phase9_content_hybrid"][user] = p9_ids[:max(K_VALUES)]
        rankings["phase12_content_only_route"][user] = p9_ids[:max(K_VALUES)]
        rankings["phase10_collaborative"][user] = cf_ids[:max(K_VALUES)]
        rankings["phase12_unified_hybrid"][user] = api_ids
        rankings["phase12_weighted_content_75"][user] = weighted_ids
        rankings["phase9_name_masked_diagnostic"][user], _ = content_user_rankings(
            "phase9_content_hybrid", spec, metadata_catalog=metadata_catalog,
            tfidf=masked_tfidf, transformer=original_transformer, cold_start=masked_cold,
            seed_rank_cache={}, top_k=max(K_VALUES),
        )
        rankings["phase9_chunked_diagnostic"][user], _ = content_user_rankings(
            "phase9_content_hybrid", spec, metadata_catalog=metadata_catalog,
            tfidf=tfidf, transformer=chunked_transformer, cold_start=chunked_cold,
            seed_rank_cache={}, top_k=max(K_VALUES),
        )
        sources[user] = {"collaborative": cf_ids, "content": p9_ids, "content_status": p9_status,
                         "collaborative_available": bool(cf_ids), "content_available": bool(p9_ids),
                         "positive_seeds": list(spec["seeds"]), "history_group": spec["history_group"],
                         "enough_history_variation": bool(enough_history)}
        source_rows[user] = {
            "collaborative": cf_ids,
            "content": p9_ids,
            "support_count": [cf_support_by_movie[mid][0] for mid in cf_ids[:10] if mid in cf_support_by_movie],
            "evidence_weight": [cf_support_by_movie[mid][1] for mid in cf_ids[:10] if mid in cf_support_by_movie],
        }
        weighted_details[user] = detail
        if user % 100 == 0:
            print(f"Phase 14 validation users: {len(rankings['phase12_unified_hybrid'])}/{len(specs)}", flush=True)

    relevant_by_user = {
        int(user): set(group.loc[group.rating.ge(RELEVANCE_THRESHOLD), "movie_id"].astype(int))
        for user, group in validation.groupby("user_id", sort=True)
    }
    ranking_reports = {
        name: _summarize_ranking_model(rankings[name], relevant_by_user, specs, K_VALUES)
        for name in model_names
    }
    paired = _paired_differences(
        rankings, relevant_by_user, baselines=model_names, k=max(K_VALUES)
    )
    source_diagnostics = _source_diagnostics(
        source_rows, rankings["phase12_unified_hybrid"], relevant_by_user, specs, phase12_outputs
    )
    feature_diagnostics = _feature_coverage_diagnostics(
        clean, enriched, specs, relevant_by_user, rankings
    )
    # Inspect popularity's training-only profile and test its list only after
    # the list is fixed. No validation outcome enters the popularity ordering.
    top100 = set(popularity.movie_id.head(100).astype(int))
    catalog_training_means = train.groupby("movie_id").rating.mean()
    popular_stats = {
        "ranking_inputs": ["Phase 14 training rating count", "Phase 14 training mean rating", "movie_id tie-break"],
        "top100_movies_with_training_mean_at_least_4": int(catalog_training_means.reindex(list(top100)).ge(4).sum()),
        "top100_movies_with_training_mean_available": int(catalog_training_means.reindex(list(top100)).notna().sum()),
        "all_movies_with_training_mean_at_least_4": int(catalog_training_means.ge(4).sum()),
        "catalog_movies_with_any_training_rating": int(catalog_training_means.size),
        "top10_titles_and_training_stats": [
            {"movie_id": int(row.movie_id), "title": str(row.title), "training_count": int(row.training_rating_count),
             "training_mean": float(row.training_mean_rating)}
            for row in popularity.head(10).itertuples(index=False)
        ],
        "catalog_ids_in_popularity_order": int(len(popularity)),
        "catalog_ids_without_training_ratings": int(popularity.training_rating_count.eq(0).sum()),
        "leakage_check": "Counts, means, and order are calculated from Phase 14 training interactions only.",
        "interpretation_limit": "This observational dataset cannot separate genuine broad appeal from exposure and rating-selection effects.",
    }

    audit = _audit_movie_recommendations(
        clean, baseline_phase9, masked_phase9, chunked_phase9, baseline_tfidf, original_transformer
    )
    query_seed_users: dict[str, Any] = {}
    for movie_id in FIXED_AUDIT_IDS:
        users = [user for user, spec in specs.items() if movie_id in spec["seeds"]]
        if not users:
            query_seed_users[str(movie_id)] = {"status": "no_validation_user_has_this_movie_as_a_positive_training_seed"}
            continue
        user = min(users)
        query_seed_users[str(movie_id)] = {
            "user_id": user,
            "training_seed_movie_ids": list(specs[user]["seeds"]),
            "phase12_status": phase12_outputs[user].status,
            "phase12_top5": phase12_outputs[user].recommendations.head(5).to_dict(orient="records"),
            "weighted_content_75_top5": [
                {"movie_id": mid, "title": str(enriched.loc[enriched.movie_id.eq(mid), "title"].iloc[0]),
                 **weighted_details[user].get(mid, {})}
                for mid in rankings["phase12_weighted_content_75"][user][:5]
            ],
        }

    report = {
        "phase": "Phase 14: Evidence-Based Optimization",
        "created_by": "scripts/phase14_optimization.py",
        "runtime_seconds": None,
        "validation_split": split_summary,
        "ranking_task": {
            "relevance_threshold": RELEVANCE_THRESHOLD,
        "candidate_universe": "All catalog movie IDs excluding each user's Phase 14 training-rated movies. Popularity includes unrated catalog movies at zero training count, below rated movies.",
            "phase13_test_used_for_selection": False,
            "unrated_candidates": "Positive-unlabeled: unrated does not mean disliked; conventional precision counts them as non-hits.",
            "users_without_relevant_validation_movie": int(sum(not bool(relevant_by_user.get(u)) for u in specs)),
            "eligible_users": int(len(specs)),
        },
        "ranking_models": ranking_reports,
        "paired_differences_at_10": paired,
        "phase12_source_diagnostics": source_diagnostics,
        "missing_feature_diagnostics": feature_diagnostics,
        "phase12_status_counts": dict(phase12_statuses),
        "source_lists_by_user": sources,
        "popularity_diagnostics": popular_stats,
        "rating_prediction": rating_report,
        "content_variants": {
            "name_masking": {
                "role": "validation diagnostic only",
                "tokens": ["Max", "Robin", "Savannah"],
                "warning": "Targeted names were selected from previous failure cases; this is an overfit-prone ablation, not a general proper-name detector or production proposal.",
            },
            "chunked_transformer": {
            "role": "validation diagnostic only",
            "cache_validation": transformer_storage,
            "ranking_cache_runtime_seconds": cache_timings,
                "runtime_note": "No re-encoding occurred; both local artifacts were loaded and compared using the same validation split.",
                "embedding_runtime_training": "Not remeasured because the pinned chunked cache already existed; generation log from the Phase 9 artifact records 17,548 windows.",
            },
        },
        "qualitative_audit": audit,
        "personalized_fixed_seed_users": query_seed_users,
        "baseline_preservation": {
            "phase5_to_phase12_source_files_modified": False,
            "phase13_report_modified": False,
            "phase13_manifest_modified": False,
            "cleaned_or_enriched_data_modified": False,
            "original_or_chunked_embedding_cache_modified": False,
            "phase9_weights": baseline_phase9.weights,
            "phase9_baseline_weights_unchanged": baseline_phase9.weights == {
                "metadata": 0.34, "tfidf": 0.16, "transformer": 0.22,
                "director": 0.10, "writer": 0.07, "cast": 0.07, "crew": 0.04,
            },
            "phase10_algorithm_unchanged": True,
            "phase11_routing_unchanged": True,
            "phase12_default_equal_weights_unchanged": True,
        },
        "proposed_fusion": {
            "weights": PROPOSED_WEIGHTS,
            "rule": "Weighted reciprocal rank contributions; if an entire source is empty, renormalize over the available source; candidate tie-break is ascending movie_id.",
            "selected_as_default": False,
            "selection_rule": "Diagnostic validation comparison only; no Phase 13 test outcomes used and no production default changed.",
        },
    }
    report["runtime_seconds"] = round(time.monotonic() - start, 2)
    JSON_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    _write_markdown(report)
    return report


def refresh_saved_feature_diagnostics() -> dict[str, Any]:
    """Rebuild only missing-field slices from saved Phase 14 outputs.

    This avoids repeating the expensive embedding/ranking caches when only the
    descriptive plot/credit coverage section needs regeneration. It reads
    validation rating rows only; Phase 13 test values are never joined.
    """
    report = json.loads(JSON_REPORT.read_text())
    manifest = pd.read_csv(VALIDATION_MANIFEST)
    report["validation_split"]["saved_manifest_partition_labels"] = ["train", "validation"]
    validation_keys = manifest.loc[
        manifest.partition.eq("validation"), ["user_id", "movie_id", "timestamp"]
    ].rename(columns={"partition": "validation_partition"})
    ratings = pd.read_csv(RATINGS, usecols=["user_id", "movie_id", "rating", "timestamp"])
    validation = validation_keys.merge(
        ratings, on=["user_id", "movie_id", "timestamp"], how="inner", validate="one_to_one"
    )
    if len(validation) != int(report["validation_split"]["phase14_validation_interactions"]):
        raise ValueError("Validation ratings do not match the saved Phase 14 manifest")
    relevant = {
        int(user): set(group.loc[group.rating.ge(RELEVANCE_THRESHOLD), "movie_id"].astype(int))
        for user, group in validation.groupby("user_id", sort=True)
    }
    source_data = report["source_lists_by_user"]
    specs = {
        int(user): {"seeds": tuple(row["positive_seeds"]), "history_group": row["history_group"]}
        for user, row in source_data.items()
    }
    rankings: dict[str, dict[int, list[int]]] = {
        name: {} for name in (
            "phase9_content_hybrid", "phase10_collaborative",
            "phase12_unified_hybrid", "phase12_weighted_content_75",
        )
    }
    for user_string, row in source_data.items():
        user = int(user_string)
        content, collab = row["content"], row["collaborative"]
        rankings["phase9_content_hybrid"][user] = content[:10]
        rankings["phase10_collaborative"][user] = collab[:10]
        rankings["phase12_unified_hybrid"][user], _ = _weighted_rank_fusion(
            collab, content, weights={"collaborative": 0.5, "content": 0.5}, top_k=10
        )
        rankings["phase12_weighted_content_75"][user], _ = _weighted_rank_fusion(
            collab, content, weights=PROPOSED_WEIGHTS, top_k=10
        )
    clean = pd.read_csv(CLEAN_MOVIES, dtype={"plot": "string"})
    enriched = load_enriched_movies(ENRICHED)
    report["missing_feature_diagnostics"] = _feature_coverage_diagnostics(
        clean, enriched, specs, relevant, rankings
    )
    JSON_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    _write_markdown(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 14 validation or refresh saved missing-feature diagnostics.")
    parser.add_argument(
        "--refresh-feature-diagnostics", action="store_true",
        help="recompute plot/credit coverage slices from existing Phase 14 rankings and validation manifest without rerunning models",
    )
    args = parser.parse_args()
    if args.refresh_feature_diagnostics:
        report = refresh_saved_feature_diagnostics()
        print(json.dumps({
            "refreshed": "missing_feature_diagnostics",
            "validation_interactions": report["validation_split"]["phase14_validation_interactions"],
            "phase13_test_used": False,
            "json_report": str(JSON_REPORT),
            "markdown_report": str(MARKDOWN_REPORT),
        }, indent=2))
        return
    report = run_experiments()
    print(json.dumps({
        "phase": report["phase"],
        "validation_split": {k: report["validation_split"][k] for k in (
            "phase14_train_interactions", "phase14_validation_interactions", "phase14_validation_users",
            "users_with_timestamp_tie_at_split_boundary", "phase13_test_overlap",
        )},
        "ranking_models": list(report["ranking_models"]),
        "proposed_weighted_p10": report["ranking_models"]["phase12_weighted_content_75"]["metrics_by_k"]["10"]["precision_at_k"],
        "runtime_seconds": report["runtime_seconds"],
        "json_report": str(JSON_REPORT),
        "markdown_report": str(MARKDOWN_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
