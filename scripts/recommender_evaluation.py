"""Small, model-independent helpers for the Phase 13 offline evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


INTERACTION_KEY = ["user_id", "movie_id"]


def make_temporal_split(
    ratings: pd.DataFrame,
    *,
    holdout_fraction: float = 0.20,
    minimum_training_ratings: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Hold out each eligible user's final timestamp-ordered 20% of ratings.

    Movie ID breaks timestamp ties deterministically. If a tie straddles the
    split boundary, the tie-break is reproducible but is not a known chronology.
    The returned manifest contains interaction keys, timestamps, order, and
    partition only; it does not duplicate rating values.
    """
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings are missing columns: {sorted(missing)}")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if minimum_training_ratings < 1:
        raise ValueError("minimum_training_ratings must be positive")
    if ratings[sorted(required)].isna().any().any():
        raise ValueError("Ratings contain missing split fields")
    if ratings.duplicated(INTERACTION_KEY).any():
        raise ValueError("Ratings must have unique user_id/movie_id interactions")

    rows: list[pd.DataFrame] = []
    excluded_users: list[dict[str, object]] = []
    tied_boundary_users = 0
    for user_id, group in ratings.groupby("user_id", sort=True):
        ordered = group.sort_values(["timestamp", "movie_id"], kind="mergesort").copy()
        test_count = max(1, math.ceil(len(ordered) * holdout_fraction))
        train_count = len(ordered) - test_count
        if train_count < minimum_training_ratings:
            excluded_users.append({"user_id": int(user_id), "ratings": len(ordered), "reason": "below_minimum_training_history"})
            continue

        ordered["user_order"] = np.arange(1, len(ordered) + 1, dtype=np.int64)
        ordered["partition"] = "train"
        ordered.loc[ordered.index[train_count:], "partition"] = "test"
        if int(ordered.iloc[train_count - 1]["timestamp"]) == int(ordered.iloc[train_count]["timestamp"]):
            tied_boundary_users += 1
        rows.append(ordered[["user_id", "movie_id", "timestamp", "user_order", "partition"]])

    if not rows:
        raise ValueError("No users satisfy the minimum training-history rule")
    manifest = pd.concat(rows, ignore_index=True).sort_values(
        ["user_id", "user_order"], kind="mergesort"
    ).reset_index(drop=True)
    train_counts = manifest.loc[manifest.partition.eq("train")].groupby("user_id").size()
    test_counts = manifest.loc[manifest.partition.eq("test")].groupby("user_id").size()
    summary = {
        "method": "per-user final holdout after sorting by timestamp, then movie_id",
        "holdout_fraction": holdout_fraction,
        "holdout_count_rule": "max(1, ceil(user_rating_count * holdout_fraction))",
        "minimum_training_ratings": minimum_training_ratings,
        "eligible_users": int(manifest.user_id.nunique()),
        "excluded_users": excluded_users,
        "train_interactions": int(manifest.partition.eq("train").sum()),
        "test_interactions": int(manifest.partition.eq("test").sum()),
        "minimum_training_history": int(train_counts.min()),
        "median_training_history": float(train_counts.median()),
        "maximum_training_history": int(train_counts.max()),
        "users_with_timestamp_tie_at_split_boundary": tied_boundary_users,
        "tie_limitation": "Movie ID gives deterministic ordering for equal timestamps; it cannot recover the true event order.",
        "seed": None,
    }
    return manifest, summary


def split_ratings_by_manifest(
    ratings: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join partition labels by actual user/movie keys, never by row positions."""
    labels = manifest[INTERACTION_KEY + ["partition"]]
    joined = ratings.merge(labels, on=INTERACTION_KEY, how="inner", validate="one_to_one")
    if len(joined) != len(ratings):
        raise ValueError("Split manifest does not cover every supplied rating")
    train = joined.loc[joined.partition.eq("train")].drop(columns="partition").sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)
    test = joined.loc[joined.partition.eq("test")].drop(columns="partition").sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)
    if set(map(tuple, train[INTERACTION_KEY].to_numpy())) & set(map(tuple, test[INTERACTION_KEY].to_numpy())):
        raise ValueError("Training and test interaction keys overlap")
    return train, test


def ranking_metrics(
    recommended_movie_ids: Iterable[int], relevant_movie_ids: Iterable[int], k: int
) -> dict[str, float | None]:
    """Calculate binary Precision@K, Recall@K, and NDCG@K.

    Precision always divides by K, so a short recommendation list has empty
    positions counted as misses. Recall and NDCG are undefined when there are
    no relevant held-out movies and are returned as None.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    unique: list[int] = []
    seen: set[int] = set()
    for value in recommended_movie_ids:
        movie_id = int(value)
        if movie_id not in seen:
            unique.append(movie_id)
            seen.add(movie_id)
    relevant = {int(movie_id) for movie_id in relevant_movie_ids}
    top = unique[:k]
    hits = [int(movie_id in relevant) for movie_id in top]
    hit_count = sum(hits)
    precision = hit_count / k
    if not relevant:
        return {"precision": precision, "recall": None, "ndcg": None, "hits": hit_count}
    recall = hit_count / len(relevant)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(k, len(relevant))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {"precision": precision, "recall": recall, "ndcg": dcg / ideal_dcg, "hits": hit_count}


def aggregate_rating_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    """Return MAE, RMSE, and the exact number of paired predictions."""
    truth = np.asarray(list(actual), dtype=np.float64)
    estimate = np.asarray(list(predicted), dtype=np.float64)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays of equal length")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("actual and predicted values must be finite")
    errors = estimate - truth
    return {
        "count": int(len(errors)),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
    }


def user_bootstrap_error_difference(
    model_errors: dict[int, np.ndarray],
    baseline_errors: dict[int, np.ndarray],
    *,
    metric: str,
    seed: int = 13,
    resamples: int = 1000,
) -> dict[str, float | int]:
    """Paired cluster bootstrap of model-minus-baseline error over users."""
    users = sorted(set(model_errors) & set(baseline_errors))
    if not users or metric not in {"mae", "rmse"} or resamples < 1:
        raise ValueError("Need paired users, a supported metric, and positive resamples")

    def calculate(sample: list[int]) -> float:
        model = np.concatenate([model_errors[user] for user in sample])
        baseline = np.concatenate([baseline_errors[user] for user in sample])
        if metric == "mae":
            return float(np.mean(np.abs(model)) - np.mean(np.abs(baseline)))
        return float(np.sqrt(np.mean(np.square(model))) - np.sqrt(np.mean(np.square(baseline))))

    observed = calculate(users)
    rng = np.random.default_rng(seed)
    draws = [calculate([users[index] for index in rng.integers(0, len(users), len(users))]) for _ in range(resamples)]
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"users": len(users), "difference_model_minus_baseline": observed,
            "ci_95_user_bootstrap": [float(lower), float(upper)], "resamples": resamples, "seed": seed}


def mean_user_bootstrap_interval(
    values_by_user: dict[int, float], *, seed: int = 13, resamples: int = 1000
) -> dict[str, float | int]:
    """Confidence interval for a user-level mean, resampling users as clusters."""
    users = sorted(values_by_user)
    values = np.asarray([values_by_user[user] for user in users], dtype=np.float64)
    if not len(values):
        raise ValueError("At least one user-level value is required")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"users": len(users), "mean": float(values.mean()),
            "ci_95_user_bootstrap": [float(lower), float(upper)], "resamples": resamples, "seed": seed}
