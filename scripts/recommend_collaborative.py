"""Phase 10: item-item collaborative recommendations from MovieLens ratings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_paths import data_path

RATINGS_PATH = data_path("cleaned", "ratings.csv")
MOVIES_PATH = data_path("cleaned", "movies.csv")
RATING_MIN = 0.5
RATING_MAX = 5.0
RESULT_COLUMNS = [
    "movie_id", "title", "predicted_rating", "support_count", "evidence_weight",
    "supporting_movies",
]


def clip_predicted_rating(value: float) -> float:
    """Keep an estimated rating on the MovieLens half-star scale's bounds."""
    return float(np.clip(value, RATING_MIN, RATING_MAX))


def predict_rating(
    user_mean: float,
    weighted_deviation_sum: float,
    evidence_weight: float,
    prediction_shrinkage: float = 1.0,
) -> float:
    """Adjust a user's mean from neighbor evidence, then clip to the rating scale."""
    if evidence_weight < 0 or prediction_shrinkage < 0:
        raise ValueError("evidence_weight and prediction_shrinkage must be nonnegative")
    if evidence_weight + prediction_shrinkage == 0:
        adjustment = 0.0
    else:
        adjustment = weighted_deviation_sum / (evidence_weight + prediction_shrinkage)
    return clip_predicted_rating(user_mean + adjustment)


class ItemItemCollaborativeRecommender:
    """Predict user ratings from similarities between co-rated movies.

    Only user_id, movie_id, and rating from the supplied interactions are used
    to fit the model. Movie titles are optional display information.
    """

    def __init__(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame | None = None,
        *,
        min_common_users: int = 3,
        shrinkage: float = 10.0,
        prediction_shrinkage: float = 1.0,
    ) -> None:
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"Ratings are missing columns: {sorted(missing)}")
        if not isinstance(min_common_users, int) or isinstance(min_common_users, bool) or min_common_users < 1:
            raise ValueError("min_common_users must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be a finite, nonnegative number")
        if not np.isfinite(prediction_shrinkage) or prediction_shrinkage < 0:
            raise ValueError("prediction_shrinkage must be a finite, nonnegative number")
        if ratings[["user_id", "movie_id", "rating"]].isna().any().any():
            raise ValueError("Ratings contain missing user_id, movie_id, or rating values")

        data = ratings[["user_id", "movie_id", "rating"]].copy()
        for column in ("user_id", "movie_id"):
            numeric_ids = pd.to_numeric(data[column], errors="coerce")
            if numeric_ids.isna().any() or not np.isfinite(numeric_ids).all():
                raise ValueError(f"{column} values must be finite integers")
            if not np.equal(numeric_ids, np.floor(numeric_ids)).all():
                raise ValueError(f"{column} values must be integers")
            data[column] = numeric_ids.astype("int64")

        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        if data["rating"].isna().any() or not np.isfinite(data["rating"]).all():
            raise ValueError("Ratings must be finite numbers")
        if not data["rating"].between(RATING_MIN, RATING_MAX).all():
            raise ValueError(f"Ratings must be between {RATING_MIN} and {RATING_MAX}")
        if not np.isclose(data["rating"] * 2, np.round(data["rating"] * 2)).all():
            raise ValueError("Ratings must use MovieLens half-star increments")
        if data.duplicated(["user_id", "movie_id"]).any():
            raise ValueError("Ratings contain duplicate user_id/movie_id pairs")
        if data.empty:
            raise ValueError("At least one rating is required to fit collaborative filtering")

        self.ratings = data.reset_index(drop=True)
        self.min_common_users = min_common_users
        self.shrinkage = float(shrinkage)
        self.prediction_shrinkage = float(prediction_shrinkage)
        self.users = sorted(self.ratings["user_id"].unique().tolist())
        self.movie_ids = sorted(self.ratings["movie_id"].unique().tolist())
        self.user_to_row = {user_id: row for row, user_id in enumerate(self.users)}
        self.movie_to_column = {movie_id: column for column, movie_id in enumerate(self.movie_ids)}

        if movies is None:
            self.movie_catalog = pd.DataFrame({"movie_id": self.movie_ids, "title": pd.NA})
        else:
            if "movie_id" not in movies.columns:
                raise ValueError("Movie catalog is missing movie_id")
            if movies["movie_id"].isna().any():
                raise ValueError("Movie catalog contains missing movie_id values")
            catalog = movies.copy()
            ids = pd.to_numeric(catalog["movie_id"], errors="coerce")
            if ids.isna().any() or not np.equal(ids, np.floor(ids)).all():
                raise ValueError("Movie catalog movie_id values must be integers")
            catalog["movie_id"] = ids.astype("int64")
            if catalog["movie_id"].duplicated().any():
                raise ValueError("Movie catalog contains duplicate movie_id values")
            unknown_movies = set(self.movie_ids) - set(catalog["movie_id"])
            if unknown_movies:
                raise ValueError("Ratings reference movie IDs absent from the supplied catalog")
            if "title" not in catalog.columns:
                catalog["title"] = pd.NA
            self.movie_catalog = catalog[["movie_id", "title"]].copy()
        self.catalog_ids = set(self.movie_catalog["movie_id"].astype(int))

        self.user_means = self.ratings.groupby("user_id")["rating"].mean().to_dict()
        user_rows = self.ratings["user_id"].map(self.user_to_row).to_numpy()
        movie_columns = self.ratings["movie_id"].map(self.movie_to_column).to_numpy()
        values = self.ratings["rating"].to_numpy(dtype=np.float64)
        means = self.ratings["user_id"].map(self.user_means).to_numpy(dtype=np.float64)
        residual_values = values - means
        matrix_shape = (len(self.users), len(self.movie_ids))

        # Sparse matrices preserve missing ratings as absent entries, never zero ratings.
        residuals = sparse.csr_matrix((residual_values, (user_rows, movie_columns)), shape=matrix_shape)
        observed = sparse.csr_matrix((np.ones(len(data)), (user_rows, movie_columns)), shape=matrix_shape)
        # Item-item intermediates are large (~movies^2 pairs), so each one is released as
        # soon as it is consumed and element-wise steps run in place. This keeps the
        # build's peak memory within serverless limits without changing the math.
        common_counts = (observed.T @ observed).tocsr()
        sufficiently_observed = common_counts.copy()
        sufficiently_observed.data = (
            sufficiently_observed.data >= min_common_users
        ).astype(np.float64)
        sufficiently_observed.eliminate_zeros()
        # Apply n/(n+alpha) to every retained pair's actual co-rater count.
        shrink_factors = common_counts.multiply(sufficiently_observed).tocsr()
        del common_counts
        shrink_factors.data = shrink_factors.data / (shrink_factors.data + self.shrinkage)

        # The overlap mask is symmetric with 1.0 values, so masking each full product
        # before combining it yields exactly the same retained pairs and values.
        squared_norms_on_overlap = (residuals.power(2).T @ observed).tocsr()
        del observed
        squared_norms_on_overlap = squared_norms_on_overlap.multiply(sufficiently_observed).tocsr()
        # Each pair's cosine denominator uses only users who rated both movies.
        inverse_denominators = squared_norms_on_overlap.multiply(squared_norms_on_overlap.T).tocsr()
        del squared_norms_on_overlap, sufficiently_observed
        np.sqrt(inverse_denominators.data, out=inverse_denominators.data)
        inverse_denominators.eliminate_zeros()
        inverse_denominators.data = 1.0 / inverse_denominators.data

        # The sparse item-item matrix keeps memory bounded; absent pairs are not evidence.
        residual_dots = (residuals.T @ residuals).tocsr()
        del residuals
        raw_similarities = residual_dots.multiply(inverse_denominators).tocsr()
        del residual_dots, inverse_denominators
        np.clip(raw_similarities.data, -1.0, 1.0, out=raw_similarities.data)
        self.similarity_matrix = raw_similarities.multiply(shrink_factors).tocsr()
        del raw_similarities, shrink_factors
        self.similarity_matrix.setdiag(0)
        self.similarity_matrix.eliminate_zeros()
        self.user_history: dict[int, dict[int, float]] = {
            user_id: dict(zip(group["movie_id"].astype(int), group["rating"].astype(float)))
            for user_id, group in self.ratings.groupby("user_id", sort=False)
        }

    def similarity(self, first_movie_id: int, second_movie_id: int) -> float | None:
        """Return the shrunken similarity, or None when evidence is insufficient."""
        for movie_id in (first_movie_id, second_movie_id):
            if movie_id not in self.catalog_ids:
                raise KeyError(f"Unknown movie_id: {movie_id}")
            if movie_id not in self.movie_to_column:
                return None
        first_row = self.movie_to_column[first_movie_id]
        second_column = self.movie_to_column[second_movie_id]
        value = float(self.similarity_matrix[first_row, second_column])
        return value if value != 0 else None

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """Recommend unseen movies using the user's mean-centered rating history."""
        if not isinstance(user_id, (int, np.integer)) or isinstance(user_id, bool):
            raise ValueError("user_id must be an integer")
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1")
        user_id = int(user_id)
        if user_id not in self.user_history:
            raise KeyError(f"Unknown user_id: {user_id}")

        history = self.user_history[user_id]
        if len(history) < 2:
            raise ValueError(
                f"Known user_id {user_id} has only {len(history)} rating(s); "
                "at least 2 are needed for mean-centered recommendations"
            )
        user_mean = float(self.user_means[user_id])
        deviations = {movie_id: rating - user_mean for movie_id, rating in history.items()}
        if all(np.isclose(value, 0.0) for value in deviations.values()):
            raise ValueError(f"Known user_id {user_id} has no rating variation to personalize from")

        history_movie_ids = list(history)
        history_columns = np.asarray(
            [self.movie_to_column[movie_id] for movie_id in history_movie_ids], dtype=np.int64
        )
        history_deviations = np.asarray(
            [deviations[movie_id] for movie_id in history_movie_ids], dtype=np.float64
        )
        candidate_links = self.similarity_matrix[:, history_columns].tocsr()
        weighted_deviations = np.asarray(candidate_links @ history_deviations).ravel()
        absolute_links = candidate_links.copy()
        absolute_links.data = np.abs(absolute_links.data)
        evidence_weights = np.asarray(absolute_links.sum(axis=1)).ravel()
        support_counts = np.diff(candidate_links.indptr)
        eligible_rows = np.flatnonzero(evidence_weights > 0)
        seen_columns = set(history_columns.tolist())

        recommendations = []
        for row in eligible_rows:
            if row in seen_columns:
                continue
            movie_id = self.movie_ids[row]
            predicted_rating = predict_rating(
                user_mean,
                weighted_deviations[row],
                evidence_weights[row],
                self.prediction_shrinkage,
            )
            start, end = candidate_links.indptr[row:row + 2]
            local_links = candidate_links.data[start:end]
            local_history_positions = candidate_links.indices[start:end]
            supporting_positions = sorted(
                range(len(local_links)),
                key=lambda position: (-abs(local_links[position]), history_movie_ids[local_history_positions[position]]),
            )
            recommendations.append({
                "movie_id": movie_id,
                "predicted_rating": predicted_rating,
                "support_count": int(support_counts[row]),
                "evidence_weight": float(evidence_weights[row]),
                "supporting_movie_ids": [
                    history_movie_ids[local_history_positions[position]]
                    for position in supporting_positions
                ],
            })

        if not recommendations:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        results = pd.DataFrame(recommendations)
        titles = self.movie_catalog.set_index("movie_id")["title"]
        results["title"] = results["movie_id"].map(titles)
        results["supporting_movies"] = results["supporting_movie_ids"].map(
            lambda movie_ids: " | ".join(
                str(titles.get(movie_id, movie_id)) for movie_id in movie_ids[:3]
            )
        )
        return results[RESULT_COLUMNS].sort_values(
            ["predicted_rating", "evidence_weight", "support_count", "movie_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).head(int(top_k)).reset_index(drop=True)


def load_ratings(path: Path = RATINGS_PATH) -> pd.DataFrame:
    """Load only interaction columns; timestamps are not a model feature here."""
    return pd.read_csv(path, usecols=["user_id", "movie_id", "rating"])


def load_movies(path: Path = MOVIES_PATH) -> pd.DataFrame:
    """Load IDs and titles for display and catalog membership validation."""
    return pd.read_csv(path, usecols=["movie_id", "title"], dtype={"title": "string"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend unseen movies from MovieLens user-rating patterns."
    )
    parser.add_argument("user_id", type=int, help="MovieLens user_id")
    parser.add_argument("--top-k", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        recommender = ItemItemCollaborativeRecommender(load_ratings(), load_movies())
        results = recommender.recommend(args.user_id, args.top_k)
    except (ValueError, KeyError) as error:
        parser.error(str(error))

    history = recommender.user_history[args.user_id]
    print(f"Collaborative recommendations for MovieLens user_id={args.user_id}")
    print("Demonstration using all supplied ratings; this is not held-out evaluation.")
    print(f"Known ratings: {len(history)}; user mean: {recommender.user_means[args.user_id]:.2f}")
    print("Predicted ratings are model estimates, not actual user ratings.")
    if results.empty:
        print("No unseen movies had sufficient co-rating evidence for this user.")
    else:
        print(results.to_string(index=False, formatters={
            "predicted_rating": "{:.3f}".format,
            "evidence_weight": "{:.3f}".format,
        }))


if __name__ == "__main__":
    main()
