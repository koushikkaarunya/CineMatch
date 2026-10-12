"""Phase 12: coordinate content, collaborative, and cold-start recommenders."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recommend_collaborative import (
    ItemItemCollaborativeRecommender,
    load_movies as load_cleaned_movies,
    load_ratings,
)
from scripts.recommend_cold_start import (
    ColdStartRecommender,
    ColdStartResponse,
)
from scripts.recommend_hybrid_content import HybridContentRecommender, load_enriched_movies


RESULT_COLUMNS = [
    "movie_id", "title", "fusion_score", "contributing_sources",
    "collaborative_rank", "content_rank", "predicted_rating",
    "support_count", "evidence_weight", "content_mean_reciprocal_rank",
    "content_explanation",
]
POSITIVE_SEED_CAP = 5


@dataclass(frozen=True)
class HybridResponse:
    """Keep routing status and scores distinguishable from recommendation rows."""

    status: str
    message: str
    recommendation_source: str
    recommendations: pd.DataFrame
    positive_seed_movie_ids: tuple[int, ...] = ()
    collaborative_available: bool = False
    content_available: bool = False


class HybridRecommender:
    """Fuse existing Phase 9 and Phase 10 rankings without changing either model."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        cold_start_recommender: ColdStartRecommender,
        collaborative_recommender: ItemItemCollaborativeRecommender,
        *,
        positive_seed_cap: int = POSITIVE_SEED_CAP,
    ) -> None:
        required = {"movie_id", "title"}
        if not required.issubset(movies.columns):
            raise ValueError("Movie catalog must contain movie_id and title")
        if movies["movie_id"].isna().any() or movies["movie_id"].duplicated().any():
            raise ValueError("Movie catalog must have unique, nonmissing movie_id values")
        if not {"user_id", "movie_id", "rating"}.issubset(ratings.columns):
            raise ValueError("Ratings must contain user_id, movie_id, and rating")
        if not isinstance(positive_seed_cap, int) or isinstance(positive_seed_cap, bool) or positive_seed_cap < 1:
            raise ValueError("positive_seed_cap must be a positive integer")

        self.movies = movies.copy()
        movie_ids = pd.to_numeric(self.movies["movie_id"], errors="coerce")
        if movie_ids.isna().any() or not np.equal(movie_ids, np.floor(movie_ids)).all():
            raise ValueError("Movie catalog movie_id values must be integers")
        self.movies["movie_id"] = movie_ids.astype("int64")
        self.movies = self.movies.set_index("movie_id", drop=False)
        self.catalog_ids = set(self.movies.index.astype(int))

        # This table is the only user-history source used for seed selection.
        self.ratings = ratings[["user_id", "movie_id", "rating"]].copy()
        for column in ("user_id", "movie_id"):
            values = pd.to_numeric(self.ratings[column], errors="coerce")
            if values.isna().any() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{column} values must be integers")
            self.ratings[column] = values.astype("int64")
        self.ratings["rating"] = pd.to_numeric(self.ratings["rating"], errors="coerce")
        if self.ratings["rating"].isna().any() or not np.isfinite(self.ratings["rating"]).all():
            raise ValueError("Ratings must be finite numbers")
        if self.ratings.duplicated(["user_id", "movie_id"]).any():
            raise ValueError("Ratings contain duplicate user_id/movie_id pairs")
        if not set(self.ratings["movie_id"]).issubset(self.catalog_ids):
            raise ValueError("Ratings reference movie IDs absent from the supplied catalog")

        fitted_interactions = getattr(collaborative_recommender, "ratings", None)
        if isinstance(fitted_interactions, pd.DataFrame):
            expected = self.ratings.sort_values(
                ["user_id", "movie_id"], kind="mergesort"
            ).reset_index(drop=True)
            actual = fitted_interactions[["user_id", "movie_id", "rating"]].copy()
            actual = actual.sort_values(["user_id", "movie_id"], kind="mergesort").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=True)
            except AssertionError as error:
                raise ValueError(
                    "The collaborative model must be fitted on the same supplied ratings used for user history"
                ) from error

        self.cold_start_recommender = cold_start_recommender
        self.collaborative_recommender = collaborative_recommender
        self.positive_seed_cap = positive_seed_cap

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1")

    @staticmethod
    def reciprocal_rank(rank: int | None) -> float:
        """Return 1/rank, or zero when the candidate is absent from a list."""
        if rank is None:
            return 0.0
        if not isinstance(rank, (int, np.integer)) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer or None")
        return 1.0 / int(rank)

    def _empty(
        self,
        status: str,
        message: str,
        *,
        source: str = "none",
        seeds: tuple[int, ...] = (),
        collaborative_available: bool = False,
        content_available: bool = False,
    ) -> HybridResponse:
        return HybridResponse(
            status, message, source, pd.DataFrame(columns=RESULT_COLUMNS), seeds,
            collaborative_available, content_available,
        )

    def _positive_seeds(self, history: pd.DataFrame) -> tuple[int, ...]:
        """Select this user's above-average movies, highest ratings first."""
        if history.empty:
            return ()
        user_average = float(history["rating"].mean())
        positives = history.loc[history["rating"].gt(user_average)].sort_values(
            ["rating", "movie_id"], ascending=[False, True], kind="mergesort"
        )
        return tuple(positives["movie_id"].astype(int).head(self.positive_seed_cap).tolist())

    @staticmethod
    def _ranked_unique(
        frame: pd.DataFrame,
        *,
        seen_ids: set[int],
        catalog_ids: set[int],
        score_column: str,
    ) -> dict[int, dict[str, object]]:
        """Filter exclusions and duplicate IDs while preserving model order."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Recommendation models must return pandas DataFrames")
        if frame.empty:
            return {}
        if "movie_id" not in frame.columns:
            raise ValueError("Recommendation results must include movie_id")
        rows: dict[int, dict[str, object]] = {}
        for row in frame.to_dict(orient="records"):
            raw_id = row["movie_id"]
            if not isinstance(raw_id, (int, np.integer)) or isinstance(raw_id, bool):
                raise ValueError("Recommendation candidate IDs must be integers")
            movie_id = int(raw_id)
            if movie_id not in catalog_ids:
                raise ValueError(f"Recommendation model returned unknown movie_id: {movie_id}")
            if movie_id in seen_ids or movie_id in rows:
                continue
            value = row.get(score_column)
            if value is not None and not pd.isna(value) and not np.isfinite(float(value)):
                raise ValueError(f"Recommendation {score_column} values must be finite")
            rows[movie_id] = row
        return rows

    def _fuse(
        self,
        collaborative_rows: dict[int, dict[str, object]],
        content_rows: dict[int, dict[str, object]],
        *,
        available_sources: tuple[str, ...],
        top_k: int,
        seeds: tuple[int, ...] = (),
    ) -> HybridResponse:
        if not collaborative_rows and not content_rows:
            return self._empty(
                "no_recommendations",
                "Neither collaborative nor content sources returned eligible unseen movies.",
                seeds=seeds,
            )

        # A source that returned a valid list contributes zero for candidates
        # missing from that list. An unavailable source is omitted from the mean.
        candidates = set(collaborative_rows) | set(content_rows)
        rows: list[dict[str, object]] = []
        source_count = len(available_sources)
        for movie_id in candidates:
            collab = collaborative_rows.get(movie_id)
            content = content_rows.get(movie_id)
            collab_rank = collab.get("_rank") if collab else None
            content_rank = content.get("_rank") if content else None
            reciprocal_sum = 0.0
            if "collaborative" in available_sources:
                reciprocal_sum += self.reciprocal_rank(collab_rank)
            if "content" in available_sources:
                reciprocal_sum += self.reciprocal_rank(content_rank)
            fusion_score = reciprocal_sum / source_count
            sources = []
            if collab:
                sources.append("collaborative")
            if content:
                sources.append("content")
            rows.append({
                "movie_id": movie_id,
                "title": self.movies.loc[movie_id, "title"],
                "fusion_score": fusion_score,
                "contributing_sources": ", ".join(sources),
                "collaborative_rank": collab_rank,
                "content_rank": content_rank,
                "predicted_rating": collab.get("predicted_rating") if collab else np.nan,
                "support_count": collab.get("support_count") if collab else np.nan,
                "evidence_weight": collab.get("evidence_weight") if collab else np.nan,
                "content_mean_reciprocal_rank": content.get("mean_reciprocal_rank") if content else np.nan,
                "content_explanation": content.get("seed_explanations") if content else None,
            })
        recommendations = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
            ["fusion_score", "movie_id"], ascending=[False, True], kind="mergesort"
        ).head(top_k).reset_index(drop=True)
        source_label = "+".join(available_sources)
        if len(available_sources) == 2:
            status = "hybrid_recommendations"
        elif available_sources == ("collaborative",):
            status = "collaborative_only"
        else:
            status = "content_only"
        return HybridResponse(
            status,
            "Fusion scores order the list; they are not probabilities or predicted ratings.",
            source_label,
            recommendations,
            seeds,
            bool(collaborative_rows),
            bool(content_rows),
        )

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> HybridResponse:
        """Personalize from supplied user history and the matching CF model."""
        self._validate_top_k(top_k)
        if not isinstance(user_id, (int, np.integer)) or isinstance(user_id, bool):
            raise ValueError("user_id must be an integer")
        user_id = int(user_id)
        history = self.ratings.loc[self.ratings["user_id"].eq(user_id)].copy()
        if history.empty:
            return self._empty("unknown_user", f"Unknown user_id: {user_id}")

        seen_ids = set(history["movie_id"].astype(int))
        seeds = self._positive_seeds(history)
        # Oversample the independent lists so filtering seen movies does not
        # unnecessarily leave the final Top-K short. This is a retrieval cap,
        # not a tuned model parameter.
        source_k = max(top_k * 5, top_k) + len(seen_ids)

        collaborative_rows: dict[int, dict[str, object]] = {}
        history_mean = float(history["rating"].mean())
        enough_history = len(history) >= 2 and not np.allclose(history["rating"], history_mean)
        if enough_history:
            raw_collaborative = self.collaborative_recommender.recommend(user_id, top_k=source_k)
            collaborative_rows = self._ranked_unique(
                raw_collaborative,
                seen_ids=seen_ids,
                catalog_ids=self.catalog_ids,
                score_column="predicted_rating",
            )
            for rank, row in enumerate(collaborative_rows.values(), start=1):
                row["_rank"] = rank

        content_rows: dict[int, dict[str, object]] = {}
        if seeds:
            cold_response: ColdStartResponse = self.cold_start_recommender.recommend_from_favorites(
                seeds, top_k=source_k
            )
            content_rows = self._ranked_unique(
                cold_response.recommendations,
                seen_ids=seen_ids,
                catalog_ids=self.catalog_ids,
                score_column="mean_reciprocal_rank",
            )
            for rank, row in enumerate(content_rows.values(), start=1):
                row["_rank"] = rank

        available_sources = tuple(
            source for source, candidates in (
                ("collaborative", collaborative_rows), ("content", content_rows)
            ) if candidates
        )
        response = self._fuse(
            collaborative_rows, content_rows,
            available_sources=available_sources, top_k=top_k, seeds=seeds,
        )
        if response.status == "no_recommendations":
            if not enough_history and not seeds:
                return self._empty(
                    "insufficient_history_and_no_positive_seeds",
                    "This known user has too little rating variation for collaborative recommendations and no above-average rated movies for a content profile.",
                )
            if not collaborative_rows and not content_rows:
                return self._empty(
                    "no_eligible_evidence",
                    "Neither source produced unseen recommendations. More ratings or explicit favorite movies may help.",
                    seeds=seeds,
                )
        if not collaborative_rows and content_rows:
            if enough_history:
                fallback_status = "content_fallback_for_insufficient_collaborative_evidence"
                fallback_message = (
                    "The collaborative model returned no eligible unseen candidates; these results use the user's above-average rated movies through Phase 11."
                )
            else:
                fallback_status = "content_fallback_for_insufficient_history"
                fallback_message = (
                    "Collaborative filtering needs more varied history; these results use the user's above-average rated movies through Phase 11."
                )
            return HybridResponse(
                fallback_status,
                fallback_message,
                response.recommendation_source,
                response.recommendations,
                seeds,
                False,
                True,
            )
        return response

    def recommend_for_new_user(
        self, favorite_movie_ids: list[int] | tuple[int, ...], top_k: int = 10
    ) -> HybridResponse:
        """Delegate a new user's explicit favorites to the Phase 11 coordinator."""
        self._validate_top_k(top_k)
        cold: ColdStartResponse = self.cold_start_recommender.recommend_from_favorites(
            favorite_movie_ids, top_k=top_k
        )
        recommendations = cold.recommendations.copy()
        if not recommendations.empty:
            recommendations = recommendations.loc[
                ~recommendations["movie_id"].isin(cold.seed_movie_ids)
            ].drop_duplicates("movie_id", keep="first").head(top_k).reset_index(drop=True)
        return HybridResponse(
            cold.status, cold.message, cold.recommendation_source, recommendations,
            cold.seed_movie_ids, False, not recommendations.empty,
        )

    def recommend_for_movie(self, movie_id: int, top_k: int = 10) -> HybridResponse:
        """Keep movie-to-movie recommendations as a Phase 9 content operation."""
        self._validate_top_k(top_k)
        cold = self.cold_start_recommender.recommend_for_movie(movie_id, top_k=top_k)
        return HybridResponse(
            cold.status, cold.message, cold.recommendation_source,
            cold.recommendations.copy(), (), bool(cold.collaborative_evidence == "item_similarity_available"),
            not cold.recommendations.empty,
        )


def build_hybrid_recommender(ratings: pd.DataFrame | None = None) -> HybridRecommender:
    """Build all models from the same explicitly supplied interaction table."""
    enriched_movies = load_enriched_movies()
    cleaned_movies = load_cleaned_movies()
    supplied_ratings = load_ratings() if ratings is None else ratings.copy()
    content = HybridContentRecommender(enriched_movies)
    collaborative = ItemItemCollaborativeRecommender(supplied_ratings, cleaned_movies)
    cold_start = ColdStartRecommender(enriched_movies, content, collaborative)
    return HybridRecommender(enriched_movies, supplied_ratings, cold_start, collaborative)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate CineMatch Phase 12 recommendation routing.")
    commands = parser.add_subparsers(dest="command", required=True)
    user = commands.add_parser("user", help="personalized known-user recommendations")
    user.add_argument("user_id", type=int)
    user.add_argument("--top-k", type=int, default=10)
    new_user = commands.add_parser("new-user", help="recommend from selected favorite movies")
    new_user.add_argument("--favorites", type=int, nargs="*", default=[])
    new_user.add_argument("--top-k", type=int, default=10)
    movie = commands.add_parser("movie", help="content recommendations for a selected movie")
    movie.add_argument("movie_id", type=int)
    movie.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    try:
        recommender = build_hybrid_recommender()
        if args.command == "user":
            response = recommender.recommend_for_user(args.user_id, args.top_k)
        elif args.command == "new-user":
            response = recommender.recommend_for_new_user(args.favorites, args.top_k)
        else:
            response = recommender.recommend_for_movie(args.movie_id, args.top_k)
    except (KeyError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Status: {response.status}")
    print(f"Recommendation source: {response.recommendation_source}")
    print(response.message)
    if response.positive_seed_movie_ids:
        print("Positive seed movie IDs: " + ", ".join(map(str, response.positive_seed_movie_ids)))
    if not response.recommendations.empty:
        print(response.recommendations.to_string(index=False))
    else:
        print("No recommendations returned.")


if __name__ == "__main__":
    main()
