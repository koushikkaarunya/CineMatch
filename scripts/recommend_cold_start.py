"""Phase 11: clear content-based options when rating evidence is missing."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence as SequenceABC
from typing import Sequence

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
from scripts.recommend_hybrid_content import (
    HybridContentRecommender,
    load_enriched_movies,
    parse_people,
)
from scripts.recommend_metadata import split_labels


AGGREGATE_COLUMNS = [
    "movie_id", "title", "mean_reciprocal_rank", "seed_support_count",
    "best_seed_rank", "seed_ranks", "phase9_hybrid_scores", "seed_explanations",
]
EMPTY_AGGREGATE = pd.DataFrame(columns=AGGREGATE_COLUMNS)


@dataclass(frozen=True)
class ColdStartResponse:
    """Keep status, recommendation source, and result scores separate."""

    status: str
    message: str
    recommendation_source: str
    recommendations: pd.DataFrame
    seed_movie_ids: tuple[int, ...] = ()
    collaborative_evidence: str | None = None
    available_content_fields: tuple[str, ...] = ()


class ColdStartRecommender:
    """Coordinate existing Phase 9 and Phase 10 interfaces without blending scores."""

    def __init__(
        self,
        movies: pd.DataFrame,
        content_recommender: HybridContentRecommender,
        collaborative_recommender: ItemItemCollaborativeRecommender,
    ) -> None:
        if not {"movie_id", "title"}.issubset(movies.columns):
            raise ValueError("Movie catalog must contain movie_id and title")
        if movies["movie_id"].isna().any() or movies["movie_id"].duplicated().any():
            raise ValueError("Movie catalog must have unique, nonmissing movie_id values")
        catalog = movies.copy()
        ids = pd.to_numeric(catalog["movie_id"], errors="coerce")
        if ids.isna().any() or not np.equal(ids, np.floor(ids)).all():
            raise ValueError("Movie catalog movie_id values must be integers")
        catalog["movie_id"] = ids.astype("int64")
        self.movies = catalog.set_index("movie_id", drop=False)
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.catalog_ids = set(self.movies.index.astype(int))

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1")

    def _normalize_seed_ids(self, seed_movie_ids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(seed_movie_ids, (str, bytes)) or not isinstance(seed_movie_ids, SequenceABC):
            raise ValueError("seed_movie_ids must be a sequence of integer movie IDs")
        try:
            supplied_ids = list(seed_movie_ids)
        except TypeError as error:
            raise ValueError("seed_movie_ids must be a sequence of integer movie IDs") from error

        unique_ids: list[int] = []
        seen: set[int] = set()
        for movie_id in supplied_ids:
            if not isinstance(movie_id, (int, np.integer)) or isinstance(movie_id, bool):
                raise ValueError("Every seed movie ID must be an integer")
            movie_id = int(movie_id)
            if movie_id not in self.catalog_ids:
                raise KeyError(f"Unknown movie_id: {movie_id}")
            if movie_id not in seen:
                unique_ids.append(movie_id)
                seen.add(movie_id)
        return tuple(unique_ids)

    def _empty_response(
        self,
        status: str,
        message: str,
        *,
        source: str = "none",
        seed_movie_ids: tuple[int, ...] = (),
        collaborative_evidence: str | None = None,
        available_content_fields: tuple[str, ...] = (),
    ) -> ColdStartResponse:
        return ColdStartResponse(
            status=status,
            message=message,
            recommendation_source=source,
            recommendations=EMPTY_AGGREGATE.copy(),
            seed_movie_ids=seed_movie_ids,
            collaborative_evidence=collaborative_evidence,
            available_content_fields=available_content_fields,
        )

    def recommend_from_favorites(
        self, seed_movie_ids: Sequence[int], top_k: int = 10
    ) -> ColdStartResponse:
        """Aggregate Phase 9 rankings from movies a new user explicitly likes."""
        self._validate_top_k(top_k)
        seeds = self._normalize_seed_ids(seed_movie_ids)
        if not seeds:
            return self._empty_response(
                "needs_preferences",
                "Choose a few movies you like first; no individual preference has been provided.",
            )

        # Each seed contributes its returned rank list; Phase 9 scores are never
        # averaged across queries because their candidate distributions differ.
        candidate_details: dict[int, dict[int, dict[str, object]]] = {}
        seeds_with_results = 0
        for seed_id in seeds:
            result = self.content_recommender.recommend(seed_id, top_k=top_k + len(seeds))
            if not isinstance(result, pd.DataFrame):
                raise TypeError("Phase 9 recommend() must return a pandas DataFrame")
            if not {"movie_id", "hybrid_score"}.issubset(result.columns):
                raise ValueError("Phase 9 results must include movie_id and hybrid_score")

            # Remove duplicates and all selected seeds before assigning ordinal ranks.
            seen_candidates: set[int] = set()
            ranked_rows: list[tuple[int, float, str]] = []
            for row in result.itertuples(index=False):
                candidate_id = getattr(row, "movie_id")
                if not isinstance(candidate_id, (int, np.integer)) or isinstance(candidate_id, bool):
                    raise ValueError("Phase 9 candidate movie IDs must be integers")
                candidate_id = int(candidate_id)
                if candidate_id not in self.catalog_ids:
                    raise ValueError(f"Phase 9 returned movie_id outside the catalog: {candidate_id}")
                if candidate_id in seeds or candidate_id in seen_candidates:
                    continue
                score = float(getattr(row, "hybrid_score"))
                if not np.isfinite(score):
                    raise ValueError("Phase 9 hybrid_score values must be finite")
                explanation = getattr(row, "explanation", "")
                ranked_rows.append((candidate_id, score, str(explanation)))
                seen_candidates.add(candidate_id)

            if ranked_rows:
                seeds_with_results += 1
            for rank, (candidate_id, hybrid_score, explanation) in enumerate(ranked_rows, start=1):
                candidate_details.setdefault(candidate_id, {})[seed_id] = {
                    "rank": rank,
                    "phase9_hybrid_score": hybrid_score,
                    "explanation": explanation,
                }

        if not candidate_details:
            return self._empty_response(
                "no_content_evidence",
                "None of the selected movies produced usable Phase 9 candidates. Try other favorites; no preferences or metadata were invented.",
                seed_movie_ids=seeds,
            )

        rows = []
        for candidate_id, per_seed in candidate_details.items():
            reciprocal_rank_sum = sum(1.0 / detail["rank"] for detail in per_seed.values())
            seed_ranks = {seed: int(detail["rank"]) for seed, detail in per_seed.items()}
            hybrid_scores = {
                seed: float(detail["phase9_hybrid_score"])
                for seed, detail in per_seed.items()
            }
            explanations = {
                seed: str(detail["explanation"])
                for seed, detail in per_seed.items()
                if detail["explanation"]
            }
            rows.append({
                "movie_id": candidate_id,
                "title": self.movies.loc[candidate_id, "title"],
                # Missing seed lists contribute zero; divide by every unique seed.
                "mean_reciprocal_rank": reciprocal_rank_sum / len(seeds),
                "seed_support_count": len(per_seed),
                "best_seed_rank": min(seed_ranks.values()),
                "seed_ranks": seed_ranks,
                "phase9_hybrid_scores": hybrid_scores,
                "seed_explanations": explanations,
            })
        recommendations = pd.DataFrame(rows, columns=AGGREGATE_COLUMNS).sort_values(
            ["mean_reciprocal_rank", "seed_support_count", "best_seed_rank", "movie_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(top_k).reset_index(drop=True)

        if seeds_with_results == len(seeds):
            status = "content_based_from_selected_movies"
            message = (
                "Recommendations use the Phase 9 experimental content model and the selected movies; "
                "this is not collaborative filtering or a quality guarantee."
            )
        else:
            status = "content_based_from_partial_seed_evidence"
            message = (
                f"Phase 9 returned candidates for {seeds_with_results} of {len(seeds)} selected movies. "
                "The other seed lists contributed no candidates; the results remain content-based and experimental."
            )
        return ColdStartResponse(
            status=status,
            message=message,
            recommendation_source="phase9_content_rank_aggregation",
            recommendations=recommendations,
            seed_movie_ids=seeds,
        )

    def _collaborative_movie_evidence(self, movie_id: int) -> str:
        collaborative = self.collaborative_recommender
        if movie_id not in getattr(collaborative, "catalog_ids", set()):
            return "not_in_collaborative_catalog"
        if movie_id not in collaborative.movie_to_column:
            return "no_ratings_in_collaborative_data"
        row = collaborative.movie_to_column[movie_id]
        if collaborative.similarity_matrix.getrow(row).nnz == 0:
            return "no_qualifying_item_similarity"
        return "item_similarity_available"

    def _available_content_fields(self, movie_id: int) -> tuple[str, ...]:
        movie = self.movies.loc[movie_id]
        available: list[str] = []
        if split_labels(movie.get("genres", pd.NA), "|"):
            available.append("genres")
        if split_labels(movie.get("plot_tags", pd.NA), ","):
            available.append("plot_tags")
        plot = movie.get("plot", pd.NA)
        if not pd.isna(plot) and str(plot).strip():
            available.append("plot")
        for field, label in (
            ("directors", "directors"),
            ("writers", "writers"),
            ("cast", "cast"),
            ("selected_crew", "selected_crew"),
        ):
            identifiers, _ = parse_people(movie.get(field, pd.NA), crew=field == "selected_crew")
            if identifiers:
                available.append(label)
        return tuple(available)

    def recommend_for_movie(self, movie_id: int, top_k: int = 10) -> ColdStartResponse:
        """Request Phase 9 content results for a catalog movie and report CF evidence separately."""
        self._validate_top_k(top_k)
        if not isinstance(movie_id, (int, np.integer)) or isinstance(movie_id, bool):
            raise ValueError("movie_id must be an integer")
        movie_id = int(movie_id)
        if movie_id not in self.catalog_ids:
            raise KeyError(f"Unknown movie_id: {movie_id}")

        evidence = self._collaborative_movie_evidence(movie_id)
        available_fields = self._available_content_fields(movie_id)
        if not available_fields:
            return self._empty_response(
                "no_content_metadata",
                "This movie is in the catalog but has no usable Phase 9 content fields; no description or credits were fabricated.",
                collaborative_evidence=evidence,
            )

        recommendations = self.content_recommender.recommend(movie_id, top_k=top_k)
        if recommendations.empty:
            return self._empty_response(
                "no_content_candidates",
                "Content fields exist, but Phase 9 returned no candidates meeting its signal requirements.",
                collaborative_evidence=evidence,
                available_content_fields=available_fields,
            )

        if evidence == "item_similarity_available":
            status = "content_recommendations_with_collaborative_evidence"
            message = (
                "These are Phase 9 content-based recommendations. Item-item evidence also exists, "
                "but no collaborative and content scores were combined."
            )
        else:
            status = "content_fallback_without_collaborative_evidence"
            message = (
                "These are Phase 9 content-based recommendations because this movie has no retained "
                "Phase 10 item-item relationship in the supplied ratings."
            )
        return ColdStartResponse(
            status=status,
            message=message,
            recommendation_source="phase9_content",
            recommendations=recommendations.copy(),
            collaborative_evidence=evidence,
            available_content_fields=available_fields,
        )

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        favorite_movie_ids: Sequence[int] | None = None,
    ) -> ColdStartResponse:
        """Use Phase 10 when supported; otherwise ask for or use explicit favorites."""
        self._validate_top_k(top_k)
        if not isinstance(user_id, (int, np.integer)) or isinstance(user_id, bool):
            raise ValueError("user_id must be an integer")
        user_id = int(user_id)
        seeds = self._normalize_seed_ids(
            favorite_movie_ids if favorite_movie_ids is not None else ()
        )
        collaborative = self.collaborative_recommender
        history = getattr(collaborative, "user_history", {}).get(user_id)
        if history is None:
            return self._empty_response(
                "unknown_user",
                f"MovieLens user_id {user_id} is unknown. For a new user, provide favorite movie IDs instead.",
            )

        user_mean = float(collaborative.user_means[user_id])
        has_too_little_history = len(history) < 2
        has_no_rating_variation = all(
            np.isclose(rating - user_mean, 0.0) for rating in history.values()
        )
        insufficient_reason = None
        if has_too_little_history:
            insufficient_reason = f"only {len(history)} rating(s) are available"
        elif has_no_rating_variation:
            insufficient_reason = "the ratings have no variation around this user's mean"

        if insufficient_reason is None:
            collaborative_results = collaborative.recommend(user_id, top_k=top_k)
            if not collaborative_results.empty:
                return ColdStartResponse(
                    status="collaborative_recommendations_available",
                    message="Phase 10 has enough rating evidence; these are collaborative predicted ratings, not content scores.",
                    recommendation_source="phase10_collaborative",
                    recommendations=collaborative_results.copy(),
                    collaborative_evidence="user_history_and_item_similarities_available",
                )
            insufficient_reason = "no unseen candidates have retained item-item evidence"

        if not seeds:
            return self._empty_response(
                "needs_preferences",
                f"Phase 10 cannot personalize this known user because {insufficient_reason}. "
                "Ask the user to select a few movies they like for a content-based fallback.",
                collaborative_evidence="insufficient_for_user_recommendation",
            )

        fallback = self.recommend_from_favorites(seeds, top_k=top_k)
        if fallback.recommendations.empty:
            return ColdStartResponse(
                status="insufficient_collaborative_and_content_evidence",
                message=(
                    f"Phase 10 lacks evidence because {insufficient_reason}; the supplied favorite movies "
                    "also produced no usable Phase 9 candidates. Try different favorites."
                ),
                recommendation_source="none",
                recommendations=EMPTY_AGGREGATE.copy(),
                seed_movie_ids=fallback.seed_movie_ids,
                collaborative_evidence="insufficient_for_user_recommendation",
            )
        return ColdStartResponse(
            status="content_fallback_for_insufficient_collaborative_history",
            message=(
                f"Phase 10 lacks evidence because {insufficient_reason}. These recommendations use only "
                "the explicitly selected favorite movies and Phase 9 content rankings."
            ),
            recommendation_source=fallback.recommendation_source,
            recommendations=fallback.recommendations,
            seed_movie_ids=fallback.seed_movie_ids,
            collaborative_evidence="insufficient_for_user_recommendation",
        )


def build_cold_start_recommender() -> ColdStartRecommender:
    """Load existing Phase 9/10 models for command-line demonstrations."""
    enriched_movies = load_enriched_movies()
    content = HybridContentRecommender(enriched_movies)
    collaborative = ItemItemCollaborativeRecommender(load_ratings(), load_cleaned_movies())
    return ColdStartRecommender(enriched_movies, content, collaborative)


def _print_response(response: ColdStartResponse) -> None:
    print(f"Status: {response.status}")
    print(f"Recommendation source: {response.recommendation_source}")
    print(response.message)
    if response.collaborative_evidence:
        print(f"Collaborative evidence: {response.collaborative_evidence}")
    if response.available_content_fields:
        print("Available content fields: " + ", ".join(response.available_content_fields))
    if not response.recommendations.empty:
        print(response.recommendations.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate CineMatch Phase 11 cold-start handling.")
    commands = parser.add_subparsers(dest="command", required=True)

    new_user = commands.add_parser("new-user", help="recommend from explicitly selected favorites")
    new_user.add_argument("--favorites", type=int, nargs="*", default=[])
    new_user.add_argument("--top-k", type=int, default=10)

    known_user = commands.add_parser("user", help="use Phase 10 or an explicit-favorite fallback")
    known_user.add_argument("user_id", type=int)
    known_user.add_argument("--favorites", type=int, nargs="*", default=[])
    known_user.add_argument("--top-k", type=int, default=10)

    movie = commands.add_parser("movie", help="request content suggestions for a catalog movie")
    movie.add_argument("movie_id", type=int)
    movie.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    try:
        recommender = build_cold_start_recommender()
        if args.command == "new-user":
            response = recommender.recommend_from_favorites(args.favorites, args.top_k)
        elif args.command == "user":
            response = recommender.recommend_for_user(args.user_id, args.top_k, args.favorites)
        else:
            response = recommender.recommend_for_movie(args.movie_id, args.top_k)
    except (KeyError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    _print_response(response)


if __name__ == "__main__":
    main()
