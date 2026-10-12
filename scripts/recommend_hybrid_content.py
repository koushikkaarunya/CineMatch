"""Phase 9: content-only hybrid recommendations from plots and movie metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Let this file work both as `python scripts/...py` and `python -m scripts...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recommend_metadata import GENRE_WEIGHT, TAG_WEIGHT, jaccard_similarity, prepare_catalog
from scripts.recommend_tfidf import TfidfMovieRecommender
from scripts.recommend_transformer import (
    EMBEDDINGS_PATH,
    MOVIE_COLUMNS as TEXT_MOVIE_COLUMNS,
    TransformerMovieRecommender,
    load_sentence_model,
    plot_fingerprint,
    usable_plot_mask,
)


ENRICHED_MOVIES_PATH = PROJECT_ROOT / "data" / "enriched" / "movies.csv"
CHUNKED_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "models" / "all_minilm_l6_v2_plots_chunked.npz"

# These are design-time defaults for an initial student baseline. They have not
# been tuned against evaluation data or the three known failure examples.
DEFAULT_WEIGHTS = {
    "metadata": 0.34,
    "tfidf": 0.16,
    "transformer": 0.22,
    "director": 0.10,
    "writer": 0.07,
    "cast": 0.07,
    "crew": 0.04,
}
IDENTIFIER_PATTERN = re.compile(r"Q[0-9]+$")
RESULT_COLUMNS = [
    "movie_id", "title", "hybrid_score", "positive_signals", "explanation",
    "evidence_coverage", "missing_signals", "genre_similarity", "tag_similarity", "tfidf_similarity",
    "transformer_similarity", "director_similarity", "writer_similarity",
    "cast_similarity", "crew_similarity",
    "metadata_similarity", "metadata_rank_percentile", "tfidf_rank_percentile",
    "transformer_rank_percentile", "director_rank_percentile", "writer_rank_percentile",
    "cast_rank_percentile", "crew_rank_percentile",
]


def parse_people(value: object, *, crew: bool = False) -> tuple[set[str], dict[str, str]]:
    """Read structured Phase 8 person records, using stable QIDs as identities."""
    if pd.isna(value) or not str(value).strip():
        return set(), {}
    try:
        records = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return set(), {}
    if not isinstance(records, list):
        return set(), {}

    identifiers: set[str] = set()
    names: dict[str, str] = {}
    for person in records:
        if not isinstance(person, dict):
            continue
        qid = person.get("wikidata_id")
        if not isinstance(qid, str) or not IDENTIFIER_PATTERN.fullmatch(qid):
            continue
        identity = f"{person.get('role', '')}:{qid}" if crew else qid
        identifiers.add(identity)
        if person.get("name"):
            names[identity] = str(person["name"])
    return identifiers, names


def _rank_percentiles(values: dict[int, float]) -> dict[int, float]:
    """Map comparable component scores to [0, 1] ranks; ties receive average rank."""
    if not values:
        return {}
    series = pd.Series(values, dtype="float64")
    if len(series) == 1:
        return {int(series.index[0]): 1.0}
    ranks = series.rank(method="average", ascending=True)
    return {int(movie_id): float((rank - 1) / (len(series) - 1))
            for movie_id, rank in ranks.items()}


def combine_rank_scores(
    ranks: dict[str, float],
    available_signals: set[str],
    weights: dict[str, float],
    query_available_weight: float,
) -> tuple[float, float]:
    """Fuse comparable ranks and downweight candidates with sparse evidence."""
    available_weight = sum(weights[name] for name in available_signals)
    if available_weight <= 0 or query_available_weight <= 0:
        return 0.0, 0.0
    weighted_rank = sum(weights[name] * ranks[name] for name in available_signals) / available_weight
    coverage = min(1.0, available_weight / query_available_weight)
    return weighted_rank * coverage, coverage


def pair_metadata_similarity(first: pd.Series, second: pd.Series) -> float | None:
    """Apply Phase 5's 0.7/0.3 weights over metadata actually present on both movies."""
    comparisons = []
    if first["genre_labels"] and second["genre_labels"]:
        comparisons.append((GENRE_WEIGHT, jaccard_similarity(first["genre_labels"], second["genre_labels"])))
    if first["tag_labels"] and second["tag_labels"]:
        comparisons.append((TAG_WEIGHT, jaccard_similarity(first["tag_labels"], second["tag_labels"])))
    if not comparisons:
        return None
    total_weight = sum(weight for weight, _ in comparisons)
    return sum(weight * score for weight, score in comparisons) / total_weight


class HybridContentRecommender:
    """Combine the Phase 5–7 signals with Phase 8 credits using weighted ranks."""

    def __init__(
        self,
        enriched_movies: pd.DataFrame,
        *,
        tfidf_model: TfidfMovieRecommender | None = None,
        transformer_model: TransformerMovieRecommender | None = None,
        weights: dict[str, float] | None = None,
        min_positive_signals: int = 2,
    ) -> None:
        required = {
            "movie_id", "title", "release_year", "genres", "plot_tags", "plot",
            "directors", "writers", "cast", "selected_crew",
        }
        missing = required - set(enriched_movies.columns)
        if missing:
            raise ValueError(f"Enriched catalog is missing columns: {sorted(missing)}")
        if enriched_movies["movie_id"].duplicated().any():
            raise ValueError("Enriched catalog contains duplicate movie_id values")
        if not isinstance(min_positive_signals, int) or min_positive_signals < 1:
            raise ValueError("min_positive_signals must be a positive integer")

        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("weights must provide exactly the configured component names")
        if any(not np.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("component weights must be finite and nonnegative")
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("component weights must sum to 1.0")
        self.min_positive_signals = min_positive_signals

        self.catalog = enriched_movies.copy().reset_index(drop=True)
        self.catalog["movie_id"] = pd.to_numeric(self.catalog["movie_id"], errors="raise").astype(int)
        self.catalog_ids = set(self.catalog["movie_id"])
        self.movie_id_to_position = {
            int(movie_id): position for position, movie_id in enumerate(self.catalog["movie_id"])
        }
        metadata = prepare_catalog(self.catalog)
        self.catalog["genre_labels"] = metadata["genre_labels"].tolist()
        self.catalog["tag_labels"] = metadata["tag_labels"].tolist()

        self.credit_sets: dict[str, list[set[str]]] = {key: [] for key in ("director", "writer", "cast", "crew")}
        self.credit_names: dict[str, list[dict[str, str]]] = {key: [] for key in self.credit_sets}
        field_for_signal = {
            "director": "directors", "writer": "writers", "cast": "cast", "crew": "selected_crew"
        }
        for signal, field in field_for_signal.items():
            for value in self.catalog[field]:
                identifiers, names = parse_people(value, crew=signal == "crew")
                self.credit_sets[signal].append(identifiers)
                self.credit_names[signal].append(names)

        text_movies = self.catalog[TEXT_MOVIE_COLUMNS].copy()
        self.tfidf = tfidf_model or TfidfMovieRecommender(text_movies)
        self.transformer = transformer_model or TransformerMovieRecommender(
            text_movies, cache_path=EMBEDDINGS_PATH
        )
        expected_ids = self.catalog["movie_id"].tolist()
        if set(self.tfidf.catalog_ids) != set(expected_ids) or set(self.transformer.catalog_ids) != set(expected_ids):
            raise ValueError("Text models and enriched catalog have different movie ID sets")

    def _query_available_weight(self, query_position: int) -> float:
        query = self.catalog.iloc[query_position]
        available = {
            "metadata": bool(query["genre_labels"] or query["tag_labels"]),
            "tfidf": int(query["movie_id"]) in self.tfidf.movie_id_to_row,
            "transformer": int(query["movie_id"]) in self.transformer.movie_id_to_row,
        }
        for signal in ("director", "writer", "cast", "crew"):
            available[signal] = bool(self.credit_sets[signal][query_position])
        return sum(self.weights[name] for name, exists in available.items() if exists)

    def _component_scores(self, query_position: int) -> dict[str, dict[int, float]]:
        query = self.catalog.iloc[query_position]
        query_id = int(query["movie_id"])
        candidates = self.catalog.loc[self.catalog["movie_id"].ne(query_id)]
        scores: dict[str, dict[int, float]] = {name: {} for name in self.weights}
        scores["genre"] = {}
        scores["tags"] = {}

        query_tfidf_row = self.tfidf.movie_id_to_row.get(query_id)
        if query_tfidf_row is not None:
            vector = self.tfidf.tfidf_matrix[query_tfidf_row]
            tfidf_values = cosine_similarity(vector, self.tfidf.tfidf_matrix).ravel()
            for candidate_id, row in self.tfidf.movie_id_to_row.items():
                if candidate_id != query_id:
                    scores["tfidf"][candidate_id] = float(tfidf_values[row])

        query_transformer_row = self.transformer.movie_id_to_row.get(query_id)
        if query_transformer_row is not None:
            vector = self.transformer.embeddings[query_transformer_row:query_transformer_row + 1]
            transformer_values = cosine_similarity(vector, self.transformer.embeddings).ravel()
            for candidate_id, row in self.transformer.movie_id_to_row.items():
                if candidate_id != query_id:
                    scores["transformer"][candidate_id] = float(transformer_values[row])

        catalog_position = {int(movie_id): pos for pos, movie_id in enumerate(self.catalog["movie_id"])}
        for _, candidate in candidates.iterrows():
            candidate_id = int(candidate["movie_id"])
            position = catalog_position[candidate_id]
            same_genres = bool(query["genre_labels"] and candidate["genre_labels"])
            same_tags = bool(query["tag_labels"] and candidate["tag_labels"])
            if same_genres:
                scores["genre"][candidate_id] = jaccard_similarity(
                    query["genre_labels"], candidate["genre_labels"]
                )
            if same_tags:
                scores["tags"][candidate_id] = jaccard_similarity(
                    query["tag_labels"], candidate["tag_labels"]
                )
            metadata_score = pair_metadata_similarity(query, candidate)
            if metadata_score is not None:
                scores["metadata"][candidate_id] = float(metadata_score)
            for signal in ("director", "writer", "cast", "crew"):
                first = self.credit_sets[signal][query_position]
                second = self.credit_sets[signal][position]
                if first and second:
                    scores[signal][candidate_id] = jaccard_similarity(first, second)
        return scores

    def _explanation(self, query_position: int, candidate_position: int,
                     raw: dict[str, float]) -> str:
        query = self.catalog.iloc[query_position]
        candidate = self.catalog.iloc[candidate_position]
        evidence: list[str] = []
        shared_genres = sorted(query["genre_labels"] & candidate["genre_labels"])
        shared_tags = sorted(query["tag_labels"] & candidate["tag_labels"])
        if shared_genres:
            evidence.append("shared genres: " + ", ".join(shared_genres))
        if shared_tags:
            evidence.append("shared plot tags: " + ", ".join(shared_tags))
        if raw.get("tfidf", 0) > 0:
            evidence.append(f"plot word overlap (TF-IDF cosine {raw['tfidf']:.3f})")
        if raw.get("transformer", 0) > 0:
            evidence.append(f"plot semantic match (Transformer cosine {raw['transformer']:.3f})")
        candidate_position = self.movie_id_to_position[int(candidate["movie_id"])]
        for signal, label in (("director", "shared director"), ("writer", "shared writer"),
                              ("cast", "shared cast member"), ("crew", "shared selected crew")):
            if raw.get(signal, 0) <= 0:
                continue
            query_names = self.credit_names[signal][query_position]
            candidate_names = self.credit_names[signal][candidate_position]
            common_ids = self.credit_sets[signal][query_position] & self.credit_sets[signal][candidate_position]
            names = sorted({query_names.get(qid) or candidate_names.get(qid) or qid for qid in common_ids})
            evidence.append(f"{label}: " + ", ".join(names[:3]))
        if not evidence:
            return "No positive component evidence; excluded from recommendations."
        return "; ".join(evidence[:5])

    def recommend(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        if not isinstance(movie_id, (int, np.integer)) or isinstance(movie_id, bool):
            raise ValueError("movie_id must be an integer")
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1")
        if int(movie_id) not in self.catalog_ids:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        query_id = int(movie_id)
        query_position = int(self.catalog.index[self.catalog["movie_id"].eq(query_id)][0])
        query = self.catalog.iloc[query_position]
        query_weight = self._query_available_weight(query_position)
        if query_weight == 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        component_scores = self._component_scores(query_position)
        component_ranks = {name: _rank_percentiles(values) for name, values in component_scores.items()}
        recommendations = []
        for candidate in self.catalog.itertuples(index=False):
            candidate_id = int(candidate.movie_id)
            if candidate_id == query_id:
                continue
            available = [name for name in self.weights if candidate_id in component_scores[name]]
            if not available:
                continue
            positive = [name for name in available if component_scores[name][candidate_id] > 0]
            if len(positive) < self.min_positive_signals:
                continue

            candidate_ranks = {name: component_ranks[name][candidate_id] for name in available}
            final_score, coverage = combine_rank_scores(
                candidate_ranks, set(available), self.weights, query_weight
            )
            raw = {name: component_scores[name][candidate_id] for name in available}
            missing = [name for name in self.weights if name not in available]
            candidate_row = self.catalog.iloc[self.movie_id_to_position[candidate_id]]
            if query["genre_labels"] and not candidate_row["genre_labels"]:
                missing.append("genres")
            if query["tag_labels"] and not candidate_row["tag_labels"]:
                missing.append("plot tags")
            recommendations.append({
                "movie_id": candidate_id,
                "title": candidate.title,
                "hybrid_score": final_score,
                "positive_signals": len(positive),
                "explanation": self._explanation(
                    query_position, self.movie_id_to_position[candidate_id], raw
                ),
                "evidence_coverage": coverage,
                "missing_signals": ", ".join(missing),
                "metadata_similarity": raw.get("metadata", np.nan),
                "genre_similarity": component_scores["genre"].get(candidate_id, np.nan),
                "tag_similarity": component_scores["tags"].get(candidate_id, np.nan),
                **{f"{name}_similarity": raw.get(name, np.nan)
                   for name in self.weights if name != "metadata"},
                **{f"{name}_rank_percentile": candidate_ranks.get(name, np.nan) for name in self.weights},
            })

        if not recommendations:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        results = pd.DataFrame(recommendations, columns=RESULT_COLUMNS)
        return results.sort_values(
            ["hybrid_score", "positive_signals", "movie_id"],
            ascending=[False, False, True], kind="mergesort",
        ).head(int(top_k)).reset_index(drop=True)


def load_enriched_movies(path: Path = ENRICHED_MOVIES_PATH) -> pd.DataFrame:
    """Load the enriched catalog only; ratings are deliberately never read."""
    return pd.read_csv(path, dtype={
        "title": "string", "imdb_id": "string", "genres": "string", "plot": "string",
        "plot_tags": "string", "directors": "string", "writers": "string", "cast": "string",
        "selected_crew": "string",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid content-only movie recommendations.")
    parser.add_argument("movie_id", type=int, help="MovieLens movie_id to use as the query")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        recommender = HybridContentRecommender(load_enriched_movies())
        result = recommender.recommend(args.movie_id, args.top_k)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    query_title = recommender.catalog.loc[recommender.catalog.movie_id.eq(args.movie_id), "title"].iloc[0]
    print(f"Hybrid content recommendations for {query_title} (movie_id={args.movie_id})")
    print("Score: weighted percentile-rank fusion × available-signal coverage; not a probability.")
    if result.empty:
        print("No candidates had enough positive, available content signals.")
    else:
        print(result.to_string(index=False, formatters={"hybrid_score": "{:.3f}".format}))


if __name__ == "__main__":
    main()
