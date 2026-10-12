"""Simple Phase 5 movie-to-movie recommendations from genres and plot tags."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies.csv"
GENRE_WEIGHT = 0.7
TAG_WEIGHT = 0.3
RESULT_COLUMNS = [
    "movie_id", "title", "release_year", "similarity", "shared_genres", "shared_tags"
]


def split_labels(value: object, separator: str) -> set[str]:
    """Turn one delimited metadata field into a normalized set of labels."""
    if pd.isna(value):
        return set()

    labels = {part.strip().casefold() for part in str(value).split(separator) if part.strip()}
    labels.discard("(no genres listed)")
    return labels


def prepare_catalog(movies: pd.DataFrame) -> pd.DataFrame:
    """Add genre and tag sets to a copy of the cleaned movie catalog."""
    required = {"movie_id", "title", "release_year", "genres", "plot_tags"}
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"Movie catalog is missing columns: {sorted(missing)}")
    if movies["movie_id"].duplicated().any():
        raise ValueError("Movie catalog contains duplicate movie_id values")

    catalog = movies.copy()
    catalog["genre_labels"] = catalog["genres"].map(lambda value: split_labels(value, "|"))
    catalog["tag_labels"] = catalog["plot_tags"].map(lambda value: split_labels(value, ","))
    return catalog


def load_catalog(path: Path = MOVIES_PATH) -> pd.DataFrame:
    """Read only the Phase 4 movie fields used or displayed by this baseline."""
    movies = pd.read_csv(
        path,
        usecols=["movie_id", "title", "release_year", "genres", "plot_tags"],
        dtype={"title": "string", "genres": "string", "plot_tags": "string"},
    )
    return prepare_catalog(movies)


def jaccard_similarity(first: set[str], second: set[str]) -> float:
    """Measure overlap between two label sets, returning zero for an empty union."""
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def metadata_similarity(first: pd.Series, second: pd.Series) -> float | None:
    """Score shared metadata; missing candidate fields add no similarity evidence."""
    first_genres, second_genres = first["genre_labels"], second["genre_labels"]
    first_tags, second_tags = first["tag_labels"], second["tag_labels"]
    query_weight = (GENRE_WEIGHT if first_genres else 0.0) + (TAG_WEIGHT if first_tags else 0.0)
    if not query_weight:
        return None

    score = 0.0
    if first_genres and second_genres:
        score += GENRE_WEIGHT * jaccard_similarity(first_genres, second_genres)
    if first_tags and second_tags:
        score += TAG_WEIGHT * jaccard_similarity(first_tags, second_tags)
    return score / query_weight


def recommend_similar_movies(
    catalog: pd.DataFrame, movie_id: int, top_k: int = 10
) -> pd.DataFrame:
    """Return the highest-scoring catalog movies, excluding the input movie."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if catalog["movie_id"].duplicated().any():
        raise ValueError("Movie catalog contains duplicate movie_id values")

    selected = catalog.loc[catalog["movie_id"].eq(movie_id)]
    if selected.empty:
        raise KeyError(f"Unknown movie_id: {movie_id}")
    query = selected.iloc[0]
    if not query["genre_labels"] and not query["tag_labels"]:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    recommendations = []
    candidates = catalog.loc[catalog["movie_id"].ne(movie_id)]
    for _, candidate in candidates.iterrows():
        score = metadata_similarity(query, candidate)
        if score is None or score <= 0:
            continue
        recommendations.append({
            "movie_id": int(candidate["movie_id"]),
            "title": candidate["title"],
            "release_year": candidate["release_year"],
            "similarity": score,
            "shared_genres": ", ".join(sorted(query["genre_labels"] & candidate["genre_labels"])),
            "shared_tags": ", ".join(sorted(query["tag_labels"] & candidate["tag_labels"])),
        })

    if not recommendations:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    results = pd.DataFrame(recommendations, columns=RESULT_COLUMNS)
    return results.sort_values(
        ["similarity", "movie_id"], ascending=[False, True], kind="mergesort"
    ).head(top_k).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find movies similar by genre and plot tags.")
    parser.add_argument("movie_id", type=int, help="MovieLens movie_id to use as the query")
    parser.add_argument("--top-k", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    catalog = load_catalog()
    query = catalog.loc[catalog["movie_id"].eq(args.movie_id)]
    if query.empty:
        parser.error(f"unknown movie_id: {args.movie_id}")

    print(f"Similar movies to {query.iloc[0]['title']} (movie_id={args.movie_id})")
    results = recommend_similar_movies(catalog, args.movie_id, args.top_k)
    if results.empty:
        print("No similar movies found from the available metadata.")
    else:
        print(results.to_string(index=False, formatters={"similarity": "{:.3f}".format}))


if __name__ == "__main__":
    main()
