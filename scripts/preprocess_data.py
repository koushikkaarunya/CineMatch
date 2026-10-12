"""Phase 4: clean and validate Phase 2 tables without changing their inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

MOVIE_COLUMNS = [
    "movie_id", "imdb_id", "title", "release_year", "genres", "plot", "plot_tags"
]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def require_columns(frame: pd.DataFrame, expected: list[str], table_name: str) -> None:
    """Fail clearly if an upstream schema changes unexpectedly."""
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def normalize_positive_integer_ids(values: pd.Series, name: str) -> pd.Series:
    """Normalize numeric IDs to int64; invalid or nonpositive IDs stop the run."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & numeric.mod(1).eq(0) & numeric.gt(0)
    if not valid.all():
        bad_rows = int((~valid).sum())
        raise ValueError(f"{name} contains {bad_rows} missing, noninteger, or nonpositive IDs")
    return numeric.astype("int64")


def normalize_imdb_id(value: Any) -> Any:
    """Return canonical tt-prefixed IMDb IDs, retaining missing IDs as pandas NA."""
    if pd.isna(value):
        return pd.NA
    identifier = str(value).strip().lower()
    if not identifier:
        return pd.NA
    digits = identifier[2:] if identifier.startswith("tt") else identifier
    if not digits.isdigit():
        raise ValueError(f"Invalid IMDb identifier: {value!r}")
    return "tt" + digits.zfill(7)


def clean_display_text(value: Any) -> Any:
    """Trim display text and collapse spacing without changing case or punctuation."""
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else pd.NA


def clean_plot(value: Any) -> Any:
    """Normalize plot whitespace while preserving words, case, punctuation and paragraphs."""
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text if text else pd.NA


def clean_genres(value: Any) -> Any:
    """Trim pipe-separated genre labels and deduplicate while preserving their order."""
    if pd.isna(value):
        return pd.NA
    tokens = [token.strip() for token in str(value).split("|")]
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.casefold()
        if token and key not in seen:
            unique_tokens.append(token)
            seen.add(key)
    return "|".join(unique_tokens) if unique_tokens else pd.NA


def clean_plot_tags(value: Any) -> Any:
    """Normalize comma-separated MPST tags to lowercase labels, preserving first-seen order."""
    if pd.isna(value):
        return pd.NA
    tokens = [re.sub(r"\s+", " ", token).strip().casefold()
              for token in str(value).split(",")]
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    return ", ".join(unique_tokens) if unique_tokens else pd.NA


def validate_release_year(values: pd.Series) -> tuple[pd.Series, int]:
    """Convert valid year values to nullable integers and report malformed values."""
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_nonmissing = values.notna() & numeric.isna()
    fractional = numeric.notna() & numeric.mod(1).ne(0)
    invalid = invalid_nonmissing | fractional
    normalized = numeric.mask(invalid).astype("Int64")
    return normalized, int(invalid.sum())


def prepare_movies(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean catalog fields and reject ambiguous duplicate identifiers."""
    require_columns(
        source,
        ["movie_id", "imdb_id", "title", "release_year", "genres", "overview", "plot_tags"],
        "movies",
    )
    movies = source.copy()
    movies["movie_id"] = normalize_positive_integer_ids(movies["movie_id"], "movie_id")
    movies["imdb_id"] = movies["imdb_id"].map(normalize_imdb_id).astype("string")
    movies["title"] = movies["title"].map(clean_display_text).astype("string")
    if movies["title"].isna().any():
        raise ValueError(f"Movie catalog contains {int(movies['title'].isna().sum())} blank titles")

    movies["release_year"], invalid_years = validate_release_year(movies["release_year"])
    movies["genres"] = movies["genres"].map(clean_genres).astype("string")
    movies["plot"] = movies["overview"].map(clean_plot).astype("string")
    movies["plot_tags"] = movies["plot_tags"].map(clean_plot_tags).astype("string")
    movies = movies[MOVIE_COLUMNS]

    duplicate_rows_removed = int(movies.duplicated().sum())
    movies = movies.drop_duplicates().copy()
    if movies["movie_id"].duplicated().any():
        raise ValueError("Conflicting movie records share a movie_id; inspect the source rows")
    imdb_ids = movies.loc[movies["imdb_id"].notna(), "imdb_id"]
    if imdb_ids.duplicated().any():
        raise ValueError("Multiple movie records share an IMDb ID; the crosswalk is ambiguous")

    counts = {
        "exact_duplicate_movie_rows_removed": duplicate_rows_removed,
        "invalid_release_years_set_missing": invalid_years,
    }
    return movies, counts


def prepare_ratings(source: pd.DataFrame, valid_movie_ids: set[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize rating event types, remove exact duplicate events, and validate references."""
    require_columns(source, RATING_COLUMNS, "ratings")
    ratings = source[RATING_COLUMNS].copy()
    ratings["user_id"] = normalize_positive_integer_ids(ratings["user_id"], "user_id")
    ratings["movie_id"] = normalize_positive_integer_ids(ratings["movie_id"], "rating movie_id")

    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    valid_rating = (
        ratings["rating"].between(0.5, 5.0)
        & (ratings["rating"] * 2).mod(1).eq(0)
    )
    if not valid_rating.all():
        raise ValueError(
            f"Ratings contain {int((~valid_rating).sum())} values outside the 0.5-to-5.0 half-star scale"
        )

    timestamps = pd.to_numeric(ratings["timestamp"], errors="coerce")
    valid_timestamp = timestamps.notna() & timestamps.mod(1).eq(0)
    if not valid_timestamp.all():
        raise ValueError(f"Ratings contain {int((~valid_timestamp).sum())} invalid timestamps")
    ratings["timestamp"] = timestamps.astype("int64")

    orphan_mask = ~ratings["movie_id"].isin(valid_movie_ids)
    if orphan_mask.any():
        examples = ratings.loc[orphan_mask, "movie_id"].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{int(orphan_mask.sum())} ratings refer to unknown movie IDs; examples: {examples}"
        )

    duplicate_events_removed = int(ratings.duplicated(RATING_COLUMNS).sum())
    ratings = ratings.drop_duplicates(RATING_COLUMNS).copy()
    return ratings, {
        "exact_duplicate_rating_events_removed": duplicate_events_removed,
        "rating_rows_with_unknown_movie_id": 0,
    }


def null_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return per-column missing-value counts for the output report."""
    return {column: int(count) for column, count in frame.isna().sum().items()}


def main() -> None:
    """Load Phase 2 outputs, create Phase 4 outputs, and print a validation summary."""
    movie_source = pd.read_csv(INPUT_DIR / "movies.csv", dtype="string")
    rating_source = pd.read_csv(INPUT_DIR / "ratings.csv")

    movies, movie_counts = prepare_movies(movie_source)
    ratings, rating_counts = prepare_ratings(rating_source, set(movies["movie_id"]))

    summary = {
        "phase": "Phase 4: Data Preprocessing",
        "inputs": {
            "movies": str(INPUT_DIR / "movies.csv"),
            "ratings": str(INPUT_DIR / "ratings.csv"),
        },
        "outputs": {
            "movies": str(OUTPUT_DIR / "movies.csv"),
            "ratings": str(OUTPUT_DIR / "ratings.csv"),
        },
        "movie_rows_in": int(len(movie_source)),
        "movie_rows_out": int(len(movies)),
        "movie_columns": MOVIE_COLUMNS,
        "movie_dtypes_before_csv_serialization": {
            name: str(dtype) for name, dtype in movies.dtypes.items()
        },
        "rating_rows_in": int(len(rating_source)),
        "rating_rows_out": int(len(ratings)),
        "rating_columns": RATING_COLUMNS,
        "rating_dtypes_before_csv_serialization": {
            name: str(dtype) for name, dtype in ratings.dtypes.items()
        },
        "movie_missing_values": null_counts(movies),
        "rating_missing_values": null_counts(ratings),
        "unique_movie_ids": int(movies["movie_id"].nunique()),
        "unique_imdb_ids_nonmissing": int(movies["imdb_id"].nunique()),
        "unique_users": int(ratings["user_id"].nunique()),
        "unique_rated_movies": int(ratings["movie_id"].nunique()),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "movie_cleaning_counts": movie_counts,
        "rating_cleaning_counts": rating_counts,
        "all_rating_movie_ids_exist_in_catalog": bool(
            ratings["movie_id"].isin(movies["movie_id"]).all()
        ),
        "not_added_because_sources_are_not_integrated": [
            "director", "writers", "principal_cast", "selected_crew", "poster_url_or_id"
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(OUTPUT_DIR / "movies.csv", index=False)
    ratings.to_csv(OUTPUT_DIR / "ratings.csv", index=False)
    (OUTPUT_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Phase 4 files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
