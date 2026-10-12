"""Phase 3 exploratory analysis for CineMatch's processed MovieLens/MPST data.

This is an analysis-only script. It does not train models or change source data.
Run from any directory with: ./.venv/bin/python analysis/phase3_eda.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase3_eda"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.
import matplotlib.pyplot as plt
import pandas as pd


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed tables with stable ID and timestamp types."""
    movies = pd.read_csv(
        PROCESSED_DIR / "movies.csv",
        dtype={"imdb_id": "string", "title": "string", "genres": "string",
               "overview": "string", "plot_tags": "string"},
    )
    ratings = pd.read_csv(
        PROCESSED_DIR / "ratings.csv",
        dtype={"user_id": "int64", "movie_id": "int64", "rating": "float64",
               "timestamp": "int64"},
    )
    return movies, ratings


def missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return missing-cell counts by column, including columns with zero missing."""
    return {column: int(count) for column, count in frame.isna().sum().items()}


def iqr_outlier_count(values: pd.Series) -> int:
    """Count values beyond the conventional 1.5-IQR fences; do not label them errors."""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    outside_fences = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return int(outside_fences.sum())


def make_figures(movies: pd.DataFrame, ratings: pd.DataFrame,
                 ratings_per_user: pd.Series, ratings_per_movie: pd.Series,
                 genre_counts: pd.Series, plot_lengths: pd.Series,
                 popular: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Rating values and the highly skewed interaction-count distributions.
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ratings["rating"].value_counts().sort_index().plot.bar(ax=axes[0], color="#3977a8")
    axes[0].set(title="Rating distribution", xlabel="Rating (stars)", ylabel="Number of ratings")
    axes[1].hist(ratings_per_user, bins=30, color="#3977a8")
    axes[1].set(title="Ratings per user", xlabel="Ratings submitted", ylabel="Users")
    axes[1].set_yscale("log")
    axes[2].hist(ratings_per_movie, bins=30, color="#3977a8")
    axes[2].set(title="Ratings per movie", xlabel="Ratings received", ylabel="Movies")
    axes[2].set_yscale("log")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "rating_and_activity_distributions.png", dpi=160)
    plt.close(figure)

    # Genres are stored as pipe-separated labels; count each movie once per genre.
    figure, axis = plt.subplots(figsize=(9, 6))
    genre_counts.head(15).sort_values().plot.barh(ax=axis, color="#3977a8")
    axis.set(title="Most common genres", xlabel="Movies containing genre", ylabel="Genre")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "genre_distribution.png", dpi=160)
    plt.close(figure)

    # Plot lengths apply only to movies with a non-empty overview.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(plot_lengths, bins=40, color="#3977a8")
    axes[0].set(title="Wikipedia plot length", xlabel="Characters per plot", ylabel="Movies")
    axes[1].bar(["Has plot", "Missing plot"],
                [int(movies["overview"].notna().sum()), int(movies["overview"].isna().sum())],
                color=["#3977a8", "#b7c9d8"])
    axes[1].set(title="Plot coverage", ylabel="Movies")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "plot_coverage_and_lengths.png", dpi=160)
    plt.close(figure)

    # Raw rating volume is useful for understanding exposure, not preference quality.
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(popular["title"].iloc[::-1], popular["rating_count"].iloc[::-1], color="#3977a8")
    axis.set(title="Most rated movies (raw rating count)", xlabel="Number of ratings", ylabel="Movie")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "most_rated_movies.png", dpi=160)
    plt.close(figure)


def main() -> None:
    movies, ratings = load_data()

    ratings_per_user = ratings.groupby("user_id").size().rename("rating_count")
    ratings_per_movie = ratings.groupby("movie_id").size().rename("rating_count")
    genre_counts = (
        movies["genres"].fillna("").str.split("|").explode()
        .loc[lambda values: values.ne("")].value_counts()
    )
    plot_lengths = movies["overview"].dropna().str.len()
    movie_rating_summary = ratings.groupby("movie_id")["rating"].agg(
        rating_count="count", mean_rating="mean"
    )
    movie_rating_summary = movie_rating_summary.join(
        movies.set_index("movie_id")["title"], how="left"
    )
    popular = movie_rating_summary.sort_values(
        ["rating_count", "mean_rating"], ascending=False
    ).head(15).reset_index()

    rating_join_check = ratings["movie_id"].isin(movies["movie_id"])
    rating_counts = ratings["rating"].value_counts().sort_index()
    rating_mean = float(ratings["rating"].mean())
    sparsity = 1 - len(ratings) / (ratings["user_id"].nunique() * movies["movie_id"].nunique())
    plot_missing = movies["overview"].isna() | movies["overview"].str.strip().eq("").fillna(False)
    plot_lengths_nonempty = movies.loc[~plot_missing, "overview"].str.len()
    plot_length_q1, plot_length_q3 = plot_lengths_nonempty.quantile([0.25, 0.75])
    plot_length_iqr = plot_length_q3 - plot_length_q1
    plot_length_outlier_count = int(
        ((plot_lengths_nonempty < plot_length_q1 - 1.5 * plot_length_iqr)
         | (plot_lengths_nonempty > plot_length_q3 + 1.5 * plot_length_iqr)).sum()
    )

    report = {
        "movies": {
            "rows": int(len(movies)), "columns": int(len(movies.columns)),
            "column_names": list(movies.columns),
            "dtypes": {name: str(dtype) for name, dtype in movies.dtypes.items()},
            "missing_by_column": missing_counts(movies),
            "exact_duplicate_rows": int(movies.duplicated().sum()),
            "duplicate_movie_ids": int(movies["movie_id"].duplicated().sum()),
            "unique_movie_ids": int(movies["movie_id"].nunique()),
            "duplicate_imdb_ids_nonmissing": int(
                movies.loc[movies["imdb_id"].notna(), "imdb_id"].duplicated().sum()
            ),
            "missing_plots": int(plot_missing.sum()),
            "plot_coverage_fraction": float((~plot_missing).mean()),
            "plot_length_characters": {
                "count": int(len(plot_lengths_nonempty)),
                "mean": float(plot_lengths_nonempty.mean()),
                "median": float(plot_lengths_nonempty.median()),
                "p90": float(plot_lengths_nonempty.quantile(0.90)),
                "max": int(plot_lengths_nonempty.max()),
                "iqr_rule_outlier_count": plot_length_outlier_count,
            },
            "release_year_numeric_range": [
                int(pd.to_numeric(movies["release_year"], errors="coerce").min()),
                int(pd.to_numeric(movies["release_year"], errors="coerce").max()),
            ],
            "genre_counts": {name: int(count) for name, count in genre_counts.items()},
        },
        "ratings": {
            "rows": int(len(ratings)), "columns": int(len(ratings.columns)),
            "column_names": list(ratings.columns),
            "dtypes": {name: str(dtype) for name, dtype in ratings.dtypes.items()},
            "missing_by_column": missing_counts(ratings),
            "exact_duplicate_rows": int(ratings.duplicated().sum()),
            "duplicate_user_movie_pairs": int(ratings.duplicated(["user_id", "movie_id"]).sum()),
            "unique_users": int(ratings["user_id"].nunique()),
            "unique_rated_movies": int(ratings["movie_id"].nunique()),
            "rating_distribution": {str(value): int(count) for value, count in rating_counts.items()},
            "rating_mean": rating_mean, "rating_median": float(ratings["rating"].median()),
            "rating_min": float(ratings["rating"].min()), "rating_max": float(ratings["rating"].max()),
            "ratings_outside_expected_0_5_to_5_range": int(
                (~ratings["rating"].between(0.5, 5)).sum()
            ),
            "ratings_per_user": {
                "mean": float(ratings_per_user.mean()), "median": float(ratings_per_user.median()),
                "p90": float(ratings_per_user.quantile(0.90)), "max": int(ratings_per_user.max()),
                "iqr_rule_outlier_count": iqr_outlier_count(ratings_per_user),
            },
            "ratings_per_movie": {
                "mean_including_unrated_catalog_movies": float(len(ratings) / len(movies)),
                "median_rated_movies_only": float(ratings_per_movie.median()),
                "p90_rated_movies_only": float(ratings_per_movie.quantile(0.90)),
                "max": int(ratings_per_movie.max()),
                "iqr_rule_outlier_count": iqr_outlier_count(ratings_per_movie),
                "unrated_catalog_movies": int(len(movies) - ratings["movie_id"].nunique()),
            },
            "rating_movie_ids_found_in_catalog": int(rating_join_check.sum()),
            "rating_movie_ids_missing_from_catalog": int((~rating_join_check).sum()),
            "timestamp_range_utc": [
                pd.to_datetime(ratings["timestamp"], unit="s", utc=True).min().isoformat(),
                pd.to_datetime(ratings["timestamp"], unit="s", utc=True).max().isoformat(),
            ],
            "rating_matrix_sparsity_fraction": float(sparsity),
            "top_15_by_raw_rating_count": popular.to_dict(orient="records"),
        },
        "leakage_and_design_notes": [
            "Ratings have timestamps: use a chronological evaluation split for next-period prediction.",
            "Compute popularity, user/movie aggregates, and learned preprocessing from training data only.",
            "For per-user ranking evaluation, ensure held-out interactions do not appear in user history.",
            "The matrix sparsity calculation assumes every catalog movie is a candidate for every user.",
        ],
    }

    make_figures(movies, ratings, ratings_per_user, ratings_per_movie,
                 genre_counts, plot_lengths_nonempty, popular)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "movies": report["movies"],
        "ratings": {key: value for key, value in report["ratings"].items()
                    if key not in {"top_15_by_raw_rating_count", "rating_distribution"}},
        "rating_distribution": report["ratings"]["rating_distribution"],
        "top_15_by_raw_rating_count": report["ratings"]["top_15_by_raw_rating_count"],
        "output_dir": str(OUTPUT_DIR),
    }, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
