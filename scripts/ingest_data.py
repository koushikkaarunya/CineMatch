from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MPST_URL = (
    "https://huggingface.co/datasets/cryptexcode/MPST/resolve/main/"
    "mpst_full_data.csv?download=true"
)


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}...")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(destination)


def main() -> None:
    movielens_dir = RAW_DIR / "movielens"
    mpst_dir = RAW_DIR / "mpst"
    movielens_zip = movielens_dir / "ml-latest-small.zip"
    mpst_csv = mpst_dir / "mpst_full_data.csv"

    download(MOVIELENS_URL, movielens_zip)
    download(MPST_URL, mpst_csv)

    with zipfile.ZipFile(movielens_zip) as archive:
        archive.extractall(movielens_dir)

    movielens_files = movielens_dir / "ml-latest-small"
    movies = pd.read_csv(movielens_files / "movies.csv")
    links = pd.read_csv(movielens_files / "links.csv", dtype={"imdbId": "string"})
    ratings = pd.read_csv(movielens_files / "ratings.csv")
    plots = pd.read_csv(
        mpst_csv,
        usecols=["imdb_id", "plot_synopsis", "tags", "synopsis_source"],
        dtype="string",
    )

    imdb_digits = links["imdbId"].fillna("").str.strip()
    links["imdb_id"] = "tt" + imdb_digits.str.zfill(7)
    links.loc[imdb_digits.eq(""), "imdb_id"] = pd.NA

    plots["imdb_id"] = plots["imdb_id"].str.strip().str.lower()
    plots["plot_synopsis"] = plots["plot_synopsis"].fillna("").str.strip()
    wikipedia_plots = plots.loc[
        plots["synopsis_source"].str.lower().eq("wikipedia")
        & plots["plot_synopsis"].ne("")
    ].copy()

    duplicate_plot_rows_removed = int(wikipedia_plots["imdb_id"].duplicated().sum())
    wikipedia_plots = wikipedia_plots.drop_duplicates("imdb_id", keep="first")

    catalog = movies.merge(
        links[["movieId", "imdb_id"]], on="movieId", how="left", validate="one_to_one"
    )
    catalog = catalog.merge(
        wikipedia_plots[["imdb_id", "plot_synopsis", "tags"]],
        on="imdb_id",
        how="left",
        validate="many_to_one",
    )
    catalog["release_year"] = catalog["title"].str.extract(r"\((\d{4})\)\s*$")
    catalog = catalog.rename(
        columns={
            "movieId": "movie_id",
            "plot_synopsis": "overview",
            "tags": "plot_tags",
        }
    )
    catalog = catalog[
        ["movie_id", "imdb_id", "title", "release_year", "genres", "overview", "plot_tags"]
    ]

    ratings = ratings.sort_values("timestamp")
    duplicate_rating_rows_removed = int(
        ratings.duplicated(["userId", "movieId"], keep="last").sum()
    )
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last")
    ratings = ratings.rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )[["user_id", "movie_id", "rating", "timestamp"]]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    ratings.to_csv(PROCESSED_DIR / "ratings.csv", index=False)

    summary = {
        "movielens_movies": int(len(catalog)),
        "movielens_ratings_after_duplicate_policy": int(len(ratings)),
        "movies_with_wikipedia_plot": int(catalog["overview"].notna().sum()),
        "movies_without_plot": int(catalog["overview"].isna().sum()),
        "duplicate_wikipedia_plot_rows_removed": duplicate_plot_rows_removed,
        "duplicate_rating_rows_removed": duplicate_rating_rows_removed,
        "plot_source_filter": "Wikipedia rows only",
        "movie_id_source": "MovieLens movieId",
    }
    (PROCESSED_DIR / "ingestion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Processed files written to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
