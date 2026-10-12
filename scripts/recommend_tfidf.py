"""Phase 6 plot-text recommendations using TF-IDF and cosine similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies.csv"
MOVIE_COLUMNS = ["movie_id", "title", "release_year", "plot"]
RESULT_COLUMNS = ["movie_id", "title", "release_year", "similarity"]


class TfidfMovieRecommender:
    """Fit a TF-IDF plot model and use it for repeated movie lookups."""

    def __init__(self, movies: pd.DataFrame) -> None:
        missing_columns = set(MOVIE_COLUMNS) - set(movies.columns)
        if missing_columns:
            raise ValueError(f"Movie catalog is missing columns: {sorted(missing_columns)}")
        if movies["movie_id"].duplicated().any():
            raise ValueError("Movie catalog contains duplicate movie_id values")

        # Keep the full catalog for ID checks; only plot-bearing rows get vectors.
        self.catalog = movies[MOVIE_COLUMNS].copy()
        self.catalog_ids = set(self.catalog["movie_id"].astype(int))
        has_plot = self.catalog["plot"].fillna("").astype("string").str.strip().ne("")
        self.plot_movies = self.catalog.loc[has_plot].copy().reset_index(drop=True)
        self.plot_movies["plot"] = self.plot_movies["plot"].astype("string").str.strip()
        if self.plot_movies.empty:
            raise ValueError("No movies have usable plot text")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50_000,
        )
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.plot_movies["plot"])
        except ValueError as error:
            raise ValueError("Could not build a TF-IDF vocabulary from the available plots") from error

        # This mapping preserves the row-to-movie relationship after missing plots are filtered.
        self.movie_id_to_row = {
            int(movie_id): row for row, movie_id in enumerate(self.plot_movies["movie_id"])
        }

    def recommend(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return unique plot-bearing movies ranked by cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if movie_id not in self.catalog_ids:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        if movie_id not in self.movie_id_to_row:
            raise ValueError(f"Movie {movie_id} has no usable plot for TF-IDF comparison")

        query_row = self.movie_id_to_row[movie_id]
        query_vector = self.tfidf_matrix[query_row]
        if query_vector.nnz == 0:
            raise ValueError(f"Movie {movie_id}'s plot has no terms in the TF-IDF vocabulary")

        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        scores[query_row] = -1.0  # Exclude the query itself before ranking.
        candidate_rows = [row for row, score in enumerate(scores) if score > 0]
        if not candidate_rows:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        results = self.plot_movies.iloc[candidate_rows][MOVIE_COLUMNS[:3]].copy()
        results["similarity"] = scores[candidate_rows]
        return results.sort_values(
            ["similarity", "movie_id"], ascending=[False, True], kind="mergesort"
        ).head(top_k).reset_index(drop=True)


def load_movies(path: Path = MOVIES_PATH) -> pd.DataFrame:
    """Read the cleaned movie catalog fields needed by TF-IDF."""
    return pd.read_csv(
        path,
        usecols=MOVIE_COLUMNS,
        dtype={"title": "string", "release_year": "Int64", "plot": "string"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies from plot text using TF-IDF.")
    parser.add_argument("movie_id", type=int, help="MovieLens movie_id to use as the query")
    parser.add_argument("--top-k", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    movies = load_movies()
    query = movies.loc[movies["movie_id"].eq(args.movie_id)]
    if query.empty:
        parser.error(f"unknown movie_id: {args.movie_id}")
    if pd.isna(query.iloc[0]["plot"]) or not str(query.iloc[0]["plot"]).strip():
        parser.error(f"movie_id {args.movie_id} has no usable plot; use the metadata recommender separately")

    recommender = TfidfMovieRecommender(movies)
    try:
        results = recommender.recommend(args.movie_id, args.top_k)
    except ValueError as error:
        parser.error(str(error))

    print(f"TF-IDF recommendations for {query.iloc[0]['title']} (movie_id={args.movie_id})")
    if results.empty:
        print("No other movies have positive plot similarity to this movie.")
    else:
        print(results.to_string(index=False, formatters={"similarity": "{:.3f}".format}))


if __name__ == "__main__":
    main()
