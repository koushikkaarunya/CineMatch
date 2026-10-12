"""Phase 7 plot recommendations using pretrained sentence embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "models" / "all_minilm_l6_v2_plots.npz"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_DIMENSION = 384
ENCODE_BATCH_SIZE = 32
MOVIE_COLUMNS = ["movie_id", "title", "release_year", "plot"]
RESULT_COLUMNS = ["movie_id", "title", "release_year", "similarity"]
ARTIFACT_VERSION = 1


def usable_plot_mask(movies: pd.DataFrame) -> pd.Series:
    """Identify plots with at least one non-whitespace character."""
    return movies["plot"].fillna("").astype("string").str.strip().ne("")


def plot_fingerprint(plot_movies: pd.DataFrame) -> str:
    """Hash ordered IDs and plots so cached vectors cannot drift out of alignment."""
    digest = hashlib.sha256()
    for movie_id, plot in zip(plot_movies["movie_id"], plot_movies["plot"]):
        digest.update(str(int(movie_id)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(plot).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_sentence_model(model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    """Load the requested model explicitly; never fall back to another model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Phase 7 requires sentence-transformers. Install the pinned Phase 7 "
            "requirements before running this recommender."
        ) from error

    try:
        return SentenceTransformer(model_id, revision=revision, device="cpu")
    except Exception as error:
        raise RuntimeError(
            f"Could not load {model_id!r}. Check internet access/model cache and "
            "dependency compatibility; no substitute model was loaded."
        ) from error


class TransformerMovieRecommender:
    """Cache plot embeddings and rank movies by cosine similarity."""

    def __init__(
        self,
        movies: pd.DataFrame,
        model=None,
        embeddings: np.ndarray | None = None,
        cache_path: Path | None = None,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        batch_size: int = ENCODE_BATCH_SIZE,
    ) -> None:
        missing_columns = set(MOVIE_COLUMNS) - set(movies.columns)
        if missing_columns:
            raise ValueError(f"Movie catalog is missing columns: {sorted(missing_columns)}")
        if movies["movie_id"].duplicated().any():
            raise ValueError("Movie catalog contains duplicate movie_id values")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # The full catalog distinguishes an unknown ID from a known movie without a plot.
        self.catalog = movies[MOVIE_COLUMNS].copy().reset_index(drop=True)
        self.catalog["movie_id"] = pd.to_numeric(self.catalog["movie_id"], errors="raise").astype(int)
        self.catalog_ids = set(self.catalog["movie_id"])
        self.plot_movies = self.catalog.loc[usable_plot_mask(self.catalog)].copy().reset_index(drop=True)
        self.plot_movies["plot"] = self.plot_movies["plot"].astype("string").str.strip()
        if self.plot_movies.empty:
            raise ValueError("No movies have usable plot text")

        self.model_id = model_id
        self.model_revision = model_revision
        self.batch_size = batch_size
        self.fingerprint = plot_fingerprint(self.plot_movies)
        self.movie_id_to_row = {
            int(movie_id): row for row, movie_id in enumerate(self.plot_movies["movie_id"])
        }

        if embeddings is not None:
            self.embeddings = self._validate_embeddings(embeddings)
        elif cache_path is not None and self._load_cache(cache_path):
            pass
        else:
            self.model = model if model is not None else load_sentence_model(model_id, model_revision)
            vectors = self.model.encode(
                self.plot_movies["plot"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.embeddings = self._validate_embeddings(vectors)
            if cache_path is not None:
                self._save_cache(cache_path)

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (len(self.plot_movies), EXPECTED_DIMENSION)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"Expected embeddings with shape {expected_shape}, got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Expected L2-normalized embeddings for cosine similarity")
        return vectors

    def _load_cache(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as artifact:
                metadata = json.loads(str(artifact["metadata"].item()))
                ids = artifact["movie_ids"].astype(np.int64)
                vectors = artifact["embeddings"]
            if (
                metadata.get("artifact_version") != ARTIFACT_VERSION
                or metadata.get("model_id") != self.model_id
                or metadata.get("model_revision") != self.model_revision
                or metadata.get("plot_fingerprint") != self.fingerprint
                or not np.array_equal(ids, self.plot_movies["movie_id"].to_numpy())
                or metadata.get("embedding_dimension") != EXPECTED_DIMENSION
                or metadata.get("normalized") is not True
            ):
                return False
            self.embeddings = self._validate_embeddings(vectors)
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # A broken or stale cache is rebuilt from source plots on the next step.
            return False

    def _save_cache(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "embedding_dimension": EXPECTED_DIMENSION,
            "batch_size": self.batch_size,
            "normalized": True,
            "plot_fingerprint": self.fingerprint,
        }
        temporary_path = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            metadata=np.array(json.dumps(metadata)),
            movie_ids=self.plot_movies["movie_id"].to_numpy(dtype=np.int64),
            embeddings=self.embeddings,
        )
        temporary_path.replace(cache_path)

    def recommend(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return unique plot-bearing movies ranked by embedding cosine similarity."""
        if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1")
        if movie_id not in self.catalog_ids:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        if movie_id not in self.movie_id_to_row:
            raise ValueError(f"Movie {movie_id} has no usable plot for semantic comparison")

        query_row = self.movie_id_to_row[movie_id]
        scores = cosine_similarity(self.embeddings[query_row:query_row + 1], self.embeddings)[0]
        scores[query_row] = -1.0
        candidate_rows = np.flatnonzero(scores > 0)
        if len(candidate_rows) == 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        results = self.plot_movies.iloc[candidate_rows][MOVIE_COLUMNS[:3]].copy()
        results["similarity"] = scores[candidate_rows]
        return results.sort_values(
            ["similarity", "movie_id"], ascending=[False, True], kind="mergesort"
        ).head(top_k).reset_index(drop=True)


def load_movies(path: Path = MOVIES_PATH) -> pd.DataFrame:
    """Read only movie ID, display fields, and plot text; ratings are never loaded."""
    return pd.read_csv(
        path,
        usecols=MOVIE_COLUMNS,
        dtype={"title": "string", "release_year": "Int64", "plot": "string"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend movies by pretrained Transformer plot embeddings."
    )
    parser.add_argument("movie_id", type=int, help="MovieLens movie_id to use as the query")
    parser.add_argument("--top-k", type=int, default=10, help="number of recommendations")
    parser.add_argument(
        "--rebuild-embeddings", action="store_true",
        help="ignore the saved embeddings and encode all available plots again",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    movies = load_movies()
    query = movies.loc[movies["movie_id"].eq(args.movie_id)]
    if query.empty:
        parser.error(f"unknown movie_id: {args.movie_id}")
    if not usable_plot_mask(query).iloc[0]:
        parser.error(f"movie_id {args.movie_id} has no usable plot for semantic comparison")

    cache_path = None if args.rebuild_embeddings else EMBEDDINGS_PATH
    try:
        recommender = TransformerMovieRecommender(movies, cache_path=cache_path)
        results = recommender.recommend(args.movie_id, args.top_k)
    except (RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))

    print(f"Transformer recommendations for {query.iloc[0]['title']} (movie_id={args.movie_id})")
    print(f"Model: {MODEL_ID}@{MODEL_REVISION}; cosine similarity (scores are not probabilities)")
    if results.empty:
        print("No other plot-bearing movies were available.")
    else:
        print(results.to_string(index=False, formatters={"similarity": "{:.3f}".format}))


if __name__ == "__main__":
    main()
