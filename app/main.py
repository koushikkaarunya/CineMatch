"""Minimal FastAPI adapter around CineMatch's existing Phase 12 recommender."""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import FavoriteRecommendationsRequest
from scripts.recommend_hybrid import HybridResponse, build_hybrid_recommender


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Build the catalog and recommendation models once for this server process."""
    try:
        recommender = build_hybrid_recommender()
    except Exception as error:
        # Keep startup failure useful in server logs without exposing paths in API responses.
        raise RuntimeError("CineMatch could not load its catalog or recommendation models") from error

    application.state.recommender = recommender
    application.state.catalog = recommender.movies.reset_index(drop=True)
    application.state.catalog_ids = set(application.state.catalog["movie_id"].astype(int))
    yield


def configured_cors_origins() -> list[str]:
    """Read explicit browser origins; an unset value keeps same-origin defaults."""
    return [
        origin.strip().rstrip("/")
        for origin in os.getenv("CINEMATCH_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]


def add_configured_cors(application: FastAPI) -> list[str]:
    """Install narrow CORS rules only when deployment supplies browser origins."""
    origins = configured_cors_origins()
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Accept", "Content-Type"],
        )
    return origins


app = FastAPI(title="CineMatch API", version="1.0.0", lifespan=lifespan)
cors_origins = add_configured_cors(app)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    """Return concise 400 validation errors without echoing request data or internals."""
    details = [
        {
            "field": ".".join(str(part) for part in item.get("loc", ())[1:]),
            "message": item.get("msg", "Invalid value"),
        }
        for item in error.errors()
    ]
    return JSONResponse(status_code=400, content={"detail": details})


def _json_safe(value: Any) -> Any:
    """Convert Pandas/NumPy values recursively into standard JSON data."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value


def _split_values(value: Any, separator: str) -> list[str]:
    if value is None or value is pd.NA or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "(no genres listed)":
        return []
    return [part.strip() for part in text.split(separator) if part.strip()]


def _parse_credit_records(value: Any) -> list[dict[str, Any]]:
    """Parse stored JSON credit arrays; malformed or missing values become empty arrays."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [_json_safe(person) for person in value if isinstance(person, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        records = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(records, list):
        return []
    return [_json_safe(person) for person in records if isinstance(person, dict)]


def _optional_int(value: Any) -> int | None:
    safe = _json_safe(value)
    if safe is None:
        return None
    try:
        number = float(safe)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _movie_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "movie_id": _optional_int(row.get("movie_id")),
        "title": _json_safe(row.get("title")),
        "release_year": _optional_int(row.get("release_year")),
        "genres": _split_values(row.get("genres"), "|"),
    }


def _movie_details(row: pd.Series) -> dict[str, Any]:
    return {
        **_movie_summary(row),
        "plot": _json_safe(row.get("plot")),
        "plot_tags": _split_values(row.get("plot_tags"), ","),
        "directors": _parse_credit_records(row.get("directors")),
        "writers": _parse_credit_records(row.get("writers")),
        "cast": _parse_credit_records(row.get("cast")),
        "selected_crew": _parse_credit_records(row.get("selected_crew")),
        "external_ids": {
            "imdb_id": _json_safe(row.get("imdb_id")),
            "tmdb_id": _optional_int(row.get("tmdb_id")),
            "wikidata_id": _json_safe(row.get("wikidata_id")),
        },
    }


def _response_payload(response: HybridResponse) -> dict[str, Any]:
    return {
        "status": response.status,
        "message": response.message,
        "recommendation_source": response.recommendation_source,
        "positive_seed_movie_ids": list(response.positive_seed_movie_ids),
        "collaborative_available": response.collaborative_available,
        "content_available": response.content_available,
        "items": _json_safe(response.recommendations),
    }


def _catalog_movie(request: Request, movie_id: int) -> pd.Series:
    catalog = request.app.state.catalog
    match = catalog.loc[catalog["movie_id"].eq(movie_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown movie_id: {movie_id}")
    return match.iloc[0]


def _validate_top_k(top_k: int) -> None:
    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    """Report that the process is serving; never rebuild models here."""
    return {"status": "ok"}


@app.get("/api/v1/movies")
def search_movies(
    request: Request,
    q: str = Query(...),
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    if not 1 <= limit <= 50:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="q must contain a non-whitespace title query")
    catalog = request.app.state.catalog
    matches = catalog.loc[
        catalog["title"].fillna("").astype("string").str.contains(query, case=False, regex=False)
    ].head(limit)
    return {
        "items": [_movie_summary(row) for _, row in matches.iterrows()],
        "count": int(len(matches)),
    }


@app.get("/api/v1/movies/{movie_id}")
def get_movie(movie_id: int, request: Request) -> dict[str, Any]:
    return _movie_details(_catalog_movie(request, movie_id))


@app.get("/api/v1/movies/{movie_id}/recommendations")
def recommend_for_movie(
    movie_id: int,
    request: Request,
    top_k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    _validate_top_k(top_k)
    _catalog_movie(request, movie_id)
    try:
        response = request.app.state.recommender.recommend_for_movie(movie_id, top_k=top_k)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _response_payload(response)


@app.get("/api/v1/users/{user_id}/recommendations")
def recommend_for_user(
    user_id: int,
    request: Request,
    top_k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    _validate_top_k(top_k)
    try:
        response = request.app.state.recommender.recommend_for_user(user_id, top_k=top_k)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if response.status == "unknown_user":
        raise HTTPException(
            status_code=404,
            detail={"status": response.status, "message": response.message},
        )
    return _response_payload(response)


@app.post("/api/v1/recommendations/from-favorites")
def recommend_from_favorites(
    body: FavoriteRecommendationsRequest,
    request: Request,
) -> dict[str, Any]:
    _validate_top_k(body.top_k)
    favorite_ids = list(dict.fromkeys(body.favorite_movie_ids))
    unknown = [movie_id for movie_id in favorite_ids if movie_id not in request.app.state.catalog_ids]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown movie_id: {unknown[0]}")
    try:
        response = request.app.state.recommender.recommend_for_new_user(
            favorite_ids, top_k=body.top_k
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _response_payload(response)
