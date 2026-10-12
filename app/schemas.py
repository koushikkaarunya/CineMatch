"""Small request schemas for the public API."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class FavoriteRecommendationsRequest(BaseModel):
    """Favorite catalog IDs supplied by a new user for content recommendations."""

    model_config = ConfigDict(extra="forbid")

    favorite_movie_ids: list[StrictInt] = Field(default_factory=list)
    top_k: StrictInt = Field(default=10, ge=1, le=50)
