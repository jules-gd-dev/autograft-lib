"""Configuration model for AutoGraft Entity Resolution."""

import os

from pydantic import BaseModel, Field


class AutoGraftConfig(BaseModel):
    """Configuration class for AutoGraft resolving settings."""

    # Provider and Authentication
    model: str = Field(
        default_factory=lambda: (
            os.getenv("AUTOGRAFT_LLM_MODEL")
            or os.getenv("AUTOGRRAFT_LLM_MODEL")
            or "groq/llama-3.3-70b-versatile"
        )
    )
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)

    # Resolution Thresholds
    match_threshold: float = Field(default=0.85)
    uncertainty_threshold: float = Field(default=0.75)

    # Graph Schema Config
    id_attr: str = Field(default="id")
    aliases_attr: str = Field(default="aliases")

    # String Matching
    matching_algorithm: str = Field(
        default="token_sort_ratio",
        description="fuzz matching algorithm (ratio, token_sort_ratio, token_set_ratio)",
    )
