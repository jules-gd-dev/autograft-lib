"""Configuration model for AutoGraft Entity Resolution."""

import os

from pydantic import BaseModel, Field


class AutoGraftConfig(BaseModel):
    """Configuration class allowing explicit API keys, model selection, and custom endpoints."""

    model: str = Field(
        default_factory=lambda: (
            os.getenv("AUTOGRAFT_LLM_MODEL")
            or os.getenv("AUTOGRRAFT_LLM_MODEL")
            or "groq/llama-3.3-70b-versatile"
        )
    )
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)
    match_threshold: float = Field(default=0.85)
    uncertainty_threshold: float = Field(default=0.75)
