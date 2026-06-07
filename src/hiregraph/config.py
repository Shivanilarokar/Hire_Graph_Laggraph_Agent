"""Central configuration. Every external knob comes from the environment.

Provider: **OpenAI only** (``gpt-4o-mini`` by default).

Mock semantics: set ``HIREGRAPH_USE_MOCKS=true`` to run with no third-party keys.
  - mocks ON   -> every external service is a deterministic mock.
  - mocks OFF  -> real service when its key is present; auto-mock when missing,
    so a keyless clone still runs end to end while a wired ``.env`` uses real APIs.
``USE_MOCKS`` is accepted as an alias for ``HIREGRAPH_USE_MOCKS``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into the process environment so libraries that read os.environ
# directly (LangSmith / langchain tracing) pick it up too. Existing shell
# variables win (override=False), so HIREGRAPH_USE_MOCKS=true still forces mocks.
load_dotenv()
if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"].strip())
    if os.getenv("LANGSMITH_PROJECT"):
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"].strip())


class Settings(BaseSettings):
    """Typed view over the process environment / ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Mock toggle (grader uses HIREGRAPH_USE_MOCKS) -----------------------
    use_mocks: bool = Field(
        default=False,
        validation_alias=AliasChoices("HIREGRAPH_USE_MOCKS", "USE_MOCKS"),
    )

    # --- LLM (OpenAI) --------------------------------------------------------
    openai_api_key: str = ""
    hire_model: str = "gpt-4o-mini"

    # --- Tools ---------------------------------------------------------------
    tavily_api_key: str = ""
    github_api_base: str = "https://api.github.com"
    github_token: str = ""

    # --- Email (Mailtrap sandbox) -------------------------------------------
    mailtrap_host: str = "sandbox.smtp.mailtrap.io"
    mailtrap_port: int = 2525
    mailtrap_username: str = Field(
        default="", validation_alias=AliasChoices("MAILTRAP_USER", "MAILTRAP_USERNAME")
    )
    mailtrap_password: str = Field(
        default="", validation_alias=AliasChoices("MAILTRAP_PASS", "MAILTRAP_PASSWORD")
    )
    email_from: str = "recruiting@hiregraph.example"

    # --- Checkpointer --------------------------------------------------------
    checkpointer: Literal["memory", "redis"] = "memory"
    redis_url: str = ""

    # --- Loop caps -----------------------------------------------------------
    email_max_revisions: int = 3

    # --- Derived helpers (not state, just convenience) -----------------------
    @property
    def llm_is_mock(self) -> bool:
        """True when the LLM should be faked (forced, or no OpenAI key)."""
        return self.use_mocks or not self.openai_api_key

    @property
    def tavily_is_mock(self) -> bool:
        return self.use_mocks or not self.tavily_api_key

    @property
    def email_is_mock(self) -> bool:
        return self.use_mocks or not self.mailtrap_password


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton so every node sees the same configuration."""
    return Settings()
