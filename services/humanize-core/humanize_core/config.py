from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_site_url: str | None = Field(default=None, alias="OPENROUTER_SITE_URL")
    openrouter_app_title: str = Field(default="Dadeum Humanize Core", alias="OPENROUTER_APP_TITLE")
    model_provider: str = Field(default="stub", alias="HUMANIZE_MODEL_PROVIDER")
    model_name: str = Field(default="stub", alias="HUMANIZE_MODEL_NAME")
    fast_model_name: str = Field(default="openai/gpt-5-mini", alias="HUMANIZE_FAST_MODEL_NAME")
    fast_fallback_model_name: str = Field(
        default="~anthropic/claude-haiku-latest",
        alias="HUMANIZE_FAST_FALLBACK_MODEL_NAME",
    )
    strict_detect_model_name: str = Field(
        default="openai/gpt-5.4-mini",
        alias="HUMANIZE_STRICT_DETECT_MODEL_NAME",
    )
    strict_rewrite_model_name: str = Field(
        default="~anthropic/claude-sonnet-latest",
        alias="HUMANIZE_STRICT_REWRITE_MODEL_NAME",
    )
    strict_audit_model_name: str = Field(
        default="~anthropic/claude-haiku-latest",
        alias="HUMANIZE_STRICT_AUDIT_MODEL_NAME",
    )
    strict_review_model_name: str = Field(
        default="openai/gpt-5.4-mini",
        alias="HUMANIZE_STRICT_REVIEW_MODEL_NAME",
    )
    strict_escalation_model_name: str = Field(
        default="~anthropic/claude-sonnet-latest",
        alias="HUMANIZE_STRICT_ESCALATION_MODEL_NAME",
    )
    core_api_key: str = Field(default="", alias="HUMANIZE_CORE_API_KEY")
    signing_secret: str = Field(default="", alias="HUMANIZE_CORE_SIGNING_SECRET")
    max_chars: int = Field(default=10_000, alias="HUMANIZE_MAX_CHARS")
    signature_tolerance_seconds: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
