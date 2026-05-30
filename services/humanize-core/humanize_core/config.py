from functools import lru_cache

from pydantic import AliasChoices, Field
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
    rewrite_model_name: str = Field(
        default="openai/gpt-5-mini",
        validation_alias=AliasChoices("HUMANIZE_REWRITE_MODEL_NAME", "HUMANIZE_FAST_MODEL_NAME"),
    )
    rewrite_fallback_model_name: str = Field(
        default="~anthropic/claude-haiku-latest",
        validation_alias=AliasChoices("HUMANIZE_REWRITE_FALLBACK_MODEL_NAME", "HUMANIZE_FAST_FALLBACK_MODEL_NAME"),
    )
    strict_audit_model_name: str = Field(
        default="~anthropic/claude-haiku-latest",
        alias="HUMANIZE_STRICT_AUDIT_MODEL_NAME",
    )
    strict_review_model_name: str = Field(
        default="openai/gpt-5.4-mini",
        alias="HUMANIZE_STRICT_REVIEW_MODEL_NAME",
    )
    core_api_key: str = Field(default="", alias="HUMANIZE_CORE_API_KEY")
    signing_secret: str = Field(default="", alias="HUMANIZE_CORE_SIGNING_SECRET")
    max_chars: int = Field(default=5_000, alias="HUMANIZE_MAX_CHARS")
    job_store_path: str = Field(default="humanize_jobs.sqlite3", alias="HUMANIZE_JOB_STORE_PATH")
    job_encryption_key: str | None = Field(default=None, alias="HUMANIZE_JOB_ENCRYPTION_KEY")
    job_worker_enabled: bool = Field(default=True, alias="HUMANIZE_JOB_WORKER_ENABLED")
    job_poll_interval_seconds: float = Field(default=1.0, alias="HUMANIZE_JOB_POLL_INTERVAL_SECONDS")
    job_lock_seconds: int = Field(default=600, alias="HUMANIZE_JOB_LOCK_SECONDS")
    job_retention_seconds: int = Field(default=86_400, alias="HUMANIZE_JOB_RETENTION_SECONDS")
    job_max_attempts: int = Field(default=2, alias="HUMANIZE_JOB_MAX_ATTEMPTS")
    debug_log_enabled: bool = Field(default=True, alias="HUMANIZE_DEBUG_LOG_ENABLED")
    debug_log_dir: str = Field(default="~/.dadeum/humanize-core/logs", alias="HUMANIZE_DEBUG_LOG_DIR")
    debug_log_include_plaintext: bool = Field(default=False, alias="HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT")
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
