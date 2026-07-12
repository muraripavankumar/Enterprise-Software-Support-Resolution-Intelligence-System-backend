from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class ConfigurationError(RuntimeError):
    """Raised when required environment configuration is missing."""


class Settings(BaseSettings):
    """Environment-backed settings for ingestion, indexing, and retrieval."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Enterprise Software Support & Resolution Intelligence System", alias="APP_NAME")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    app_env: str = Field(default="local", alias="APP_ENV")

    cors_allow_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,https://enterprise-software-support-resolut-blond.vercel.app",
        alias="CORS_ALLOW_ORIGINS",
    )
    cors_allow_methods: str = Field(default="GET,POST,OPTIONS", alias="CORS_ALLOW_METHODS")
    cors_allow_headers: str = Field(default="Authorization,Content-Type", alias="CORS_ALLOW_HEADERS")
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")

    log_format: str = Field(default="json", alias="LOG_FORMAT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    auth0_domain: Optional[str] = Field(default=None, alias="AUTH0_DOMAIN")
    auth0_audience: Optional[str] = Field(default=None, alias="AUTH0_AUDIENCE")
    api_audience: Optional[str] = Field(default=None, alias="API_AUDIENCE")
    auth0_issuer: Optional[str] = Field(default=None, alias="AUTH0_ISSUER")
    auth0_algorithms: str = Field(default="RS256", alias="AUTH0_ALGORITHMS")
    enable_auth0: bool = Field(default=True, alias="ENABLE_AUTH0")
    auth0_roles_claim: str = Field(
        default="https://stateful-agent.com/roles",
        alias="AUTH0_ROLES_CLAIM",
    )
    auth0_permissions_claim: str = Field(default="permissions", alias="AUTH0_PERMISSIONS_CLAIM")

    enable_langfuse: bool = Field(default=True, alias="ENABLE_LANGFUSE")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: Optional[str] = Field(default=None, alias="LANGFUSE_BASE_URL")
    langfuse_host: Optional[str] = Field(default=None, alias="LANGFUSE_HOST")
    enable_llm_judge: bool = Field(default=True, alias="ENABLE_LLM_JUDGE")
    llm_judge_model: Optional[str] = Field(default=None, alias="LLM_JUDGE_MODEL")

    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_llm_deployment: str = Field(alias="AZURE_OPENAI_LLM_DEPLOYMENT")
    azure_openai_chat_deployment: Optional[str] = Field(default=None, alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    azure_openai_api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")
    embedding_dims: int = Field(alias="EMBEDDING_DIMS")

    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    db_user: Optional[str] = Field(default=None, alias="DB_USER")
    db_password: Optional[str] = Field(default=None, alias="DB_PASSWORD")
    db_host: Optional[str] = Field(default=None, alias="DB_HOST")
    db_port: Optional[int] = Field(default=5432, alias="DB_PORT")
    db_name: Optional[str] = Field(default=None, alias="DB_NAME")
    db_sslmode: Optional[str] = Field(default=None, alias="DB_SSLMODE")
    db_channel_binding: Optional[str] = Field(default=None, alias="DB_CHANNEL_BINDING")
    db_table_name: str = Field(alias="DB_TABLE_NAME")
    retrieval_vector_table_name: Optional[str] = Field(default=None, alias="RETRIEVAL_VECTOR_TABLE_NAME")
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    retrieval_enable_full_text: bool = Field(default=True, alias="RETRIEVAL_ENABLE_FULL_TEXT")
    retrieval_vector_weight: float = Field(default=0.7, alias="RETRIEVAL_VECTOR_WEIGHT")
    retrieval_keyword_weight: float = Field(default=0.3, alias="RETRIEVAL_KEYWORD_WEIGHT")
    retrieval_enable_reranking: bool = Field(default=False, alias="RETRIEVAL_ENABLE_RERANKING")
    retrieval_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-2-v2",
        alias="RETRIEVAL_RERANKER_MODEL",
    )

    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    redis_service_name: Optional[str] = Field(default=None, alias="REDIS_SERVICE_NAME")
    redis_database_name: Optional[str] = Field(default=None, alias="REDIS_DATABASE_NAME")
    redis_store_id: Optional[str] = Field(default=None, alias="REDIS_STORE_ID")
    redis_api_url: Optional[str] = Field(default=None, alias="REDIS_API_URL")
    redis_api_key: Optional[str] = Field(default=None, alias="REDIS_API_KEY")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    rate_limit_default_limit: int = Field(default=30, alias="RATE_LIMIT_DEFAULT_LIMIT")
    rate_limit_support_agent_limit: int = Field(default=60, alias="RATE_LIMIT_SUPPORT_AGENT_LIMIT")
    rate_limit_manager_limit: int = Field(default=90, alias="RATE_LIMIT_MANAGER_LIMIT")
    rate_limit_admin_limit: int = Field(default=120, alias="RATE_LIMIT_ADMIN_LIMIT")
    require_redis_for_rate_limiting_non_local: bool = Field(
        default=True,
        alias="REQUIRE_REDIS_FOR_RATE_LIMITING_NON_LOCAL",
    )
    input_guardrail_min_query_length: int = Field(default=3, alias="INPUT_GUARDRAIL_MIN_QUERY_LENGTH")
    input_guardrail_max_query_length: int = Field(default=2000, alias="INPUT_GUARDRAIL_MAX_QUERY_LENGTH")
    guardrail_confidence_threshold: float = Field(default=0.70, alias="GUARDRAIL_CONFIDENCE_THRESHOLD")
    orchestration_max_agent_calls: int = Field(default=12, alias="ORCHESTRATION_MAX_AGENT_CALLS")
    orchestration_runtime_budget_ms: int = Field(default=60000, alias="ORCHESTRATION_RUNTIME_BUDGET_MS")
    orchestration_token_budget: int = Field(default=12000, alias="ORCHESTRATION_TOKEN_BUDGET")
    enable_semantic_cache: bool = Field(default=False, alias="ENABLE_SEMANTIC_CACHE")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    cache_similarity_threshold: float = Field(default=0.85, alias="CACHE_SIMILARITY_THRESHOLD")
    cache_rag_ttl_seconds: int = Field(default=600, alias="CACHE_RAG_TTL_SECONDS")
    cache_sql_ttl_seconds: int = Field(default=30, alias="CACHE_SQL_TTL_SECONDS")
    cache_safety_critical_ttl_seconds: int = Field(default=0, alias="CACHE_SAFETY_CRITICAL_TTL_SECONDS")
    semantic_cache_timeout_seconds: float = Field(default=2.0, alias="SEMANTIC_CACHE_TIMEOUT_SECONDS")

    enable_jira_mcp: bool = Field(default=False, alias="ENABLE_JIRA_MCP")
    jira_url: Optional[str] = Field(default=None, alias="JIRA_URL")
    jira_username: Optional[str] = Field(default=None, alias="JIRA_USERNAME")
    jira_api_token: Optional[str] = Field(default=None, alias="JIRA_API_TOKEN")
    jira_project_key: Optional[str] = Field(default=None, alias="JIRA_PROJECT_KEY")
    jira_security_project_key: Optional[str] = Field(default=None, alias="JIRA_SECURITY_PROJECT_KEY")
    jira_default_issue_type: str = Field(default="Task", alias="JIRA_DEFAULT_ISSUE_TYPE")
    jira_critical_issue_type: str = Field(default="Bug", alias="JIRA_CRITICAL_ISSUE_TYPE")
    jira_human_engineering_issue_type: str = Field(default="Task", alias="JIRA_HUMAN_ENGINEERING_ISSUE_TYPE")
    jira_critical_priority: str = Field(default="Highest", alias="JIRA_CRITICAL_PRIORITY")
    jira_high_priority: str = Field(default="High", alias="JIRA_HIGH_PRIORITY")
    jira_medium_priority: str = Field(default="Medium", alias="JIRA_MEDIUM_PRIORITY")
    jira_default_label: str = Field(default="eris", alias="JIRA_DEFAULT_LABEL")
    jira_component_name: Optional[str] = Field(default=None, alias="JIRA_COMPONENT_NAME")
    jira_triage_status: str = Field(default="Triage", alias="JIRA_TRIAGE_STATUS")
    jira_triage_transition_id: Optional[str] = Field(default=None, alias="JIRA_TRIAGE_TRANSITION_ID")
    jira_dedupe_window_days: int = Field(default=7, alias="JIRA_DEDUPE_WINDOW_DAYS")
    jira_mcp_command: str = Field(default="uvx", alias="JIRA_MCP_COMMAND")
    jira_mcp_args: str = Field(default="mcp-atlassian", alias="JIRA_MCP_ARGS")
    jira_mcp_timeout_seconds: float = Field(default=45.0, alias="JIRA_MCP_TIMEOUT_SECONDS")
    audit_log_base_url: Optional[str] = Field(default=None, alias="AUDIT_LOG_BASE_URL")

    enable_escalation_email_mcp: bool = Field(default=False, alias="ENABLE_ESCALATION_EMAIL_MCP")
    escalation_email_to: Optional[str] = Field(default=None, alias="ESCALATION_EMAIL_TO")
    escalation_email_from: Optional[str] = Field(default=None, alias="ESCALATION_EMAIL_FROM")
    smtp_host: Optional[str] = Field(default=None, alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, alias="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, alias="SMTP_USE_TLS")
    smtp_timeout_seconds: float = Field(default=15.0, alias="SMTP_TIMEOUT_SECONDS")

    llama_cloud_api_key: str = Field(alias="LLAMA_CLOUD_API_KEY")

    temp_upload_dir: Path = Field(alias="TEMP_UPLOAD_DIR")
    temp_image_dir: Path = Field(alias="TEMP_IMAGE_DIR")
    stored_image_dir: Path = Field(alias="STORED_IMAGE_DIR")
    max_upload_mb: int = Field(alias="MAX_UPLOAD_MB")
    document_parser_provider: str = Field(alias="DOCUMENT_PARSER_PROVIDER")
    use_semantic_splitter: bool = Field(alias="USE_SEMANTIC_SPLITTER")
    semantic_breakpoint_percentile: int = Field(alias="SEMANTIC_BREAKPOINT_PERCENTILE")
    enable_vector_indexing: bool = Field(alias="ENABLE_VECTOR_INDEXING")

    @model_validator(mode="after")
    def _normalize_settings(self) -> "Settings":
        self.app_env = (self.app_env or "local").strip().lower()
        if not self.app_env:
            self.app_env = "local"
        if not self.azure_openai_chat_deployment:
            self.azure_openai_chat_deployment = self.azure_openai_llm_deployment
        if self.auth0_domain:
            self.auth0_domain = self.auth0_domain.replace("https://", "").replace("http://", "").strip("/")
        if not self.auth0_audience and self.api_audience:
            self.auth0_audience = self.api_audience
        if self.auth0_domain and not self.auth0_issuer:
            self.auth0_issuer = f"https://{self.auth0_domain}/"
        if self.auth0_issuer:
            self.auth0_issuer = self.auth0_issuer.rstrip("/") + "/"
        if not self.auth0_algorithms.strip():
            self.auth0_algorithms = "RS256"
        if not self.langfuse_host and self.langfuse_base_url:
            self.langfuse_host = self.langfuse_base_url
        if not self.llm_judge_model:
            self.llm_judge_model = self.azure_openai_chat_deployment
        if not self.retrieval_vector_table_name:
            self.retrieval_vector_table_name = self.db_table_name
        if self._is_neon_database and not self.db_sslmode:
            self.db_sslmode = "require"
        if self._is_neon_database and not self.db_channel_binding:
            self.db_channel_binding = "require"
        self.log_format = (self.log_format or "text").strip().lower()
        if self.log_format not in {"text", "json"}:
            self.log_format = "text"
        self.log_level = (self.log_level or "INFO").strip().upper()
        self.temp_upload_dir = self._resolve_path(self.temp_upload_dir)
        self.temp_image_dir = self._resolve_path(self.temp_image_dir)
        self.stored_image_dir = self._resolve_path(self.stored_image_dir)
        self._validate_auth_mode()
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def database_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        self._ensure_database_parts()
        return self._append_database_query(
            f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def async_database_dsn(self) -> str:
        dsn = self.database_dsn
        if dsn.startswith("postgresql+asyncpg://"):
            return dsn
        if dsn.startswith("postgresql+psycopg://"):
            return dsn.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        return dsn

    @property
    def sqlalchemy_database_url(self) -> str:
        dsn = self.database_dsn
        if dsn.startswith("postgresql+psycopg://"):
            return dsn
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        return dsn

    @property
    def pgvector_connection_string(self) -> str:
        return self.sqlalchemy_database_url

    @property
    def pgvector_async_connection_string(self) -> str:
        return self.async_database_dsn

    @property
    def operational_tables(self) -> list[str]:
        return ["customers", "support_tickets", "incident_logs", "knowledge_article_usage"]

    @property
    def auth0_jwks_url(self) -> str:
        self.validate_for_auth()
        return f"https://{self.auth0_domain}/.well-known/jwks.json"

    @property
    def auth0_algorithm_list(self) -> list[str]:
        return [algorithm.strip() for algorithm in self.auth0_algorithms.split(",") if algorithm.strip()]

    @property
    def is_local_env(self) -> bool:
        return self.app_env in {"local", "dev", "development"}

    @property
    def cors_allowed_origins(self) -> list[str]:
        origins = [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]
        return origins

    @property
    def cors_allowed_methods(self) -> list[str]:
        methods = [item.strip().upper() for item in self.cors_allow_methods.split(",") if item.strip()]
        return methods or ["GET", "POST", "OPTIONS"]

    @property
    def cors_allowed_headers(self) -> list[str]:
        headers = [item.strip() for item in self.cors_allow_headers.split(",") if item.strip()]
        return headers or ["Authorization", "Content-Type"]

    def validate_for_auth(self) -> None:
        self._ensure_present(
            {
                "AUTH0_DOMAIN": self.auth0_domain,
                "AUTH0_AUDIENCE": self.auth0_audience,
                "AUTH0_ISSUER": self.auth0_issuer,
                "AUTH0_ALGORITHMS": self.auth0_algorithms,
            }.items()
        )

    @property
    def langfuse_configured(self) -> bool:
        return bool(
            self.enable_langfuse
            and self.langfuse_public_key
            and self.langfuse_secret_key
            and self.langfuse_host
        )

    @property
    def jira_configured(self) -> bool:
        return bool(
            self.enable_jira_mcp
            and self.jira_url
            and self.jira_username
            and self.jira_api_token
            and self.jira_project_key
        )

    @property
    def jira_mcp_arg_list(self) -> list[str]:
        return [arg for arg in self.jira_mcp_args.split() if arg]

    def validate_for_jira_mcp(self) -> None:
        if not self.enable_jira_mcp:
            return
        self._ensure_present(
            {
                "JIRA_URL": self.jira_url,
                "JIRA_USERNAME": self.jira_username,
                "JIRA_API_TOKEN": self.jira_api_token,
                "JIRA_PROJECT_KEY": self.jira_project_key,
                "JIRA_MCP_COMMAND": self.jira_mcp_command,
                "JIRA_MCP_ARGS": self.jira_mcp_args,
            }.items()
        )

    @property
    def escalation_email_recipients(self) -> list[str]:
        return [item.strip() for item in (self.escalation_email_to or "").split(",") if item.strip()]

    @staticmethod
    def _is_placeholder_setting(value: Optional[str]) -> bool:
        if value is None:
            return True
        text = value.strip().lower()
        return text in {"", "...", "example", "smtp.example.com"} or text.endswith("@example.com")

    @property
    def escalation_email_configured(self) -> bool:
        required_values_present = bool(
            self.enable_escalation_email_mcp
            and self.escalation_email_recipients
            and self.escalation_email_from
            and self.smtp_host
        )
        if not required_values_present:
            return False

        configured_values = [
            self.escalation_email_from,
            self.smtp_host,
            *self.escalation_email_recipients,
        ]
        optional_values = [self.smtp_username, self.smtp_password]
        configured_values.extend(value for value in optional_values if value)
        return not any(self._is_placeholder_setting(value) for value in configured_values)

    def validate_for_escalation_email_mcp(self) -> None:
        if not self.enable_escalation_email_mcp:
            return
        self._ensure_present(
            {
                "ESCALATION_EMAIL_TO": self.escalation_email_to,
                "ESCALATION_EMAIL_FROM": self.escalation_email_from,
                "SMTP_HOST": self.smtp_host,
                "SMTP_PORT": str(self.smtp_port or ""),
            }.items()
        )

    def validate_for_indexing(self) -> None:
        if not self.enable_vector_indexing:
            return
        if self.database_url:
            self._ensure_present(
                {
                    "DATABASE_URL": self.database_url,
                    "DB_TABLE_NAME": self.db_table_name,
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai_embedding_deployment,
                }.items()
            )
            return
        self._ensure_present(
            {
                "DB_USER": self.db_user,
                "DB_PASSWORD": self.db_password,
                "DB_HOST": self.db_host,
                "DB_PORT": str(self.db_port or ""),
                "DB_NAME": self.db_name,
                "DB_TABLE_NAME": self.db_table_name,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai_embedding_deployment,
            }.items()
        )

    def _resolve_path(self, value: Path) -> Path:
        if value.is_absolute():
            return value
        return Path(__file__).resolve().parents[2] / value

    @property
    def _is_neon_database(self) -> bool:
        if self.database_url:
            return "neon.tech" in self.database_url.lower()
        return bool(self.db_host and "neon.tech" in self.db_host.lower())

    def _ensure_database_parts(self) -> None:
        self._ensure_present(
            {
                "DB_USER": self.db_user,
                "DB_PASSWORD": self.db_password,
                "DB_HOST": self.db_host,
                "DB_PORT": str(self.db_port or ""),
                "DB_NAME": self.db_name,
            }.items()
        )

    def _append_database_query(self, dsn: str) -> str:
        split = urlsplit(dsn)
        query = dict(parse_qsl(split.query, keep_blank_values=True))
        if self.db_sslmode:
            query.setdefault("sslmode", self.db_sslmode)
        if self.db_channel_binding:
            query.setdefault("channel_binding", self.db_channel_binding)
        encoded_query = urlencode(query)
        return urlunsplit((split.scheme, split.netloc, split.path, encoded_query, split.fragment))

    def _ensure_present(self, entries: Iterable[tuple[str, Optional[str]]]) -> None:
        missing = [key for key, value in entries if not value]
        if missing:
            raise ConfigurationError("Missing required .env configuration: " + ", ".join(missing))

    def _validate_auth_mode(self) -> None:
        if self.enable_auth0:
            return
        if not self.is_local_env:
            raise ConfigurationError(
                "ENABLE_AUTH0=false is only allowed for local development. "
                "Set APP_ENV=local or enable Auth0 in non-local environments."
            )


try:
    settings = Settings()
except Exception as exc:  # pydantic raises validation errors before the app starts.
    raise ConfigurationError(str(exc)) from exc
