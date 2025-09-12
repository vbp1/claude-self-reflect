"""Claude Reflect MCP Server with Memory Decay."""

import math
import os
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime, timezone
import logging
import unicodedata
import re

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client import models
from qdrant_client.models import (
    FormulaQuery,
    SumExpression,
    MultExpression,
    ExpDecayExpression,
    DecayParamsExpression,
    DatetimeExpression,
    DatetimeKeyExpression,
)

from dotenv import load_dotenv
from fastembed import TextEmbedding

# Load environment variables from current directory
load_dotenv()

# Configure logging from environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", None)  # None means console only

# Set up handlers
handlers = []
if LOG_FILE:
    handlers.append(logging.FileHandler(LOG_FILE))
handlers.append(logging.StreamHandler())  # Always log to console

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

# Suppress debug logs from various libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
ENABLE_MEMORY_DECAY = os.getenv("ENABLE_MEMORY_DECAY", "false").lower() == "true"
DECAY_WEIGHT = float(os.getenv("DECAY_WEIGHT", "0.3"))
DECAY_SCALE_DAYS = float(os.getenv("DECAY_SCALE_DAYS", "90"))
USE_NATIVE_DECAY = os.getenv("USE_NATIVE_DECAY", "true").lower() == "true"
MCP_CLIENT_CWD = os.getenv("MCP_CLIENT_CWD", os.getcwd())
PROJECT_ID = os.getenv("PROJECT_ID", "").strip()

# Embedding configuration
# Required environment variables
if not os.getenv("EMBEDDING_MODEL"):
    raise ValueError("EMBEDDING_MODEL environment variable must be set")
if not os.getenv("VECTOR_SIZE"):
    raise ValueError("VECTOR_SIZE environment variable must be set")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))

# Get cache directory from environment
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/home/mcpuser/.cache/huggingface")
MODEL_CACHE_DAYS = int(os.getenv("MODEL_CACHE_DAYS", "7"))

# Determine project name that will be used for all searches
# Priority: PROJECT_ID (explicit) -> MCP_CLIENT_CWD-derived (backward compatibility)
if PROJECT_ID:
    DEFAULT_PROJECT_NAME = PROJECT_ID
else:
    # Convert MCP_CLIENT_CWD to watcher format: /path/to/project -> -path-to-project
    if not MCP_CLIENT_CWD:
        raise ValueError("Neither PROJECT_ID nor MCP_CLIENT_CWD is set - cannot determine project")
    # Convert slashes to dashes for project name (keeping backward compatibility)
    DEFAULT_PROJECT_NAME = re.sub(r"[\\/]+", "-", MCP_CLIENT_CWD)

# Main collection for all conversations
MAIN_COLLECTION = "claude_logs"


# Global state for async model initialization
local_embedding_model = None
model_ready = asyncio.Event()
model_initialization_task = None


async def initialize_embedding_model_async(model_name: str, cache_dir: str):
    """Initialize embedding model asynchronously with smart offline/online mode."""
    global local_embedding_model

    def is_model_fresh(model_name: str, cache_dir: str, max_age_days: int = 7) -> bool:
        """Check if the model is cached and fresh (not older than max_age_days)."""

        # Model cache path (FastEmbed format)
        model_cache_path = Path(cache_dir) / model_name.replace("/", "_")
        timestamp_file = model_cache_path / ".timestamp"

        # Check if model directory exists
        if not model_cache_path.exists():
            logger.debug(f"Model cache not found: {model_cache_path}")
            return False

        # Check timestamp file
        if not timestamp_file.exists():
            logger.debug(f"Timestamp file not found: {timestamp_file}")
            return False

        try:
            # Read timestamp
            with open(timestamp_file, "r") as f:
                timestamp = float(f.read().strip())

            # Check age
            age_seconds = time.time() - timestamp
            age_days = age_seconds / (24 * 60 * 60)

            logger.debug(f"Model age: {age_days:.1f} days (max: {max_age_days})")
            return age_days <= max_age_days

        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Error reading timestamp: {e}")
            return False

    # Check if model is fresh
    model_fresh = is_model_fresh(model_name, cache_dir, MODEL_CACHE_DAYS)

    if model_fresh:
        # Use offline mode
        logger.info(f"Using cached model in offline mode: {model_name}")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        # Use online mode
        logger.info(f"Model not found or stale, downloading: {model_name}")
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"

    try:
        # Run model initialization in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def init_model():
            # Update timestamp file after successful initialization
            model_cache_path = Path(cache_dir) / model_name.replace("/", "_")
            timestamp_file = model_cache_path / ".timestamp"

            # Ensure directory exists
            model_cache_path.mkdir(parents=True, exist_ok=True)

            model = TextEmbedding(model_name=model_name, cache_dir=model_cache_path)

            # Write current timestamp
            with open(timestamp_file, "w") as f:
                f.write(str(time.time()))

            logger.info(f"Model initialized successfully: {model_name}")
            return model

        # Initialize model in thread pool to avoid blocking
        local_embedding_model = await loop.run_in_executor(None, init_model)

        # Signal that model is ready
        model_ready.set()
        logger.info(f"Embedding model is ready for use: {model_name}")

        return local_embedding_model

    except ImportError:
        logger.error("FastEmbed not available. Install with: pip install fastembed")
        model_ready.set()  # Set even on error to unblock waiting tasks
        raise
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        model_ready.set()  # Set even on error to unblock waiting tasks
        raise


async def start_model_initialization():
    """Start async model initialization in background."""
    global model_initialization_task

    logger.info(f"Starting background initialization of embedding model: {EMBEDDING_MODEL} (vector size: {VECTOR_SIZE})")

    # Schedule the initialization task
    model_initialization_task = asyncio.create_task(initialize_embedding_model_async(EMBEDDING_MODEL, CACHE_DIR))

    # Don't wait for it - let it run in background
    logger.info("Model initialization started in background")

    return model_initialization_task


# Log effective configuration
logger.info("Effective configuration:")
logger.info(f"QDRANT_URL: {QDRANT_URL}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"VECTOR_SIZE: {VECTOR_SIZE}")
logger.info(f"ENABLE_MEMORY_DECAY: {ENABLE_MEMORY_DECAY}")
logger.info(f"USE_NATIVE_DECAY: {USE_NATIVE_DECAY}")
logger.info(f"DECAY_WEIGHT: {DECAY_WEIGHT}")
logger.info(f"DECAY_SCALE_DAYS: {DECAY_SCALE_DAYS}")
logger.info(f"TRANSFORMERS_CACHE: {CACHE_DIR}")
logger.info(f"MODEL_CACHE_DAYS: {MODEL_CACHE_DAYS}")
logger.info(f"PROJECT_ID: {PROJECT_ID if PROJECT_ID else '(not set)'}")
logger.info(f"MCP_CLIENT_CWD: {MCP_CLIENT_CWD}")
logger.info(f"Default project name for searches: {DEFAULT_PROJECT_NAME}")


class SearchResult(BaseModel):
    """A single search result."""

    id: str
    score: float
    timestamp: str
    role: str
    excerpt: str
    project_name: str
    conversation_id: Optional[str] = None
    field: Optional[str] = None  # Тип контента: text, code, stdout, error
    tags: Optional[List[str]] = None  # Теги для рефлексий


# Initialize FastMCP instance
mcp = FastMCP(
    name="claude-self-reflect",
    instructions="Search past conversations and store reflections with time-based memory decay",
)

# Create Qdrant client
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)


def normalize_text(text: str) -> str:
    """
    Normalize text for embedding using Unicode normalization.

    This ensures consistent representation of text regardless of:
    - Different Unicode representations (é vs e + ́)
    - Case differences
    - Whitespace variations
    """
    # Unicode normalization to NFC (Canonical Composition)
    # This ensures that é is represented as a single character, not e + combining accent
    normalized = unicodedata.normalize("NFC", text)

    # Convert to lowercase for case-insensitive matching
    normalized = normalized.casefold()

    # Normalize whitespace - replace multiple spaces/tabs/newlines with single space
    # and strip leading/trailing whitespace
    normalized = " ".join(normalized.split())

    return normalized


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding using local FastEmbed model."""
    # Wait for model to be ready (will return immediately if already ready)
    if not model_ready.is_set():
        logger.info("Waiting for embedding model to initialize...")
        await model_ready.wait()

    if not local_embedding_model:
        raise ValueError("Local embedding model failed to initialize")

    # Normalize text before embedding
    normalized_text = normalize_text(text)

    # Run in executor since fastembed is synchronous
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, lambda: list(local_embedding_model.embed([normalized_text])))
    return embeddings[0].tolist()


# Helper functions for search


def build_native_decay_query(_: list) -> FormulaQuery:
    """
    Итоговая формула:
      final_score = $score + DECAY_WEIGHT * exp_decay(timestamp → now, scale)
    """
    from datetime import datetime, timezone

    # Получаем текущее время в ISO-8601 формате
    current_time = datetime.now(timezone.utc).isoformat()

    decay = ExpDecayExpression(
        exp_decay=DecayParamsExpression(
            x=DatetimeKeyExpression(datetime_key="timestamp"),  # поле с датой в payload
            target=DatetimeExpression(datetime=current_time),  # текущее время в ISO-8601
            # scale в секундах (POSIX seconds), а НЕ миллисекундах:
            scale=int(DECAY_SCALE_DAYS * 24 * 60 * 60),
        )
    )

    formula = SumExpression(
        sum=[
            "$score",  # базовый скор в формуле передаётся как строка
            MultExpression(mult=[DECAY_WEIGHT, decay]),  # вес умножаем через MultExpression
        ]
    )

    return FormulaQuery(formula=formula)


def calculate_client_side_decay(point, min_score: float) -> Optional[float]:
    """
    Calculate client-side decay for a search result.

    Формула: final_score = (1 - DECAY_WEIGHT) * similarity + DECAY_WEIGHT * exp(-age_days/scale_days)
    где:
    - similarity: базовый score от Qdrant (косинусная близость)
    - DECAY_WEIGHT: вес временного фактора (0.3 = 30%)
    - age_days: возраст документа в днях
    - scale_days: масштаб затухания в днях (90)
    - exp(-age_days/90): от 1 (сегодня) до 0.37 (90 дней) до 0.14 (180 дней)

    Пример:
    - Документ от сегодня: final = 0.7 * similarity + 0.3 * 1.0
    - Документ 90 дней: final = 0.7 * similarity + 0.3 * 0.37
    - Документ 180 дней: final = 0.7 * similarity + 0.3 * 0.14
    """
    # Пропускаем документы с низким базовым score
    if point.score < min_score:
        return None

    timestamp_str = point.payload.get("timestamp")
    if not timestamp_str:
        # Если нет timestamp, возвращаем базовый score без decay
        return point.score

    try:
        # Парсим timestamp документа (Python 3.11+ поддерживает 'Z')
        point_time = datetime.fromisoformat(timestamp_str)
        current_time = datetime.now(timezone.utc)

        # Вычисляем возраст документа в днях
        age_seconds = max(0.0, (current_time - point_time).total_seconds())
        age_days = age_seconds / (24 * 60 * 60)

        # Экспоненциальное затухание: exp(-age/scale)
        # При age=0: decay=1.0, age=90: decay≈0.37, age=180: decay≈0.14
        decay_factor = math.exp(-age_days / DECAY_SCALE_DAYS)

        # Взвешенная комбинация: 70% similarity + 30% freshness
        final_score = (1 - DECAY_WEIGHT) * point.score + DECAY_WEIGHT * decay_factor

        return final_score

    except (ValueError, TypeError) as e:
        logger.debug(f"Error parsing timestamp: {e}")
        # При ошибке парсинга возвращаем базовый score
        return point.score


def convert_point_to_search_result(point, min_score: float) -> SearchResult:
    """
    Convert a Qdrant point to SearchResult.

    Преобразует результат поиска Qdrant в унифицированный формат SearchResult.
    Обрабатывает различные форматы timestamp и полей.
    """
    # Извлекаем timestamp - обрабатываем два возможных поля:
    # 'start_timestamp' (от watcher) или 'timestamp' (от reflection)
    timestamp_str = point.payload.get("start_timestamp", point.payload.get("timestamp", ""))

    # Нормализуем timestamp к UTC
    if timestamp_str:
        try:
            # Python 3.11+ поддерживает 'Z' в fromisoformat
            dt = datetime.fromisoformat(timestamp_str)
            # Убеждаемся, что время в UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            elif dt.tzinfo != timezone.utc:
                dt = dt.astimezone(timezone.utc)
            clean_timestamp = dt.isoformat()
        except (ValueError, TypeError):
            # При ошибке используем текущее время
            clean_timestamp = datetime.now(timezone.utc).isoformat()
    else:
        # Если нет timestamp, используем текущее время
        clean_timestamp = datetime.now(timezone.utc).isoformat()

    # Получаем score - может быть в point.score (native) или в payload (client-side decay)
    score = point.score if hasattr(point, "score") and point.score is not None else point.payload.get("score")

    # Возвращаем только если не ниже порога
    if score is None or score < min_score:
        return None
    else:
        return SearchResult(
            id=str(point.id),
            score=score,  # Используем найденный score
            timestamp=clean_timestamp,
            # Используем start_role если есть, иначе role, иначе 'unknown'
            role=point.payload.get("start_role", point.payload.get("role", "unknown")),
            excerpt=point.payload.get("text", ""),
            project_name=point.payload.get("project_name", point.payload.get("project", "unknown")),
            conversation_id=point.payload.get("conversation_id"),
            field=point.payload.get("field"),  # Тип контента: text, code, stdout, error
            tags=point.payload.get("tags"),  # Теги для рефлексий
        )


async def perform_qdrant_search(
    query_embedding: list,
    search_filter: Optional[dict],
    limit: int,
    min_score: float,
    should_use_decay: bool,
) -> List[SearchResult]:
    """
    Perform search in Qdrant with optional decay.

    Логика выбора метода поиска:
    1. Native decay (new API) - если доступен новый API Qdrant
    2. Native decay (legacy API) - если только старый API
    3. Client-side decay - если decay включён, но native не используется
    4. Standard search - если decay выключен
    """
    all_results = []

    try:
        logger.info(f"Searching in collection: {MAIN_COLLECTION}")
        logger.info(f"Search filter: {search_filter}")
        logger.info(f"Limit: {limit}, Min score: {min_score}")
        logger.info(f"Should use decay: {should_use_decay}, USE_NATIVE_DECAY: {USE_NATIVE_DECAY}")
        logger.info(f"Query embedding length: {len(query_embedding) if query_embedding else 0}")

        # Выбираем стратегию поиска на основе конфигурации
        if should_use_decay and USE_NATIVE_DECAY:
            # Вариант 1: Native decay
            logger.info(f"Using NATIVE Qdrant decay for {MAIN_COLLECTION}")

            # Шаг 1: Строим запрос
            query_obj = build_native_decay_query(query_embedding)
            logger.info("Native decay query object created")

            # Шаг 2: Выполняем поиск с decay на стороне Qdrant
            logger.info("Executing query_points with native decay...")
            query_result = await qdrant_client.query_points(
                collection_name=MAIN_COLLECTION,
                prefetch=models.Prefetch(
                    query=query_embedding,  # вектор для кандидатов
                    limit=limit * 3,  # кандидатов больше, чем итоговый limit
                    filter=search_filter,  # тот же фильтр на этапе кандидатов
                    # using="dense",                 # если используешь named vector
                ),
                query=query_obj,
                limit=limit,
                with_payload=True,
                query_filter=search_filter,
                with_vectors=False,
            )
            results = query_result.points
            logger.info(f"Native decay query returned {len(results)} points")

        elif should_use_decay:
            # Вариант 2: Client-side decay (когда native недоступен)
            logger.info(f"Using CLIENT-SIDE decay for {MAIN_COLLECTION}")

            # Шаг 1: Получаем больше кандидатов без фильтра по score
            # (умножаем limit на 3, так как часть отсеется после decay)
            logger.info("Executing query_points for client-side decay...")
            query_result = await qdrant_client.query_points(
                collection_name=MAIN_COLLECTION,
                query=query_embedding,
                limit=limit * 3,  # Берём с запасом для последующей фильтрации
                with_payload=True,
                query_filter=search_filter,  # Фильтр по проекту
                with_vectors=False,
            )
            results = query_result.points

            # Шаг 2: Применяем client-side decay к каждому результату
            for point in results:
                final_score = calculate_client_side_decay(point, min_score)
                point.payload.update({"score": final_score})

        else:
            # Вариант 3: Стандартный поиск без decay
            # Шаг 1: Обычный векторный поиск с порогом similarity
            logger.info(f"Using STANDARD search without decay for {MAIN_COLLECTION}")
            logger.info(f"Executing query_points with score_threshold={min_score}...")
            query_result = await qdrant_client.query_points(
                collection_name=MAIN_COLLECTION,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                score_threshold=min_score,  # Фильтруем по минимальному score
                query_filter=search_filter,  # Фильтр по проекту
                with_vectors=False,
            )
            results = query_result.points

        # Шаг финальный: Обрабатываем результаты от Qdrant
        logger.info(f"Processing {len(results)} results from Qdrant")

        # Шаг 2.2: Добавляем только если score выше порога
        for i, point in enumerate(results):
            logger.debug(f"Processing point {i + 1}/{len(results)}: id={point.id}")
            try:
                search_result = convert_point_to_search_result(point=point, min_score=min_score)
                if search_result is not None:
                    all_results.append(search_result)
                    logger.debug(f"Added result with score {search_result.score}")
                else:
                    logger.debug(f"Filtered out result below min_score {min_score}")
            except Exception as e:
                logger.error(f"Error converting point {point.id}: {str(e)}")
                continue

        # Шаг 3: Сортируем по убыванию финального score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Шаг 4: Ограничиваем количество результатов
        all_results = all_results[:limit]

        logger.info(f"Found {len(all_results)} results in {MAIN_COLLECTION} with score >= {min_score}")

        return all_results

    except Exception as e:
        logger.error(f"Error searching {MAIN_COLLECTION}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Search filter was: {search_filter}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def resolve_project_name(project: Optional[str], context: str = "") -> str:
    """
    Resolve project name from user input to internal format.

    Args:
        project: User-provided project name or None for default, "all" for all projects.
        context: Context string for logging (e.g., "reflect_on_past", "store_reflection").

    Returns:
        Resolved project name for internal use.
    """
    if project is None:
        # Use default project name determined at startup
        project_name = DEFAULT_PROJECT_NAME
        if context:
            logger.info(f"{context}: No project specified, using default: {project_name}")
    elif project == "all":
        project_name = "all"
        if context:
            logger.info(f"{context}: Searching across all projects")
    else:
        # Convert user-provided project to watcher format
        project_name = project.replace("/", "-") if "/" in project else project
        if context:
            logger.info(f"{context}: Using specified project: {project_name}")

    return project_name


def build_search_filter(
    project_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    additional_conditions: Optional[List[dict]] = None,
) -> Optional[dict]:
    """
    Build a Qdrant search filter based on provided criteria.

    Args:
        project_name: Project name to filter by. Use "all" to skip project filtering.
        tags: List of tags to filter by (AND condition - must have ALL tags).
        additional_conditions: Additional filter conditions to include.

    Returns:
        Qdrant filter dict or None if no filters needed.

    Example:
        >>> build_search_filter(project_name="my-project", tags=["bug", "feature"])
        {'must': [{'key': 'project_name', 'match': {'value': 'my-project'}},
                  {'key': 'tags', 'match': {'value': 'bug'}},
                  {'key': 'tags', 'match': {'value': 'feature'}}]}
    """
    must_conditions = []

    # Add project filter if not searching all projects
    if project_name and project_name != "all":
        logger.debug(f"build_search_filter: Adding project filter for '{project_name}'")
        must_conditions.append({"key": "project_name", "match": {"value": project_name}})

    # Add tags filter if tags are specified (AND condition - must have ALL tags)
    if tags:
        logger.debug(f"build_search_filter: Adding tags filter for {tags} (AND condition)")
        # Each tag gets its own condition to ensure ALL tags must be present
        for tag in tags:
            must_conditions.append({"key": "tags", "match": {"value": tag}})

    # Add any additional conditions
    if additional_conditions:
        logger.debug(f"build_search_filter: Adding {len(additional_conditions)} additional conditions")
        must_conditions.extend(additional_conditions)

    # Build the final filter if we have any conditions
    if must_conditions:
        search_filter = {"must": must_conditions}
        logger.debug(f"build_search_filter: Final filter with {len(must_conditions)} conditions")
        return search_filter

    logger.debug("build_search_filter: No filters applied")
    return None


# Register tools
@mcp.tool()
async def reflect_on_past(
    ctx: Context,
    query: str = Field(description="The search query to find semantically similar conversations"),
    limit: int = Field(default=5, description="Maximum number of results to return"),
    min_score: float = Field(default=0.7, description="Minimum similarity score (0-1)"),
    use_decay: Union[int, str] = Field(
        default=-1,
        description="Apply time-based decay: 1=enable, 0=disable, -1=use environment default (accepts int or str)",
    ),
    project: Optional[str] = Field(
        default=None,
        description="Optional. Omit to search current project (auto-detected). For specific project use path with dashes instead of slashes (e.g., '-home-user-project'). Use 'all' to search all projects.",
    ),
    tags: Optional[Union[List[str], str]] = Field(
        default=None,
        description="Optional tags to filter results. Only results with ALL specified tags will be returned. Can be a list of strings or comma-separated string. If not specified, all results are returned regardless of tags.",
    ),
) -> str:
    """Search through previously stored project information, insights, and reflections using semantic search. Supports optional time-based relevance decay and tag filtering for precise retrieval."""

    # Log all incoming parameters
    logger.debug("reflect_on_past called with parameters:")
    logger.debug(f"  query: {query}")
    logger.debug(f"  limit: {limit}")
    logger.debug(f"  min_score: {min_score}")
    logger.debug(f"  use_decay: {use_decay}")
    logger.debug(f"  project: {project}")
    logger.debug(f"  tags: {tags} (type: {type(tags)})")

    # Parse tags if they come as a string
    if tags is not None:
        if isinstance(tags, str):
            # Handle comma-separated string
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif not isinstance(tags, list):
            # Try to handle other formats
            tags = [str(tags)]

    # Normalize use_decay to integer
    if isinstance(use_decay, str):
        try:
            use_decay = int(use_decay)
        except ValueError:
            raise ValueError("use_decay must be '1', '0', or '-1'")

    # Parse decay parameter using integer approach
    should_use_decay = (
        True if use_decay == 1 else False if use_decay == 0 else ENABLE_MEMORY_DECAY  # -1 or any other value
    )

    # Determine project scope
    project_name = resolve_project_name(project, "reflect_on_past")

    try:
        # Generate embedding
        query_embedding = await generate_embedding(query)

        # Check if main collection exists
        try:
            await qdrant_client.get_collection(MAIN_COLLECTION)
        except Exception:
            return f"Collection '{MAIN_COLLECTION}' not found."

        # Build search filter using helper function
        logger.info(f"reflect_on_past: Building filter for project_name='{project_name}', tags={tags}")
        search_filter = build_search_filter(project_name=project_name, tags=tags)

        if search_filter:
            logger.info(f"reflect_on_past: Using filter with {len(search_filter.get('must', []))} conditions")
            logger.info(f"reflect_on_past: Full filter: {search_filter}")
        else:
            logger.info("reflect_on_past: No filters applied - searching all content")

        # Perform search using helper function
        logger.info("reflect_on_past: Calling perform_qdrant_search...")
        try:
            all_results = await perform_qdrant_search(
                query_embedding=query_embedding,
                search_filter=search_filter,
                limit=limit,
                min_score=min_score,
                should_use_decay=should_use_decay,
            )

        except Exception as e:
            logger.error(f"Error searching {MAIN_COLLECTION}: {str(e)}")
            return f"Error searching conversations: {str(e)}"

        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Total results before limiting: {len(all_results)}")
        logger.debug(f"Top 3 results: {[(r.score, r.project_name) for r in all_results[:3]]}")
        all_results = all_results[:limit]

        if not all_results:
            return f"No conversations found matching '{query}'. Try different keywords or check if conversations have been imported."

        # Format results
        result_text = f"Found {len(all_results)} relevant conversation(s) for '{query}':\n\n"
        for i, result in enumerate(all_results):
            result_text += f"**Result {i + 1}** (Score: {result.score:.3f})\n"
            # Python 3.11+ поддерживает 'Z' в fromisoformat
            result_text += f"Time: {datetime.fromisoformat(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
            result_text += f"Project: {result.project_name}\n"
            result_text += f"Role: {result.role}\n"
            if result.field:
                result_text += f"Type: {result.field}\n"
            if result.tags:
                result_text += f"Tags: {', '.join(result.tags)}\n"
            result_text += f"Excerpt: {result.excerpt}\n"
            result_text += "---\n\n"

        return result_text

    except Exception as e:
        await ctx.error(f"Search failed: {str(e)}")
        return f"Failed to search conversations: {str(e)}"


@mcp.tool()
async def store_reflection(
    ctx: Context,
    content: str = Field(description="The information, insight or reflection to store"),
    tags: List[str] = Field(default=[], description="Tags to categorize this information, insight or reflection"),
    project: Optional[str] = Field(
        default=None,
        description="Target project for the stored information, insight or reflection. If not provided, uses current project.",
    ),
) -> str:
    """Store an important information about this project, insight or reflection that can be useful for future tasks to a memory"""

    try:
        # Determine project scope
        project_name = resolve_project_name(project, "store_reflection")

        # Ensure main collection exists
        try:
            await qdrant_client.get_collection(MAIN_COLLECTION)
        except Exception:
            # Create collection if it doesn't exist
            await qdrant_client.create_collection(
                collection_name=MAIN_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.debug(f"Created collection: {MAIN_COLLECTION}")

        # Generate embedding for the reflection
        embedding = await generate_embedding(content)

        # Create point with metadata
        point_id = datetime.now(timezone.utc).timestamp()
        point = PointStruct(
            id=int(point_id),
            vector=embedding,
            payload={
                "text": content,
                "tags": tags,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "reflection",
                "role": "user_reflection",
                "start_role": "user_reflection",  # For compatibility with search
                "project_name": project_name,  # Use consistent field name
                "conversation_id": f"reflection_{int(point_id)}",
                "field": "text",
                "source": "reflection",
            },
        )

        # Store in Qdrant
        await qdrant_client.upsert(collection_name=MAIN_COLLECTION, points=[point])

        tags_str = ", ".join(tags) if tags else "none"
        return f"Reflection stored successfully in project '{project_name}' with tags: {tags_str}"

    except Exception as e:
        await ctx.error(f"Store failed: {str(e)}")
        return f"Failed to store reflection: {str(e)}"


# Debug output
logger.info(f"FastMCP server created with name: {mcp.name}")
logger.info(f"Log level: {LOG_LEVEL}")
if LOG_FILE:
    logger.info(f"Log file: {LOG_FILE}")
else:
    logger.info("Logging to console only")
logger.info("Server starting...")


async def create_server() -> FastMCP:
    """Factory function to create and initialize the MCP server."""
    # Start model initialization in background
    await start_model_initialization()

    # Return the configured MCP server
    return mcp


# Export both mcp and create_server for different use cases
__all__ = ["mcp", "create_server"]


if __name__ == "__main__":
    logger.info(f"MCP server started with args: {sys.argv}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")

    # Run the server using factory function for async initialization
    import asyncio

    server = asyncio.run(create_server())
    server.run()
