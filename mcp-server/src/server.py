"""Claude Reflect MCP Server with Memory Decay."""

import os
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime, timezone
import numpy as np
import logging
import unicodedata

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

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

# Try to import newer Qdrant API for native decay
try:
    from qdrant_client.models import (
        Query,
        Formula,
        Expression,
        MultExpression,
        DecayParamsExpression,
    )

    NATIVE_DECAY_AVAILABLE = True
    logger.info("Using native decay")
except ImportError:
    # Fall back to older API
    from qdrant_client.models import (
        FormulaQuery,
        DecayParamsExpression,
        SumExpression,
        DatetimeExpression,
        DatetimeKeyExpression,
    )

    NATIVE_DECAY_AVAILABLE = False
    logger.info("Using older decay API")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
ENABLE_MEMORY_DECAY = os.getenv("ENABLE_MEMORY_DECAY", "false").lower() == "true"
DECAY_WEIGHT = float(os.getenv("DECAY_WEIGHT", "0.3"))
DECAY_SCALE_DAYS = float(os.getenv("DECAY_SCALE_DAYS", "90"))
USE_NATIVE_DECAY = os.getenv("USE_NATIVE_DECAY", "false").lower() == "true"
MCP_CLIENT_CWD = os.getenv("MCP_CLIENT_CWD", os.getcwd())

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
# Convert MCP_CLIENT_CWD to watcher format: /path/to/project -> -path-to-project
if not MCP_CLIENT_CWD:
    raise ValueError("MCP_CLIENT_CWD is not set - cannot determine project")
DEFAULT_PROJECT_NAME = MCP_CLIENT_CWD.replace("/", "-")

# Main collection for all conversations
MAIN_COLLECTION = "claude_logs"


def initialize_embedding_model(model_name: str, cache_dir: str):
    """Initialize embedding model with smart offline/online mode."""

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

    # Get max age from environment
    max_age_days = int(os.getenv("MODEL_CACHE_DAYS", "7"))

    # Check if model is fresh
    model_fresh = is_model_fresh(model_name, cache_dir, max_age_days)

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

    except ImportError:
        logger.error("FastEmbed not available. Install with: pip install fastembed")
        raise


# Initialize local embedding model with smart caching
logger.info(
    f"Initializing embedding model: {EMBEDDING_MODEL} (vector size: {VECTOR_SIZE})"
)
local_embedding_model = initialize_embedding_model(EMBEDDING_MODEL, CACHE_DIR)

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
    if not local_embedding_model:
        raise ValueError("Local embedding model not initialized")

    # Normalize text before embedding
    normalized_text = normalize_text(text)

    # Run in executor since fastembed is synchronous
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, lambda: list(local_embedding_model.embed([normalized_text]))
    )
    return embeddings[0].tolist()


# Helper functions for search


def build_native_decay_query_new_api(query_embedding: list) -> Query:
    """
    Build native decay query using newer Qdrant API.

    Formula: final_score = similarity_score + (DECAY_WEIGHT * exp(-age/scale))
    где:
    - similarity_score: косинусная близость между векторами (0-1)
    - DECAY_WEIGHT: вес временного фактора (default 0.3)
    - age: возраст документа в миллисекундах от текущего момента
    - scale: масштаб затухания в миллисекундах (90 дней * 24 * 60 * 60 * 1000)
    - exp(-age/scale): экспоненциальное затухание от 1 (новый) до 0 (старый)
    """

    # Настройка экспоненциального затухания
    exp_decay = DecayParamsExpression(
        # Поле с timestamp документа для вычисления возраста
        x=Expression(datetime_key="timestamp"),
        # Целевое время - текущий момент (вычисляется на сервере Qdrant)
        target=Expression(datetime="now"),
        # Масштаб затухания: через 90 дней score уменьшится в e раз (~2.7 раза)
        scale=DECAY_SCALE_DAYS * 24 * 60 * 60 * 1000,  # В миллисекундах
        # Точка, где decay_factor = 0.5 (не используется при exp_decay)
        midpoint=0.5,
    )

    # Множители для decay компонента: DECAY_WEIGHT * exp_decay
    mult = [
        # Вес decay в финальном score (0.3 = 30% веса)
        Expression(constant=DECAY_WEIGHT),
        # Результат экспоненциального затухания (от 1 до 0)
        Expression(exp_decay=exp_decay),
    ]

    # Финальная формула: similarity + (weight * decay)
    sum = [
        # Базовый similarity score от Qdrant (косинусная близость)
        Expression(variable="score"),
        # Добавка от временного decay (может увеличить score для свежих документов)
        Expression(mult=MultExpression(mult=mult)),
    ]

    return Query(nearest=query_embedding, formula=Formula(sum=sum))


def build_native_decay_query_legacy_api(query_embedding: list) -> FormulaQuery:
    """
    Build native decay query using legacy Qdrant API.

    Та же формула, но с другим синтаксисом API:
    final_score = similarity_score + (DECAY_WEIGHT * exp(-age/scale))
    """
    # Шаг 1: Настраиваем функцию экспоненциального затухания
    exp_decay = DatetimeExpression(
        # Поле timestamp для вычисления возраста документа
        x=DatetimeKeyExpression(key="timestamp"),
        # Текущее время (вычисляется на сервере Qdrant)
        target=DatetimeExpression(datetime="now"),
        # Масштаб в миллисекундах (90 дней * 24 часа * 60 минут * 60 секунд * 1000 мс)
        scale=DECAY_SCALE_DAYS * 24 * 60 * 60 * 1000,
        # Не используется для exp_decay
        midpoint=0.5,
    )

    # Шаг 2: Создаём взвешенный decay компонент
    mult = DecayParamsExpression(
        # Вес decay компонента (0.3 = 30% влияния на финальный score)
        weight=DECAY_WEIGHT,
        # Функция затухания от шага 1
        exp_decay=exp_decay,
    )

    # Шаг 3: Собираем финальную формулу как сумму компонентов
    sum = [
        # Базовый similarity score от Qdrant (косинусная близость)
        Expression(variable="score"),
        # Добавка за свежесть документа (может увеличить score для новых документов)
        Expression(mult=mult),
    ]

    # Шаг 4: Создаём запрос с формулой
    return FormulaQuery(nearest=query_embedding, formula=SumExpression(sum=sum))


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
        age_seconds = (current_time - point_time).total_seconds()
        age_days = age_seconds / (24 * 60 * 60)

        # Экспоненциальное затухание: exp(-age/scale)
        # При age=0: decay=1.0, age=90: decay≈0.37, age=180: decay≈0.14
        decay_factor = np.exp(-age_days / DECAY_SCALE_DAYS)

        # Взвешенная комбинация: 70% similarity + 30% freshness
        final_score = (1 - DECAY_WEIGHT) * point.score + DECAY_WEIGHT * decay_factor

        # Возвращаем только если выше порога
        return final_score if final_score >= min_score else None
    except (ValueError, TypeError) as e:
        logger.debug(f"Error parsing timestamp: {e}")
        # При ошибке парсинга возвращаем базовый score
        return point.score


def convert_point_to_search_result(
    point, score: float, project_name: str
) -> SearchResult:
    """
    Convert a Qdrant point to SearchResult.

    Преобразует результат поиска Qdrant в унифицированный формат SearchResult.
    Обрабатывает различные форматы timestamp и полей.
    """
    # Извлекаем timestamp - обрабатываем два возможных поля:
    # 'start_timestamp' (от watcher) или 'timestamp' (от reflection)
    timestamp_str = point.payload.get(
        "start_timestamp", point.payload.get("timestamp", "")
    )

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

    return SearchResult(
        id=str(point.id),
        score=score,  # Может быть с decay или без
        timestamp=clean_timestamp,
        # Используем start_role если есть, иначе role, иначе 'unknown'
        role=point.payload.get("start_role", point.payload.get("role", "unknown")),
        excerpt=point.payload.get("text", ""),
        project_name=project_name,
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
        logger.debug(f"Searching in collection: {MAIN_COLLECTION}")

        # Выбираем стратегию поиска на основе конфигурации
        if should_use_decay and USE_NATIVE_DECAY and NATIVE_DECAY_AVAILABLE:
            # Вариант 1: Native decay с новым API (самый эффективный)
            logger.debug(f"Using NATIVE Qdrant decay (new API) for {MAIN_COLLECTION}")

            # Шаг 1: Строим запрос с формулой decay
            query_obj = build_native_decay_query_new_api(query_embedding)

            # Шаг 2: Выполняем поиск с decay на стороне Qdrant
            results = await qdrant_client.query_points(
                collection_name=MAIN_COLLECTION,
                query=query_obj,
                limit=limit,
                with_payload=True,  # Включаем payload для получения метаданных
                query_filter=search_filter,  # Фильтр по проекту, если задан
            )

        elif should_use_decay and USE_NATIVE_DECAY and not NATIVE_DECAY_AVAILABLE:
            # Вариант 2: Native decay со старым API (совместимость)
            logger.debug(
                f"Using NATIVE Qdrant decay (legacy API) for {MAIN_COLLECTION}"
            )

            # Шаг 1: Строим запрос для старого API
            query_obj = build_native_decay_query_legacy_api(query_embedding)

            # Шаг 2: Выполняем поиск с decay на стороне Qdrant
            results = await qdrant_client.query_points(
                collection_name=MAIN_COLLECTION,
                query=query_obj,
                limit=limit,
                with_payload=True,
                query_filter=search_filter,
            )

        elif should_use_decay:
            # Вариант 3: Client-side decay (когда native недоступен)
            logger.debug(f"Using CLIENT-SIDE decay for {MAIN_COLLECTION}")

            # Шаг 1: Получаем больше кандидатов без фильтра по score
            # (умножаем limit на 3, так как часть отсеется после decay)
            results = await qdrant_client.search(
                collection_name=MAIN_COLLECTION,
                query_vector=query_embedding,
                limit=limit * 3,  # Берём с запасом для последующей фильтрации
                with_payload=True,
                query_filter=search_filter,  # Фильтр по проекту
            )

            # Шаг 2: Применяем client-side decay к каждому результату
            for point in results:
                # Шаг 2.1: Вычисляем score с учётом возраста документа
                final_score = calculate_client_side_decay(point, min_score)

                # Шаг 2.2: Добавляем только если score выше порога после decay
                if final_score is not None:
                    point_project = point.payload.get("project_name", "unknown")
                    all_results.append(
                        convert_point_to_search_result(
                            point, final_score, point_project
                        )
                    )

            # Шаг 3: Сортируем по убыванию финального score
            all_results.sort(key=lambda x: x.score, reverse=True)

            # Шаг 4: Ограничиваем количество результатов
            all_results = all_results[:limit]

            # Возвращаем раньше, так как обработка завершена
            return all_results

        else:
            # Вариант 4: Стандартный поиск без decay
            # Шаг 1: Обычный векторный поиск с порогом similarity
            results = await qdrant_client.search(
                collection_name=MAIN_COLLECTION,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                score_threshold=min_score,  # Фильтруем по минимальному score
                query_filter=search_filter,  # Фильтр по проекту
            )
            logger.debug(
                f"Found {len(results)} results in {MAIN_COLLECTION} with score >= {min_score}"
            )

        # Шаг финальный: Обрабатываем результаты от Qdrant
        for point in results:
            if should_use_decay and USE_NATIVE_DECAY:
                logger.debug(f"Point score with decay: {point.score}")
            point_project = point.payload.get("project_name", "unknown")
            all_results.append(
                convert_point_to_search_result(point, point.score, point_project)
            )

    except Exception as e:
        logger.error(f"Error searching {MAIN_COLLECTION}: {str(e)}")
        raise

    return all_results


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
        tags: List of tags to filter by (OR condition - matches any tag).
        additional_conditions: Additional filter conditions to include.
    
    Returns:
        Qdrant filter dict or None if no filters needed.
    
    Example:
        >>> build_search_filter(project_name="my-project", tags=["bug", "feature"])
        {'must': [{'key': 'project_name', 'match': {'value': 'my-project'}},
                  {'key': 'tags', 'match': {'any': ['bug', 'feature']}}]}
    """
    must_conditions = []
    
    # Add project filter if not searching all projects
    if project_name and project_name != "all":
        logger.debug(f"build_search_filter: Adding project filter for '{project_name}'")
        must_conditions.append({"key": "project_name", "match": {"value": project_name}})
    
    # Add tags filter if tags are specified
    if tags:
        logger.debug(f"build_search_filter: Adding tags filter for {tags}")
        must_conditions.append({"key": "tags", "match": {"any": tags}})
    
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
    query: str = Field(
        description="The search query to find semantically similar conversations"
    ),
    limit: int = Field(default=5, description="Maximum number of results to return"),
    min_score: float = Field(default=0.7, description="Minimum similarity score (0-1)"),
    use_decay: Union[int, str] = Field(
        default=-1,
        description="Apply time-based decay: 1=enable, 0=disable, -1=use environment default (accepts int or str)",
    ),
    project: Optional[str] = Field(
        default=None,
        description="Search specific project only. If not provided, searches current project based on working directory. Use 'all' to search across all projects.",
    ),
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional tags to filter results. Only results with at least one of these tags will be returned. If not specified, all results are returned regardless of tags.",
    ),
) -> str:
    """Search for relevant past conversations using semantic search with optional time decay and tag filtering."""

    # Log all incoming parameters
    logger.debug("reflect_on_past called with parameters:")
    logger.debug(f"  query: {query}")
    logger.debug(f"  limit: {limit}")
    logger.debug(f"  min_score: {min_score}")
    logger.debug(f"  use_decay: {use_decay}")
    logger.debug(f"  project: {project}")
    logger.debug(f"  tags: {tags}")

    # Normalize use_decay to integer
    if isinstance(use_decay, str):
        try:
            use_decay = int(use_decay)
        except ValueError:
            raise ValueError("use_decay must be '1', '0', or '-1'")

    # Parse decay parameter using integer approach
    should_use_decay = (
        True
        if use_decay == 1
        else False
        if use_decay == 0
        else ENABLE_MEMORY_DECAY  # -1 or any other value
    )

    # Determine project scope
    project_name = resolve_project_name(project, "reflect_on_past")

    try:
        # Generate embedding
        query_embedding = await generate_embedding(query)

        # Check if main collection exists
        collections = await qdrant_client.get_collections()
        if not any(c.name == MAIN_COLLECTION for c in collections.collections):
            return f"Collection '{MAIN_COLLECTION}' not found."

        # Build search filter using helper function
        search_filter = build_search_filter(
            project_name=project_name,
            tags=tags
        )
        
        if search_filter:
            logger.info(f"reflect_on_past: Using filter with {len(search_filter.get('must', []))} conditions")
        else:
            logger.info("reflect_on_past: No filters applied - searching all content")

        # Perform search using helper function
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
        logger.debug(
            f"Top 3 results: {[(r.score, r.project_name) for r in all_results[:3]]}"
        )
        all_results = all_results[:limit]

        if not all_results:
            return f"No conversations found matching '{query}'. Try different keywords or check if conversations have been imported."

        # Format results
        result_text = (
            f"Found {len(all_results)} relevant conversation(s) for '{query}':\n\n"
        )
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
    content: str = Field(description="The insight or reflection to store"),
    tags: List[str] = Field(
        default=[], description="Tags to categorize this reflection"
    ),
    project: Optional[str] = Field(
        default=None,
        description="Target project for the reflection. If not provided, uses current project.",
    ),
) -> str:
    """Store an important insight or reflection for future reference."""

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

if __name__ == "__main__":
    logger.info(f"MCP server started with args: {sys.argv}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
