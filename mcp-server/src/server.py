"""Claude Reflect MCP Server with Memory Decay."""

import os
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Optional, List, Dict, Union
from datetime import datetime, timezone
import numpy as np
import hashlib
import logging

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance
)

# Try to import newer Qdrant API for native decay
try:
    from qdrant_client.models import (
        Query, Formula, Expression, MultExpression,
        DecayParamsExpression
    )
    NATIVE_DECAY_AVAILABLE = True
except ImportError:
    # Fall back to older API
    from qdrant_client.models import (
        FormulaQuery, DecayParamsExpression, SumExpression,
        DatetimeExpression, DatetimeKeyExpression
    )
    NATIVE_DECAY_AVAILABLE = False
from dotenv import load_dotenv
from fastembed import TextEmbedding

# Load environment variables first to get logging config
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Configure logging from environment variables
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('LOG_FILE', None)  # None means console only

# Set up handlers
handlers = []
if LOG_FILE:
    handlers.append(logging.FileHandler(LOG_FILE))
handlers.append(logging.StreamHandler())  # Always log to console

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Suppress debug logs from various libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
ENABLE_MEMORY_DECAY = os.getenv('ENABLE_MEMORY_DECAY', 'false').lower() == 'true'
DECAY_WEIGHT = float(os.getenv('DECAY_WEIGHT', '0.3'))
DECAY_SCALE_DAYS = float(os.getenv('DECAY_SCALE_DAYS', '90'))
USE_NATIVE_DECAY = os.getenv('USE_NATIVE_DECAY', 'false').lower() == 'true'
MCP_CLIENT_CWD = os.getenv('MCP_CLIENT_CWD', os.getcwd())

# Embedding configuration
# Required environment variables
if not os.getenv('EMBEDDING_MODEL'):
    raise ValueError("EMBEDDING_MODEL environment variable must be set")
if not os.getenv('VECTOR_SIZE'):
    raise ValueError("VECTOR_SIZE environment variable must be set")

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
VECTOR_SIZE = int(os.getenv('VECTOR_SIZE'))

# Smart model initialization with age check
def is_model_fresh(model_name: str, cache_dir: str, max_age_days: int = 7) -> bool:
    """Check if the model is cached and fresh (not older than max_age_days)."""
    import time
    from pathlib import Path
    
    # Model cache path (FastEmbed format)
    model_cache_path = Path(cache_dir) / model_name.replace('/', '_')
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
        with open(timestamp_file, 'r') as f:
            timestamp = float(f.read().strip())
        
        # Check age
        age_seconds = time.time() - timestamp
        age_days = age_seconds / (24 * 60 * 60)
        
        logger.debug(f"Model age: {age_days:.1f} days (max: {max_age_days})")
        return age_days <= max_age_days
        
    except (ValueError, FileNotFoundError) as e:
        logger.debug(f"Error reading timestamp: {e}")
        return False

def initialize_embedding_model(model_name: str, cache_dir: str):
    """Initialize embedding model with smart offline/online mode."""
    import time
    from pathlib import Path
    
    # Get max age from environment
    max_age_days = int(os.getenv('MODEL_CACHE_DAYS', '7'))
    
    # Check if model is fresh
    model_fresh = is_model_fresh(model_name, cache_dir, max_age_days)
    
    if model_fresh:
        # Use offline mode
        logger.info(f"Using cached model in offline mode: {model_name}")
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    else:
        # Use online mode
        logger.info(f"Model not found or stale, downloading: {model_name}")
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_OFFLINE'] = '0'
    
    try:        
        # Update timestamp file after successful initialization
        model_cache_path = Path(cache_dir) / model_name.replace('/', '_')
        timestamp_file = model_cache_path / ".timestamp"
        
        # Ensure directory exists
        model_cache_path.mkdir(parents=True, exist_ok=True)

        model = TextEmbedding(
            model_name=model_name,
            cache_dir=model_cache_path
        )
        
        # Write current timestamp
        with open(timestamp_file, 'w') as f:
            f.write(str(time.time()))
        
        logger.info(f"Model initialized successfully: {model_name}")
        return model
        
    except ImportError:
        logger.error("FastEmbed not available. Install with: pip install fastembed")
        raise

# Get cache directory from environment
CACHE_DIR = os.getenv('TRANSFORMERS_CACHE', '/home/mcpuser/.cache/huggingface')
MODEL_CACHE_DAYS = int(os.getenv('MODEL_CACHE_DAYS', '7'))
HF_HOME = os.getenv('HF_HOME', None)

# Initialize local embedding model with smart caching
logger.info(f"Initializing embedding model: {EMBEDDING_MODEL} (vector size: {VECTOR_SIZE})")
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
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"VECTOR_SIZE: {VECTOR_SIZE}")
logger.info(f"TRANSFORMERS_CACHE: {CACHE_DIR}")
logger.info(f"HF_HOME: {HF_HOME or 'not set'}")
logger.info(f"MODEL_CACHE_DAYS: {MODEL_CACHE_DAYS}")
# Determine project name that will be used for all searches
# Convert MCP_CLIENT_CWD to watcher format: /path/to/project -> -path-to-project
if not MCP_CLIENT_CWD:
    raise ValueError("MCP_CLIENT_CWD is not set - cannot determine project")

DEFAULT_PROJECT_NAME = MCP_CLIENT_CWD.replace('/', '-')
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
    collection_name: str
    field: Optional[str] = None  # Тип контента: text, code, stdout, error


# Initialize FastMCP instance
mcp = FastMCP(
    name="claude-self-reflect",
    instructions="Search past conversations and store reflections with time-based memory decay"
)

# Create Qdrant client
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)

# Main collection for all conversations
MAIN_COLLECTION = "claude_logs"

# Removed get_all_collections_metadata - no longer needed with single collection

# Removed find_project_collections_by_path - no longer needed with single collection
    
async def get_all_collections() -> List[str]:
    """Get all collections (claude_logs and reflections)."""
    collections = await qdrant_client.get_collections()
    # Support claude_logs and reflections collections
    return [c.name for c in collections.collections 
            if c.name == 'claude_logs' or c.name.startswith('reflections')]

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding using local FastEmbed model."""
    if not local_embedding_model:
        raise ValueError("Local embedding model not initialized")
    
    # Run in executor since fastembed is synchronous
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, lambda: list(local_embedding_model.embed([text]))
    )
    return embeddings[0].tolist()

def get_embedding_dimension() -> int:
    """Get embedding dimension from environment (defaults to 1024 for e5-large)."""
    return VECTOR_SIZE

def get_collection_suffix() -> str:
    """Get the collection suffix for local embeddings."""
    return "_local"
 
# Register tools
@mcp.tool()
async def reflect_on_past(
    ctx: Context,
    query: str = Field(description="The search query to find semantically similar conversations"),
    limit: int = Field(default=5, description="Maximum number of results to return"),
    min_score: float = Field(default=0.7, description="Minimum similarity score (0-1)"),
    use_decay: Union[int, str] = Field(default=-1, description="Apply time-based decay: 1=enable, 0=disable, -1=use environment default (accepts int or str)"),
    project: Optional[str] = Field(default=None, description="Search specific project only. If not provided, searches current project based on working directory. Use 'all' to search across all projects."),
) -> str:
    """Search for relevant past conversations using semantic search with optional time decay."""
    
    # Log all incoming parameters
    logger.debug("reflect_on_past called with parameters:")
    logger.debug(f"  query: {query}")
    logger.debug(f"  limit: {limit}")
    logger.debug(f"  min_score: {min_score}")
    logger.debug(f"  use_decay: {use_decay}")
    logger.debug(f"  project: {project}")
    
    # Normalize use_decay to integer
    if isinstance(use_decay, str):
        try:
            use_decay = int(use_decay)
        except ValueError:
            raise ValueError("use_decay must be '1', '0', or '-1'")
    
    # Parse decay parameter using integer approach
    should_use_decay = (
        True if use_decay == 1
        else False if use_decay == 0
        else ENABLE_MEMORY_DECAY  # -1 or any other value
    )
    
    # Determine project scope
    if project is None:
        # Use default project name determined at startup
        project_name = DEFAULT_PROJECT_NAME
        logger.info(f"reflect_on_past: No project specified, using default: {project_name}")
    elif project == 'all':
        project_name = 'all'
        logger.info(f"reflect_on_past: Searching across all projects")
    else:
        # Convert user-provided project to watcher format
        project_name = project.replace('/', '-') if '/' in project else project
        logger.info(f"reflect_on_past: Using specified project: {project_name}")
 
    try:
        # Generate embedding
        query_embedding = await generate_embedding(query)
        
        # Check if claude_logs collection exists
        all_collections = await get_all_collections()
        if 'claude_logs' not in all_collections:
            return "No conversation collection found. Please import conversations first."
        
        # Build search filter based on project
        search_filter = None
        if project_name != 'all':
            logger.info(f"reflect_on_past: Creating filter for project_name={project_name}")
            search_filter = {
                "must": [
                    {
                        "key": "project_name",
                        "match": {"value": project_name}
                    }
                ]
            }
        else:
            logger.info("reflect_on_past: No filter - searching across all projects")
        
        logger.debug("Searching in claude_logs collection")
        logger.debug("Using local FastEmbed embeddings")
        
        all_results = []
        
        # Search in claude_logs collection
        collection_name = 'claude_logs'
        try:
            logger.debug(f"Searching in collection: {collection_name}")
            if should_use_decay and USE_NATIVE_DECAY and NATIVE_DECAY_AVAILABLE:
                    # Use native Qdrant decay with newer API
                    logger.debug(f"Using NATIVE Qdrant decay (new API) for {collection_name}")
                    
                    # Build the query with native Qdrant decay formula using newer API
                    query_obj = Query(
                        nearest=query_embedding,
                        formula=Formula(
                            sum=[
                                # Original similarity score
                                Expression(variable="score"),
                                # Decay boost term
                                Expression(
                                    mult=MultExpression(
                                        mult=[
                                            # Decay weight
                                            Expression(constant=DECAY_WEIGHT),
                                            # Exponential decay function
                                            Expression(
                                                exp_decay=DecayParamsExpression(
                                                    # Use timestamp field for decay
                                                    x=Expression(datetime_key="timestamp"),
                                                    # Decay from current time (server-side)
                                                    target=Expression(datetime="now"),
                                                    # Scale in milliseconds
                                                    scale=DECAY_SCALE_DAYS * 24 * 60 * 60 * 1000,
                                                    # Standard exponential decay midpoint
                                                    midpoint=0.5
                                                )
                                            )
                                        ]
                                    )
                                )
                            ]
                        )
                    )
                    
                    # Execute query with native decay (new API)
                    results = await qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_obj,
                        limit=limit,
                        score_threshold=min_score,
                        with_payload=True,
                        query_filter=search_filter  # Apply project filter
                    )
            elif should_use_decay and USE_NATIVE_DECAY and not NATIVE_DECAY_AVAILABLE:
                    # Use native Qdrant decay with older API
                    logger.debug(f"Using NATIVE Qdrant decay (legacy API) for {collection_name}")
                    
                    # Build the query with native Qdrant decay formula using older API
                    query_obj = FormulaQuery(
                        nearest=query_embedding,
                        formula=SumExpression(
                            sum=[
                                # Original similarity score
                                'score',  # Variable expression can be a string
                                # Decay boost term
                                {
                                    'mult': [
                                        # Decay weight (constant as float)
                                        DECAY_WEIGHT,
                                        # Exponential decay function
                                        {
                                            'exp_decay': DecayParamsExpression(
                                                # Use timestamp field for decay
                                                x=DatetimeKeyExpression(datetime_key='timestamp'),
                                                # Decay from current time (server-side)
                                                target=DatetimeExpression(datetime='now'),
                                                # Scale in milliseconds
                                                scale=DECAY_SCALE_DAYS * 24 * 60 * 60 * 1000,
                                                # Standard exponential decay midpoint
                                                midpoint=0.5
                                            )
                                        }
                                    ]
                                }
                            ]
                        )
                    )
                    
                    # Execute query with native decay
                    results = await qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_obj,
                        limit=limit,
                        score_threshold=min_score,
                        with_payload=True,
                        query_filter=search_filter  # Apply project filter
                    )
                    
                    # Process results from native decay search
                    for point in results.points:
                        # Clean timestamp for proper parsing
                        raw_timestamp = point.payload.get('timestamp', datetime.now(timezone.utc).isoformat())
                        clean_timestamp = raw_timestamp.replace('Z', '+00:00') if raw_timestamp.endswith('Z') else raw_timestamp
                        
                        # Get project name from payload
                        point_project = point.payload.get('project_name', 'unknown')
                        
                        all_results.append(SearchResult(
                            id=str(point.id),
                            score=point.score,  # Score already includes decay
                            timestamp=clean_timestamp,
                            role=point.payload.get('start_role', point.payload.get('role', 'unknown')),
                            excerpt=(point.payload.get('text', '')[:500] + '...'),
                            project_name=point_project,
                            conversation_id=point.payload.get('conversation_id'),
                            collection_name=collection_name,
                            field=point.payload.get('field')  # Тип контента
                        ))
                    
            elif should_use_decay:
                    # Use client-side decay (existing implementation)
                    logger.debug(f"Using CLIENT-SIDE decay for {collection_name}")
                    
                    # Search without score threshold to get all candidates
                    results = await qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=limit * 3,  # Get more candidates for decay filtering
                        with_payload=True,
                        query_filter=search_filter  # Apply project filter
                    )
                    
                    # Apply decay scoring manually
                    now = datetime.now(timezone.utc)  # Use UTC timezone-aware datetime
                    scale_ms = DECAY_SCALE_DAYS * 24 * 60 * 60 * 1000
                    
                    decay_results = []
                    for point in results:
                        try:
                            # Get timestamp from payload
                            timestamp_str = point.payload.get('timestamp')
                            if timestamp_str:
                                # Parse timestamp, handling both naive and aware datetimes
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                # If timestamp is naive, assume UTC
                                if timestamp.tzinfo is None:
                                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                                age_ms = (now - timestamp).total_seconds() * 1000
                                
                                # Calculate decay factor
                                decay_factor = np.exp(-age_ms / scale_ms)
                                
                                # Apply decay formula
                                adjusted_score = point.score + (DECAY_WEIGHT * decay_factor)
                                
                                # Debug: show the calculation
                                age_days = age_ms / (24 * 60 * 60 * 1000)
                                logger.debug(f"Point: age={age_days:.1f} days, original_score={point.score:.3f}, decay_factor={decay_factor:.3f}, adjusted_score={adjusted_score:.3f}")
                            else:
                                adjusted_score = point.score
                            
                            # Only include if above min_score after decay
                            if adjusted_score >= min_score:
                                decay_results.append((adjusted_score, point))
                        
                        except Exception as e:
                            logger.debug(f"Error applying decay to point: {e}")
                            decay_results.append((point.score, point))
                    
                    # Sort by adjusted score and take top results
                    decay_results.sort(key=lambda x: x[0], reverse=True)
                    
                    # Convert to SearchResult format
                    for adjusted_score, point in decay_results[:limit]:
                        # Clean timestamp for proper parsing
                        raw_timestamp = point.payload.get('timestamp', datetime.now(timezone.utc).isoformat())
                        clean_timestamp = raw_timestamp.replace('Z', '+00:00') if raw_timestamp.endswith('Z') else raw_timestamp
                        
                        # Get project name from payload
                        point_project = point.payload.get('project_name', 'unknown')
                        
                        all_results.append(SearchResult(
                            id=str(point.id),
                            score=adjusted_score,  # Use adjusted score
                            timestamp=clean_timestamp,
                            role=point.payload.get('start_role', point.payload.get('role', 'unknown')),
                            excerpt=(point.payload.get('text', '')[:500] + '...'),
                            project_name=point_project,
                            conversation_id=point.payload.get('conversation_id'),
                            collection_name=collection_name,
                            field=point.payload.get('field')  # Тип контента
                        ))
            else:
                # Standard search without decay
                    results = await qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=limit,
                        score_threshold=min_score,
                        with_payload=True,
                        query_filter=search_filter  # Apply project filter
                    )
                    logger.debug(f"Found {len(results)} results in {collection_name} with score >= {min_score}")
                    
                    for point in results:
                        # Clean timestamp for proper parsing
                        raw_timestamp = point.payload.get('timestamp', datetime.now(timezone.utc).isoformat())
                        clean_timestamp = raw_timestamp.replace('Z', '+00:00') if raw_timestamp.endswith('Z') else raw_timestamp
                        
                        # Get project name from payload
                        point_project = point.payload.get('project_name', 'unknown')
                        
                        all_results.append(SearchResult(
                            id=str(point.id),
                            score=point.score,
                            timestamp=clean_timestamp,
                            role=point.payload.get('start_role', point.payload.get('role', 'unknown')),
                            excerpt=(point.payload.get('text', '')[:500] + '...'),
                            project_name=point_project,
                            conversation_id=point.payload.get('conversation_id'),
                            collection_name=collection_name,
                            field=point.payload.get('field')  # Тип контента
                        ))
            
        except Exception as e:
            logger.error(f"Error searching {collection_name}: {str(e)}")
            return f"Error searching conversations: {str(e)}"
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Total results before limiting: {len(all_results)}")
        logger.debug(f"Top 3 results: {[(r.score, r.project_name, r.collection_name) for r in all_results[:3]]}")
        all_results = all_results[:limit]
        
        if not all_results:
            return f"No conversations found matching '{query}'. Try different keywords or check if conversations have been imported."
        
        # Format results
        result_text = f"Found {len(all_results)} relevant conversation(s) for '{query}':\n\n"
        for i, result in enumerate(all_results):
            result_text += f"**Result {i+1}** (Score: {result.score:.3f})\n"
            # Handle timezone suffix 'Z' properly
            timestamp_clean = result.timestamp.replace('Z', '+00:00') if result.timestamp.endswith('Z') else result.timestamp
            result_text += f"Time: {datetime.fromisoformat(timestamp_clean).strftime('%Y-%m-%d %H:%M:%S')}\n"
            result_text += f"Project: {result.project_name}\n"
            result_text += f"Role: {result.role}\n"
            if result.field:
                result_text += f"Type: {result.field}\n"
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
    tags: List[str] = Field(default=[], description="Tags to categorize this reflection"),
    project: Optional[str] = Field(default=None, description="Target project for the reflection. If not provided, uses current project."),
) -> str:
    """Store an important insight or reflection for future reference."""
    
    try:
        # Determine project scope
        if project is None:
            # Use default project name determined at startup
            project_name = DEFAULT_PROJECT_NAME
            logger.info(f"store_reflection: No project specified, using default: {project_name}")
        else:
            # Convert user-provided project to watcher format
            project_name = project.replace('/', '-') if '/' in project else project
            logger.info(f"store_reflection: Using specified project: {project_name}")
        
        # Use claude_logs collection
        collection_name = 'claude_logs'
        
        # Ensure collection exists
        try:
            await qdrant_client.get_collection(collection_name)
        except Exception:
            # Create collection if it doesn't exist
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=get_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )
            logger.debug(f"Created collection: {collection_name}")
        
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
                "field": "text",  # For compatibility with watcher
                "source": "reflection"  # To distinguish from watcher imports
            }
        )
        
        # Store in Qdrant
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        tags_str = ', '.join(tags) if tags else 'none'
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
    import sys
    logger.info(f"MCP server started with args: {sys.argv}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
