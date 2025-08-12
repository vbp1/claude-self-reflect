#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import-watcher-smart.py — обновлённый watcher для JSONL-логов Claude Code.

Ключевые изменения:
- Учитывает роли (message.role).
- Разделяет текст и код: парсит fenced-блоки из message.content.
- Индексирует toolUseResult.* как отдельные чанки (code/stdout/error).
- Чанкование по символам (текст) и по строкам (код) с overlap.
- Payload-индексы и HNSW-конфиг для Qdrant.
- Сохранены допущения: full-check читает с start_line; удаление точек из Qdrant не выполняется.

Требуемые пакеты: qdrant-client, fastembed
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import queue
import hashlib
import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    PayloadIndexParams,
    PayloadSchemaType,
)
from fastembed import TextEmbedding


# -----------------------
# Конфигурация (ENV)
# -----------------------

LOGS_DIR = os.getenv("LOGS_DIR", "/logs")
STATE_FILE = Path(os.getenv("STATE_FILE", "/config/watcher-state.json"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "claude_logs")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))  # default for multilingual-e5-large
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")  # FastEmbed model
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "96"))
NORMALIZE_L2 = os.getenv("NORMALIZE_L2", "true").lower() == "true"

# Чанкование
TEXT_CHARS = int(os.getenv("TEXT_CHARS", "1200"))
TEXT_OVERLAP = int(os.getenv("TEXT_OVERLAP", "180"))
ERROR_CHARS = int(os.getenv("ERROR_CHARS", "1200"))
ERROR_OVERLAP = int(os.getenv("ERROR_OVERLAP", "180"))
CODE_LINES = int(os.getenv("CODE_LINES", "80"))
CODE_OVERLAP = int(os.getenv("CODE_OVERLAP", "10"))
BIGFILE_LINES = int(os.getenv("BIGFILE_LINES", "200"))
BIGFILE_OVERLAP = int(os.getenv("BIGFILE_OVERLAP", "10"))
TIME_GAP_SEC = int(os.getenv("TIME_GAP_SEC", "1800"))

# Параллельность и интервал
MAX_PARALLEL_FILES = int(os.getenv("MAX_PARALLEL_FILES", "2"))
IMPORT_INTERVAL = int(os.getenv("IMPORT_INTERVAL", "30"))

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("watcher")


# -----------------------
# Утилиты
# -----------------------

FENCE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.S)

def safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def get_role(msg: Dict[str, Any]) -> str:
    r = safe_get(msg, "message.role", "")
    if r in ("user", "human"):
        return "user"
    if r == "assistant":
        return "assistant"
    # запасной вариант: type=user/assistant
    t = safe_get(msg, "message.type", "")
    if t in ("user", "assistant"):
        return t
    return "assistant"

def norm_ts(ts: Optional[str]) -> Optional[str]:
    if not ts or not isinstance(ts, str):
        return None
    # не нормализуем до datetime; храним строкой как есть
    return ts

def l2_normalize(vec: List[float]) -> List[float]:
    if not vec:
        return vec
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# -----------------------
# Состояние
# -----------------------

@dataclass
class FileInfo:
    path: str
    mtime: float
    size: int
    hash: Optional[str] = None
    start_line: int = 0  # последняя обработанная строка (включительно)
    last_processed_at: Optional[float] = None
    chunks_count: int = 0
    lines_processed: int = 0

@dataclass
class WatcherState:
    files: Dict[str, FileInfo] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "WatcherState":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                files = {
                    k: FileInfo(**v) if isinstance(v, dict) else None
                    for k, v in data.get("files", {}).items()
                }
                files = {k: v for k, v in files.items() if v is not None}
                return cls(files=files)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return cls()

    def save(self, path: Path):
        tmp = path.with_suffix(".tmp")
        data = {"files": {k: asdict(v) for k, v in self.files.items()}}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


# -----------------------
# Qdrant
# -----------------------

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._ensure_collection()

    def _ensure_collection(self):
        exists = False
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            exists = True
            # Проверим размер вектора (если отличается — сообщим)
            vs = info.config.params.vectors
            try:
                size = vs["size"]  # dict-like
            except Exception:
                size = vs.size
            if size != VECTOR_SIZE:
                logger.warning(f"Collection {COLLECTION_NAME} vector size={size}, expected {VECTOR_SIZE}")
        except Exception:
            exists = False

        if not exists:
            logger.info(f"Creating collection {COLLECTION_NAME} (size={VECTOR_SIZE})")
            self.client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=64, ef_construct=256),
            )
            # payload индексы
            for fld, t in [
                ("role", PayloadSchemaType.KEYWORD),
                ("field", PayloadSchemaType.KEYWORD),
                ("session_id", PayloadSchemaType.KEYWORD),
                ("filePath", PayloadSchemaType.KEYWORD),
                ("project_name", PayloadSchemaType.KEYWORD),
                ("model", PayloadSchemaType.KEYWORD),
            ]:
                try:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fld,
                        field_schema=PayloadIndexParams(t)
                    )
                except Exception:
                    pass
            # timestamp — как integer (unix) или keyword-строка; оставим строкой (keyword)
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="timestamp",
                    field_schema=PayloadIndexParams(PayloadSchemaType.KEYWORD),
                )
            except Exception:
                pass

    def upsert_points(self, points: List[PointStruct]):
        if not points:
            return
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)


# -----------------------
# Эмбеддинги
# -----------------------

class EmbeddingService:
    def __init__(self):
        logger.info(f"Initializing FastEmbed model: {EMBEDDING_MODEL}")
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # FastEmbed возвращает генератор numpy массивов; приведём к python list
        vectors = []
        for vec in self.model.embed(texts, batch_size=EMBED_BATCH):
            v = vec.tolist()
            if NORMALIZE_L2:
                v = l2_normalize(v)
            vectors.append(v)
        return vectors


# -----------------------
# Извлечение и чанкование
# -----------------------

def split_message_content(text: str) -> List[Dict[str, Any]]:
    """Разделяет message.content на текстовые и кодовые части."""
    parts: List[Dict[str, Any]] = []
    if not isinstance(text, str):
        return parts
    last = 0
    for m in FENCE_RE.finditer(text):
        # текст до блока
        if m.start() > last:
            t = text[last:m.start()].strip()
            if t:
                parts.append({"field": "text", "content": t})
        lang = (m.group(1) or "").lower()
        code = (m.group(2) or "").rstrip()
        if code:
            parts.append({"field": "code", "content": code, "lang": lang})
        last = m.end()
    # хвост
    tail = text[last:].strip()
    if tail:
        parts.append({"field": "text", "content": tail})
    return parts


def chunk_text_chars(s: str, size: int, overlap: int) -> List[str]:
    res = []
    if not s:
        return res
    start = 0
    n = len(s)
    size = max(1, size)
    while start < n:
        end = min(n, start + size)
        res.append(s[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= n:
            break
    return res


def chunk_code_lines(code: str, size_lines: int, overlap_lines: int) -> List[str]:
    res = []
    if not code:
        return res
    lines = code.splitlines()
    n = len(lines)
    if n == 0:
        return []
    i = 0
    while i < n:
        j = min(n, i + size_lines)
        block = "\n".join(lines[i:j])
        if block.strip():
            res.append(block)
        if j >= n:
            break
        i = max(0, j - overlap_lines)
        if i >= n:
            break
    return res


def build_points_from_log_lines(
    lines: List[str],
    project_name: str,
    file_path: str,
) -> Tuple[List[PointStruct], int]:
    """
    Превращает список JSONL строк в список PointStruct для Qdrant.
    Не хранит историю удаления; full-check/удаление оставлено как есть.
    Возвращает (points, messages_count).
    """
    points: List[PointStruct] = []

    # Текущий «текстовый» чанк диалога
    cur_role = None  # роль первого сообщения в чанке
    cur_parts: List[Tuple[str, str]] = []  # (role, text) только для TEXT чанка
    cur_len = 0
    cur_first_ts: Optional[str] = None
    cur_session: Optional[str] = None
    cur_turn_id: Optional[str] = None
    text_chunks_total = 0
    msgs_in_text_chunk = 0

    def flush_text_chunk():
        nonlocal cur_role, cur_parts, cur_len, cur_first_ts, cur_session, cur_turn_id
        nonlocal text_chunks_total, msgs_in_text_chunk

        if not cur_parts:
            return

        # Собираем combined_text c префиксами ролей
        pieces = []
        for r, t in cur_parts:
            prefix = "User:" if r == "user" else "Assistant:"
            pieces.append(f"{prefix}\n{t}".strip())
        combined = "\n\n".join(pieces)
        # Делим combined по символам
        for part in chunk_text_chars(combined, TEXT_CHARS, TEXT_OVERLAP):
            pid = sha256_hex(f"{file_path}|text|{cur_first_ts}|{part}")[:32]
            payload = {
                "project_name": project_name,
                "filePath": file_path,
                "field": "text",
                "role": cur_role or "assistant",
                "session_id": cur_session or "unknown",
                "turn_id": cur_turn_id or "",
                "timestamp": cur_first_ts or "",
                "source": "message",
                "model": "",  # при необходимости заполнить из последнего msg
            }
            points.append(PointStruct(id=pid, vector=None, payload=payload))  # vector заполним позже
            # Временный текст кладём в payload для эмбеддинга позже
            payload["_text"] = part
            text_chunks_total += 1

        # reset
        cur_role = None
        cur_parts = []
        cur_len = 0
        msgs_in_text_chunk = 0
        cur_first_ts = None
        cur_session = None
        cur_turn_id = None

    def add_text(role: str, text: str, ts: Optional[str], session_id: Optional[str], turn_id: Optional[str]):
        nonlocal cur_role, cur_len, cur_first_ts, cur_session, cur_turn_id, msgs_in_text_chunk
        # Создаём или продолжаем чанк
        if cur_role is None:
            cur_role = role
            cur_first_ts = ts
            cur_session = session_id
            cur_turn_id = turn_id
        cur_parts.append((role, text))
        cur_len += len(text)
        msgs_in_text_chunk += 1
        # Делим при переполнении
        if cur_len >= TEXT_CHARS:
            flush_text_chunk()

    # Обход строк
    last_ts_value: Optional[float] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        ts = norm_ts(obj.get("timestamp") or obj.get("ts"))
        session_id = obj.get("sessionId") or obj.get("session_id") or "unknown"
        turn_id = obj.get("message", {}).get("id") or obj.get("uuid") or ""
        role = get_role(obj)
        model = safe_get(obj, "message.model", "")

        # Проверка на разрыв по времени (по соседним сообщениям)
        # Если есть предыдущий ts, сравним
        ts_float = None
        if isinstance(ts, str) and len(ts) >= 19:
            # простая эвристика: брать unix из поля unixTs, если есть
            unix_ts = obj.get("unixTs") or obj.get("unix_ts")
            if isinstance(unix_ts, (int, float)):
                ts_float = float(unix_ts)
            else:
                ts_float = None
        if ts_float is not None and last_ts_value is not None and (ts_float - last_ts_value) > TIME_GAP_SEC:
            flush_text_chunk()
        if ts_float is not None:
            last_ts_value = ts_float

        # 1) message.content → split на text/code
        content = safe_get(obj, "message.content", "")
        if content:
            parts = split_message_content(content)
            for p in parts:
                if p["field"] == "text":
                    add_text(role, p["content"], ts, session_id, turn_id)
                else:
                    # code из message.content → chunk по CODE_LINES
                    for block in chunk_code_lines(p["content"], CODE_LINES, CODE_OVERLAP):
                        pid = sha256_hex(f"{file_path}|mc_code|{ts}|{block[:64]}")[:32]
                        payload = {
                            "project_name": project_name,
                            "filePath": file_path,
                            "field": "code",
                            "role": role,
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "timestamp": ts or "",
                            "source": "message",
                            "model": model,
                        }
                        points.append(PointStruct(id=pid, vector=None, payload=payload))
                        payload["_text"] = block  # для эмбеддинга

        # Если смена роли с assistant → user, то закрываем текущий текстовый чанк
        if cur_role == "assistant" and role == "user":
            flush_text_chunk()

        # 2) toolUseResult.* → отдельные чанки
        tur = obj.get("toolUseResult") or {}
        if isinstance(tur, dict):
            # stdout
            stdout = tur.get("stdout")
            if stdout:
                for block in chunk_code_lines(stdout, CODE_LINES, CODE_OVERLAP):
                    pid = sha256_hex(f"{file_path}|stdout|{ts}|{block[:64]}")[:32]
                    payload = {
                        "project_name": project_name,
                        "filePath": tur.get("filePath") or file_path,
                        "field": "stdout",
                        "role": role,
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "timestamp": ts or "",
                        "source": "toolUseResult",
                        "model": model,
                    }
                    points.append(PointStruct(id=pid, vector=None, payload=payload))
                    payload["_text"] = block

            stderr = tur.get("stderr")
            if stderr:
                for block in chunk_text_chars(stderr, ERROR_CHARS, ERROR_OVERLAP):
                    pid = sha256_hex(f"{file_path}|stderr|{ts}|{block[:64]}")[:32]
                    payload = {
                        "project_name": project_name,
                        "filePath": tur.get("filePath") or file_path,
                        "field": "error",
                        "role": role,
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "timestamp": ts or "",
                        "source": "toolUseResult",
                        "model": model,
                    }
                    points.append(PointStruct(id=pid, vector=None, payload=payload))
                    payload["_text"] = block

            # code-like поля
            for name, kind, linesz, over in [
                ("newString", "code", CODE_LINES, CODE_OVERLAP),
                ("oldString", "code", CODE_LINES, CODE_OVERLAP),
                ("originalFile", "code", BIGFILE_LINES, BIGFILE_OVERLAP),
                ("originalFileContents", "code", BIGFILE_LINES, BIGFILE_OVERLAP),
            ]:
                val = tur.get(name)
                if val:
                    for block in chunk_code_lines(val, linesz, over):
                        pid = sha256_hex(f"{file_path}|{name}|{ts}|{block[:64]}")[:32]
                        payload = {
                            "project_name": project_name,
                            "filePath": tur.get("filePath") or file_path,
                            "field": kind,
                            "role": role,
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "timestamp": ts or "",
                            "source": "toolUseResult",
                            "model": model,
                            "src": name,
                        }
                        points.append(PointStruct(id=pid, vector=None, payload=payload))
                        payload["_text"] = block

        # можно расширить: toolUseResult.file.content и т.п.

    # финальный сброс текстового чанка
    flush_text_chunk()

    msgs_count = 0
    # попытаться оценить число сообщений в lines (не критично)
    for line in lines:
        try:
            obj = json.loads(line)
            if "message" in obj:
                msgs_count += 1
        except Exception:
            pass

    return points, msgs_count


# -----------------------
# Watcher
# -----------------------

class Watcher:
    def __init__(self):
        self.state = WatcherState.load(STATE_FILE)
        self.qdrant = QdrantService()
        self.embed = EmbeddingService()

    def _project_of(self, file_path: str) -> str:
        try:
            return Path(file_path).parent.name
        except Exception:
            return "default"

    def _scan_files(self) -> List[Tuple[str, os.stat_result]]:
        out = []
        p = Path(LOGS_DIR)
        if not p.exists():
            return out
        for f in p.rglob("*.jsonl"):
            try:
                st = f.stat()
                out.append((str(f), st))
            except Exception:
                pass
        return out

    def _read_lines_from(self, file_path: str, start_line: int) -> Tuple[List[str], int]:
        """Читает файл, начиная с start_line (0-based). Возвращает (lines, new_start_line)."""
        res = []
        cur = 0
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if cur >= start_line:
                    res.append(line)
                cur += 1
        return res, cur  # cur = total_lines

    def _embed_and_upsert(self, points: List[PointStruct]):
        if not points:
            return
        # Собираем тексты для эмбеддинга
        texts = []
        order = []
        for idx, p in enumerate(points):
            payload = p.payload or {}
            txt = payload.pop("_text", None)
            if not isinstance(txt, str):
                # если нет текста (не должно быть), пропускаем point
                continue
            texts.append(txt)
            order.append(idx)
        if not texts:
            return

        vectors = self.embed.embed_batch(texts)
        # Раскладываем обратно по points
        for i, idx in enumerate(order):
            vec = vectors[i]
            # Создаём новый PointStruct с вектором и payload
            payload = points[idx].payload
            pid = points[idx].id
            points[idx] = PointStruct(id=pid, vector=vec, payload=payload)

        # Upsert
        self.qdrant.upsert_points(points)

    def process_file(self, file_path: str, st: os.stat_result):
        fi = self.state.files.get(file_path)
        if fi is None:
            fi = FileInfo(path=file_path, mtime=st.st_mtime, size=st.st_size, start_line=0)
            self.state.files[file_path] = fi

        start_line = fi.start_line or 0
        lines, total_lines = self._read_lines_from(file_path, start_line)
        if not lines:
            # обновим метаданные и выйдем
            fi.mtime = st.st_mtime
            fi.size = st.st_size
            fi.last_processed_at = time.time()
            return

        project_name = self._project_of(file_path)
        points, msgs_count = build_points_from_log_lines(lines, project_name, file_path)
        self._embed_and_upsert(points)

        # обновим state
        fi.mtime = st.st_mtime
        fi.size = st.st_size
        fi.start_line = total_lines  # продолжаем append-only
        fi.last_processed_at = time.time()
        fi.chunks_count += len(points)
        fi.lines_processed += len(lines)

    def run_once(self):
        files = self._scan_files()
        if not files:
            logger.info("No files found")
            return
        # обрабатываем в несколько потоков
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES) as ex:
            futs = []
            for path, st in files:
                futs.append(ex.submit(self.process_file, path, st))
            for f in futs:
                try:
                    f.result()
                except Exception as e:
                    logger.exception(f"process_file failed: {e}")

        # сохранить state
        try:
            self.state.save(STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def run(self):
        logger.info(
            f"Start watcher | LOGS_DIR={LOGS_DIR} QDRANT_URL={QDRANT_URL} "
            f"COLLECTION={COLLECTION_NAME} MODEL={EMBEDDING_MODEL} SIZE={VECTOR_SIZE}"
        )
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.exception(f"run_once error: {e}")
            time.sleep(IMPORT_INTERVAL)


# -----------------------
# main
# -----------------------

def main():
    w = Watcher()
    w.run()


if __name__ == "__main__":
    main()
