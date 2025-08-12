# План улучшений импорта/поиска для логов Claude Code (обновлённый)

**Допущения (оставляем как есть):**
- При `full-check` читаем с `start_line` (потеря изменений «в середине» допустима).
- При удалении/переиндексации файлов точки в Qdrant **не** удаляем.

---

## 1) Цели
- Повысить релевантность по RU/EN и коду.
- Учитывать роли (`user/assistant`) и строго разделять текст/код.
- Стабилизировать размеры чанков → исключить усечение в эмбеддере.
- Подготовить базу к миграции на многоязычный эмбеддер.

## 2) Состояние (кратко)
- Чанкование по количеству сообщений (`CHUNK_SIZE`), сплит по `user → assistant`, разрыв по времени от старта чанка, без префиксов ролей, ID — усечённый SHA256.
- Эмбеддер: `all-MiniLM-L6-v2` (384d), усечение по словам/токенам, коллекции созданы под 384d.
- В Compose нет payload-индексов/настроек HNSW для Qdrant.

---

## 3) Извлечение и схема данных
**Разделяем текст и код; учитываем роль и источник.**
- `message.content`: парсить markdown-фенсы. Вне ``` → `field=text`; внутри ``` → `field=code`, payload `lang`.
- `toolUseResult.*` индексировать отдельно:
  - `originalFile`, `originalFileContents` → `field=code`
  - `newString`, `oldString` → `field=code`
  - `stdout` → `field=stdout`
  - `stderr/stack` (если есть) → `field=error`
- Payload чанка: `role`, `source (message|toolUseResult)`, `field (text|code|stdout|error)`, `session_id`, `turn_id`, `timestamp`, `filePath`, `project_name`, `model`.
- Для `message.content` формировать **два типа чанков**: текст (Q+A вместе) и код (каждый fenced-блок — отдельно).

Мини-парсер фенсов (пример):
```python
FENCE = re.compile(r"```(\w+)?\n(.*?)```", re.S)
def split_message_content(s: str):
    parts, last = [], 0
    for m in FENCE.finditer(s or ""):
        if m.start() > last:
            txt = (s[last:m.start()] or "").strip()
            if txt: parts.append({"field":"text","content":txt})
        lang = (m.group(1) or "").lower()
        code = (m.group(2) or "").rstrip()
        if code: parts.append({"field":"code","content":code,"lang":lang})
        last = m.end()
    tail = (s[last:] or "").strip()
    if tail: parts.append({"field":"text","content":tail})
    return parts
```

---

## 4) Чанкование
- **Граница по ролям:** делить на `assistant → user`, чтобы «вопрос+ответ» чаще оставались в одном **текстовом** чанке.
- **Ограничение размера:** по символам/токенам (не по числу сообщений).
- **Разрывы по времени:** начинать новый чанк при крупном gap между соседними сообщениями (например, >30 мин).
- **Overlap**: текст — 15% длины; код — 10 строк.
- **ID чанка:** увеличить префикс SHA256 до 24–32 hex.

**Рекомендуемые параметры (по вашей статистике):**
- `text` (message.content вне фенсов): **1200 симв.**, overlap **180**
- `error/stack`: **1200 симв.**, overlap **180**
- `code` (message.content и toolUseResult.new/oldString): **80 строк**, overlap **10**
- `originalFile*`: **200 строк**, overlap **10**
- `stdout`: **80 строк**, overlap **10**

В `combined_text` для текстовых чанков добавить префиксы ролей `User:` / `Assistant:`.

---

## 5) Эмбеддинги и миграция
- Переход с `all-MiniLM-L6-v2 (384d)` на **`intfloat/multilingual-e5-large (1024d)`**.
- Последствия: ↑вектор, ↑память/диск, ↓скорость; оценить ресурсы.
- **A/B подход:** завести вторую коллекцию под 1024d, импортировать параллельно, сравнить качество/скорость, затем принять решение.

Настройки:
- L2-нормализация, метрика **Cosine**.
- Batch-энкодинг 64–128; модель либо на воркер, либо mutex вокруг энкодера.

---

## 6) Поисковый пайплайн
- **Гибридный поиск**: dense + sparse (BM25 или named-sparse в Qdrant).
  - Стартовые веса: `α=0.55` (dense), `β=0.35` (sparse), `γ=0.10` (boost полей/ролей)
- **MMR**: topK=50 → finalK=20, `λ=0.3` (снижаем дубли стеков/патчей).
- **Role-aware boosts**: анализ ассистента → `role=assistant` +0.2; поиск по вводу → `role=human` +0.2.
- **Query routing**: признаки кода (`()`, `{}`, `.py/.ts`, `TypeError`, и т.п.) → поднять вес `code/error` и `β`.

---

## 7) Qdrant
- Вектора: **size=1024**, `distance=Cosine`.
- HNSW: `M=64`, `ef_construct=256`, `ef_search=256`.
- Payload-индексы: `role`, `field`, `session_id`, `filePath`, `project_name`, `timestamp`, `model`.
- (Опц.) Named sparse в той же коллекции для гибридного скоринга.

---

## 8) Изменения в коде (минимально)
- `ChunkGenerator.create_chunks()`:
  - сплит по `assistant → user`;
  - gap между **соседними** сообщениями;
  - лимит по символам/токенам и overlap.
- `_finalize_chunk()`:
  - префиксы `User:`/`Assistant:` в `combined_text` для текстовых чанков;
  - увеличить префикс SHA256.
- Парсер фенсов и ingest `toolUseResult.*` (код/stdout/stderr) как отдельные чанки.
- `EmbeddingService`: новая модель и размерность; убрать усечение «по словам».
- `QdrantService.ensure_collection()`: обновить `VectorParams.size`; добавить payload-индексы.

(Логика `full-check` и отсутствие удаления точек — без изменений.)

---

## 9) Docker/Compose
- Вынести в env: `EMBEDDING_MODEL`, `VECTOR_SIZE`, `BATCH_SIZE`, лимиты памяти.
- Отдельные volumes: модели/кеш, `STATE_FILE`, данные Qdrant.
- Healthchecks для Qdrant и watcher.
- Настройка `QDRANT__STORAGE__OPTIMIZERS__DEFAULT_MEMMAP_THRESHOLD` под размер коллекции.

---

## 10) Миграция (A/B)
1) Обновить код (чанкование, парсер фенсов, ingest toolUseResult.*, новая модель/размерность).
2) Создать **новые** коллекции под 1024d (или 768d).
3) Запустить ре-импорт в новые коллекции, старые не трогать.
4) Сравнить качество/скорость на типовых запросах (RU/EN/код).
5) Принять решение, затем переключить прод на новую коллекцию.

---

## 11) Критерии успеха
- Лучшая релевантность для RU/EN и кодовых запросов.
- Меньше усечённых/слишком длинных текстов.
- Стабильные задержки и память под нагрузкой.
- Удобная фильтрация по роли/полю/файлу/сессии/времени.
