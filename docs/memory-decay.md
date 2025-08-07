# Memory Decay in Claude Self-Reflect

## Philosophy

### The Problem with Perfect Digital Memory

Traditional digital systems suffer from the "perfect memory curse" - as conversations accumulate over months and years, finding relevant information becomes increasingly difficult. Unlike human memory, which naturally prioritizes recent experiences, digital systems treat all data equally, leading to:

- **Information overload**: Older, less relevant results crowd out recent insights
- **Context drift**: Yesterday's solutions may not apply to today's problems
- **Noise accumulation**: The signal-to-noise ratio decreases over time

### Mimicking Human Memory Patterns

Claude Self-Reflect implements memory decay to mirror how human memory works:

1. **Recency bias**: Recent conversations are naturally more accessible
2. **Gradual fading**: Older memories don't disappear abruptly but gradually lose prominence
3. **Contextual relevance**: Important memories can still surface when highly relevant

This approach creates a more intuitive and effective search experience that aligns with how we naturally think about past conversations.

## Implementation Details

### Client-Side Exponential Decay

Claude Self-Reflect uses a client-side exponential decay algorithm with a 90-day half-life:

```python
# Exponential decay formula
decay_factor = exp(-ln(2) * days_old / half_life)
adjusted_score = base_score + (decay_weight * decay_factor)
```

### Technical Architecture

The decay is implemented using Qdrant's native formula capabilities:

```python
Formula(
    sum=[
        # Original similarity score
        Expression(variable="score"),
        # Decay boost term
        Expression(
            mult=MultExpression(
                mult=[
                    # Decay weight (default: 0.3)
                    Expression(constant=DECAY_WEIGHT),
                    # Exponential decay function
                    Expression(
                        exp_decay=DecayParamsExpression(
                            x=Expression(datetime_key="timestamp"),
                            target=Expression(datetime="now"),
                            scale=DECAY_SCALE_DAYS * 24 * 60 * 60 * 1000,
                            midpoint=0.5
                        )
                    )
                ]
            )
        )
    ]
)
```

### Key Parameters

- **Half-life**: 90 days (configurable via `DECAY_SCALE_DAYS`)
- **Decay weight**: 0.3 (configurable via `DECAY_WEIGHT`)
- **Midpoint**: 0.5 (standard exponential decay)

### Score Impact Timeline

With default settings:
- **Today**: 100% boost (full decay weight added)
- **1 week old**: ~95% boost
- **1 month old**: ~79% boost
- **3 months old**: ~50% boost (half-life)
- **6 months old**: ~25% boost
- **1 year old**: ~6% boost

## Benefits

### 1. Prioritizing Recency

Recent conversations contain the most up-to-date context about:
- Current project state
- Latest decisions and approaches
- Recent bug fixes and solutions
- Active development patterns

### 2. Reducing Noise

By de-emphasizing older content:
- Obsolete solutions naturally fade away
- Outdated patterns don't mislead current work
- Search results stay focused and relevant

### 3. Natural Memory Patterns

Users intuitively expect:
- Yesterday's conversation to be highly accessible
- Last month's discussion to be findable but less prominent
- Last year's chat to surface only when highly relevant

### 4. Adaptive Relevance

The system balances two factors:
- **Semantic similarity**: How well content matches the query
- **Temporal relevance**: How recent the content is

Highly relevant old content can still rank well, but recent content gets a natural boost.

## Configuration

### Enabling Memory Decay

Memory decay is disabled by default. To enable it:

```bash
# In your .env file
ENABLE_MEMORY_DECAY=true
```

Or set it when running Docker:

```bash
docker-compose up -d \
  -e ENABLE_MEMORY_DECAY=true
```

### Customizing Parameters

Fine-tune the decay behavior:

```bash
# Half-life in days (default: 90)
DECAY_SCALE_DAYS=90

# Decay weight factor (default: 0.3)
# Higher = more emphasis on recency
# Lower = more emphasis on content similarity
DECAY_WEIGHT=0.3
```

### Per-Search Control

Override the global setting for individual searches:

```python
# Force enable decay for this search
results = await reflect_on_past(
    query="deployment strategies",
    use_decay=1  # Force enable
)

# Disable decay for historical research
results = await reflect_on_past(
    query="initial architecture decisions",
    use_decay=0  # Force disable
)
```

## Performance Implications

### Computational Overhead

Memory decay adds minimal overhead:
- **Calculation**: Simple exponential function computed by Qdrant
- **No data modification**: Original data remains unchanged
- **Query-time only**: Decay is applied during search, not storage

### Scalability

The implementation scales well:
- **O(1) per result**: Decay calculation is constant time
- **Native support**: Uses Qdrant's built-in formula engine
- **No additional storage**: No pre-computed decay values needed

### Performance Tips

1. **Batch operations**: Search multiple collections in parallel
2. **Index optimization**: Ensure timestamp fields are properly indexed
3. **Result limits**: Use appropriate `limit` values to reduce processing

## Use Cases

### When to Enable Memory Decay

Enable decay when:
- Working on active, evolving projects
- Searching for recent solutions and patterns
- Daily development workflows
- Troubleshooting current issues

### When to Disable Memory Decay

Disable decay when:
- Researching project history
- Analyzing long-term patterns
- Auditing past decisions
- Learning from historical implementations

## Future Enhancements

Potential improvements being considered:

1. **Adaptive decay rates**: Different half-lives for different content types
2. **Importance weighting**: Starred or marked conversations decay slower
3. **Project-specific decay**: Customize decay per project context
4. **Decay visualization**: Show how decay affects search results

## Technical References

- [Qdrant Decay Functions Documentation](https://qdrant.tech/documentation/concepts/search/#decay-functions)
- [Exponential Decay in Information Retrieval](https://en.wikipedia.org/wiki/Exponential_decay)
- [Time-Based Relevance in Search Systems](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html#function-decay)