#!/usr/bin/env python3
"""Monitor search performance and collect metrics over time."""

import asyncio
import json
import time
import statistics
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import signal
import sys
import os

from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding
import argparse


@dataclass
class PerformanceMetric:
    """A single performance measurement."""
    timestamp: str
    query: str
    latency_ms: float
    embedding_time_ms: float
    search_time_ms: float
    result_count: int
    max_score: float
    min_score: float
    collection_name: str
    threshold: float
    error: Optional[str] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    period_start: str
    period_end: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_result_count: float
    avg_max_score: float
    queries_per_second: float
    error_rate: float
    
    def print_summary(self):
        """Print formatted summary of stats."""
        print("\n" + "="*60)
        print(f"PERFORMANCE SUMMARY ({self.period_start} to {self.period_end})")
        print("="*60)
        print(f"Total Queries: {self.total_queries}")
        print(f"Successful: {self.successful_queries}")
        print(f"Failed: {self.failed_queries}")
        print(f"Error Rate: {self.error_rate:.1%}")
        print("\nLatency Statistics:")
        print(f"  Average: {self.avg_latency_ms:.1f}ms")
        print(f"  P50: {self.p50_latency_ms:.1f}ms")
        print(f"  P95: {self.p95_latency_ms:.1f}ms")
        print(f"  P99: {self.p99_latency_ms:.1f}ms")
        print(f"  Min: {self.min_latency_ms:.1f}ms")
        print(f"  Max: {self.max_latency_ms:.1f}ms")
        print(f"\nThroughput: {self.queries_per_second:.2f} queries/sec")
        print(f"Avg Results: {self.avg_result_count:.1f}")
        print(f"Avg Max Score: {self.avg_max_score:.3f}")


class SearchPerformanceMonitor:
    """Monitor search performance over time."""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        window_size: int = 100
    ):
        self.client = AsyncQdrantClient(url=qdrant_url)
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.model = TextEmbedding(model_name=embedding_model)
        self.metrics = deque(maxlen=window_size)
        self.all_metrics = []
        self.running = False
        self.start_time = None
        
    async def measure_query_performance(
        self,
        query: str,
        collection_name: str = "conversations_local",
        threshold: float = 0.7,
        limit: int = 10
    ) -> PerformanceMetric:
        """Measure performance of a single query."""
        
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            query=query,
            latency_ms=0,
            embedding_time_ms=0,
            search_time_ms=0,
            result_count=0,
            max_score=0,
            min_score=0,
            collection_name=collection_name,
            threshold=threshold
        )
        
        try:
            # Measure embedding generation
            start = time.time()
            embedding = list(self.model.embed([query]))[0].tolist()
            embedding_time = (time.time() - start) * 1000
            metric.embedding_time_ms = embedding_time
            
            # Measure search
            start = time.time()
            results = await self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit,
                score_threshold=threshold,
                with_payload=False  # Faster without payload
            )
            search_time = (time.time() - start) * 1000
            metric.search_time_ms = search_time
            
            # Total latency
            metric.latency_ms = embedding_time + search_time
            metric.result_count = len(results)
            
            if results:
                metric.max_score = max(r.score for r in results)
                metric.min_score = min(r.score for r in results)
            
        except Exception as e:
            metric.error = str(e)
            print(f"Error in query '{query}': {e}")
        
        # Store metric
        self.metrics.append(metric)
        self.all_metrics.append(metric)
        
        return metric
    
    async def continuous_monitor(
        self,
        test_queries: List[str],
        interval_seconds: float = 1.0,
        collection_name: str = "conversations_local",
        threshold: float = 0.7
    ):
        """Continuously monitor performance with test queries."""
        
        self.running = True
        self.start_time = datetime.now()
        query_index = 0
        
        print(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        print("Press Ctrl+C to stop and see summary")
        print("-" * 60)
        
        while self.running:
            # Rotate through test queries
            query = test_queries[query_index % len(test_queries)]
            query_index += 1
            
            # Measure performance
            metric = await self.measure_query_performance(
                query=query,
                collection_name=collection_name,
                threshold=threshold
            )
            
            # Print real-time feedback
            if metric.error:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {metric.error}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Query #{query_index}: {metric.latency_ms:.1f}ms, "
                      f"{metric.result_count} results, "
                      f"max_score={metric.max_score:.3f}")
            
            # Print rolling stats every 10 queries
            if query_index % 10 == 0:
                self.print_rolling_stats()
            
            await asyncio.sleep(interval_seconds)
    
    def print_rolling_stats(self):
        """Print statistics for recent queries."""
        if len(self.metrics) < 2:
            return
        
        recent_metrics = list(self.metrics)
        successful = [m for m in recent_metrics if m.error is None]
        
        if successful:
            latencies = [m.latency_ms for m in successful]
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            
            print(f"\n  Rolling stats (last {len(recent_metrics)} queries):")
            print(f"    Avg latency: {avg_latency:.1f}ms")
            print(f"    P50 latency: {p50_latency:.1f}ms")
            print(f"    Success rate: {len(successful)/len(recent_metrics):.1%}\n")
    
    def calculate_stats(self, metrics: Optional[List[PerformanceMetric]] = None) -> Optional[PerformanceStats]:
        """Calculate statistics from metrics."""
        
        if metrics is None:
            metrics = self.all_metrics
        
        if not metrics:
            return None
        
        successful = [m for m in metrics if m.error is None]
        failed = [m for m in metrics if m.error is not None]
        
        # Calculate time range
        timestamps = [datetime.fromisoformat(m.timestamp) for m in metrics]
        period_start = min(timestamps)
        period_end = max(timestamps)
        duration = (period_end - period_start).total_seconds()
        
        # Calculate latency percentiles
        if successful:
            latencies = sorted([m.latency_ms for m in successful])
            p50_idx = int(len(latencies) * 0.5)
            p95_idx = int(len(latencies) * 0.95)
            p99_idx = int(len(latencies) * 0.99)
            
            stats = PerformanceStats(
                period_start=period_start.isoformat(),
                period_end=period_end.isoformat(),
                total_queries=len(metrics),
                successful_queries=len(successful),
                failed_queries=len(failed),
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=latencies[p50_idx] if p50_idx < len(latencies) else latencies[-1],
                p95_latency_ms=latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1],
                p99_latency_ms=latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1],
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                avg_result_count=statistics.mean([m.result_count for m in successful]),
                avg_max_score=statistics.mean([m.max_score for m in successful if m.max_score > 0]),
                queries_per_second=len(metrics) / duration if duration > 0 else 0,
                error_rate=len(failed) / len(metrics)
            )
        else:
            # All queries failed
            stats = PerformanceStats(
                period_start=period_start.isoformat(),
                period_end=period_end.isoformat(),
                total_queries=len(metrics),
                successful_queries=0,
                failed_queries=len(failed),
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                avg_result_count=0,
                avg_max_score=0,
                queries_per_second=len(metrics) / duration if duration > 0 else 0,
                error_rate=1.0
            )
        
        return stats
    
    async def benchmark(
        self,
        test_queries: List[str],
        iterations: int = 100,
        collection_name: str = "conversations_local",
        threshold: float = 0.7,
        concurrent: bool = False,
        concurrency: int = 10
    ) -> Optional[PerformanceStats]:
        """Run a benchmark test."""
        
        print(f"Running benchmark: {iterations} iterations")
        print(f"Concurrent: {concurrent} (concurrency={concurrency if concurrent else 1})")
        print("-" * 60)
        
        self.all_metrics = []
        start_time = time.time()
        
        if concurrent:
            # Concurrent benchmark
            tasks = []
            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                task = self.measure_query_performance(
                    query=query,
                    collection_name=collection_name,
                    threshold=threshold
                )
                tasks.append(task)
                
                # Control concurrency
                if len(tasks) >= concurrency:
                    await asyncio.gather(*tasks)
                    tasks = []
                    print(f"Completed {i+1}/{iterations} queries")
            
            # Complete remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
        else:
            # Sequential benchmark
            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                await self.measure_query_performance(
                    query=query,
                    collection_name=collection_name,
                    threshold=threshold
                )
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i+1}/{iterations} queries")
        
        total_time = time.time() - start_time
        
        # Calculate and print stats
        stats = self.calculate_stats()
        if stats:
            stats.print_summary()
        
        print(f"\nTotal benchmark time: {total_time:.1f}s")
        
        return stats
    
    def save_metrics(self, filepath: str = "search_performance_metrics.json"):
        """Save metrics to file."""
        
        stats = self.calculate_stats()
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total_metrics': len(self.all_metrics),
            'metrics': [asdict(m) for m in self.all_metrics],
            'statistics': asdict(stats) if stats else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nMetrics saved to {filepath}")
    
    def export_csv(self, filepath: str = "search_performance_metrics.csv"):
        """Export metrics to CSV for analysis."""
        import csv
        
        if not self.all_metrics:
            print("No metrics to export")
            return
        
        with open(filepath, 'w', newline='') as f:  # type: ignore
            fieldnames = [
                'timestamp', 'query', 'latency_ms', 'embedding_time_ms',
                'search_time_ms', 'result_count', 'max_score', 'min_score',
                'collection_name', 'threshold', 'error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in self.all_metrics:
                writer.writerow(asdict(metric))
        
        print(f"Metrics exported to {filepath}")


# Default test queries
DEFAULT_TEST_QUERIES = [
    "vector database implementation",
    "MCP server Python FastMCP",
    "semantic search with embeddings",
    "Docker compose configuration",
    "memory decay exponential",
    "Qdrant collection management",
    "conversation chunking strategy",
    "similarity threshold optimization",
    "cross-collection search",
    "FastEmbed model comparison"
]


def signal_handler(monitor: SearchPerformanceMonitor):
    """Handle Ctrl+C gracefully."""
    def handler(signum, frame):  # type: ignore
        print("\n\nStopping monitor...")
        monitor.running = False
        
        # Print final stats
        if monitor.all_metrics:
            stats = monitor.calculate_stats()
            if stats:
                stats.print_summary()
            
            # Save metrics
            monitor.save_metrics()
            monitor.export_csv()
        
        sys.exit(0)
    
    return handler


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Monitor search performance')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection', default='conversations_local',
                        help='Collection name to search')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Similarity threshold')
    parser.add_argument('--mode', choices=['continuous', 'benchmark'],
                        default='benchmark',
                        help='Monitoring mode')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Interval between queries in continuous mode (seconds)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for benchmark mode')
    parser.add_argument('--concurrent', action='store_true',
                        help='Run benchmark queries concurrently')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent queries')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Rolling window size for metrics')
    parser.add_argument('--test-file', type=str,
                        help='Load test queries from file')
    parser.add_argument('--save-metrics', type=str,
                        default='search_performance_metrics.json',
                        help='Save metrics to file')
    parser.add_argument('--export-csv', type=str,
                        help='Export metrics to CSV')
    
    args = parser.parse_args()
    
    # Load test queries
    test_queries = DEFAULT_TEST_QUERIES
    if args.test_file:
        with open(args.test_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and isinstance(data[0], str):
                test_queries = data
            elif isinstance(data, list) and isinstance(data[0], dict):
                test_queries = [item['query'] for item in data]
    
    monitor = SearchPerformanceMonitor(
        qdrant_url=args.qdrant_url,
        window_size=args.window_size
    )
    
    try:
        if args.mode == 'continuous':
            # Set up signal handler for graceful shutdown
            signal.signal(signal.SIGINT, signal_handler(monitor))
            
            # Run continuous monitoring
            await monitor.continuous_monitor(
                test_queries=test_queries,
                interval_seconds=args.interval,
                collection_name=args.collection,
                threshold=args.threshold
            )
        else:
            # Run benchmark
            stats = await monitor.benchmark(
                test_queries=test_queries,
                iterations=args.iterations,
                collection_name=args.collection,
                threshold=args.threshold,
                concurrent=args.concurrent,
                concurrency=args.concurrency
            )
            
            # Save results
            monitor.save_metrics(args.save_metrics)
            
            if args.export_csv:
                monitor.export_csv(args.export_csv)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if monitor.all_metrics:
            stats = monitor.calculate_stats()
            if stats:
                stats.print_summary()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())