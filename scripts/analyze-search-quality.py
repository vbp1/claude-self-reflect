#!/usr/bin/env python3
"""Analyze search quality with precision, recall, and F1 metrics."""

import asyncio
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding
import argparse


@dataclass
class SearchMetrics:
    """Metrics for search quality evaluation."""
    query: str
    precision: float
    recall: float
    f1_score: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    avg_score: float
    result_count: int
    threshold_used: float
    latency_ms: float
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TestCase:
    """A test case for search quality evaluation."""
    query: str
    expected_results: List[str]  # Expected content snippets
    context: Optional[str] = None
    project: Optional[str] = None


class SearchQualityAnalyzer:
    """Analyze search quality metrics."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = AsyncQdrantClient(url=qdrant_url)
        self.model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.metrics_history = []
        
    async def analyze_query(
        self, 
        query: str, 
        expected_results: List[str],
        collection_name: str = "conversations_local",
        threshold: float = 0.7,
        limit: int = 10,
        apply_decay: bool = False
    ) -> SearchMetrics:
        """Analyze search quality for a single query."""
        import time
        
        # Generate embedding
        embedding = list(self.model.embed([query]))[0].tolist()
        
        # Measure search latency
        start_time = time.time()
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit,
            score_threshold=threshold,
            with_payload=True
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Apply decay if requested
        if apply_decay:
            results = self._apply_decay(results)
        
        # Calculate metrics
        retrieved_texts = [r.payload.get('content', '') if r.payload else '' for r in results]
        
        # Find matches (simple substring matching, can be improved)
        true_positives = 0
        for expected in expected_results:
            for retrieved in retrieved_texts[:len(expected_results)]:
                if expected.lower() in retrieved.lower():
                    true_positives += 1
                    break
        
        precision = true_positives / len(results) if results else 0
        recall = true_positives / len(expected_results) if expected_results else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(expected_results, retrieved_texts)
        
        # Calculate NDCG
        ndcg = self._calculate_ndcg(expected_results, retrieved_texts)
        
        # Average score of results
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        metrics = SearchMetrics(
            query=query,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mrr=mrr,
            ndcg=ndcg,
            avg_score=avg_score,
            result_count=len(results),
            threshold_used=threshold,
            latency_ms=latency_ms
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _apply_decay(self, results):
        """Apply time-based decay to search results."""
        decayed_results = []
        for result in results:
            if 'timestamp' in result.payload:
                try:
                    timestamp = datetime.fromisoformat(result.payload['timestamp'])
                    days_old = (datetime.now() - timestamp).days
                    decay_factor = math.exp(-0.00771 * days_old)  # 90-day half-life
                    result.score *= decay_factor
                except (ValueError, KeyError, AttributeError):
                    pass  # Keep original score if timestamp parsing fails
            decayed_results.append(result)
        
        # Re-sort by decayed scores
        return sorted(decayed_results, key=lambda x: x.score, reverse=True)
    
    def _calculate_mrr(self, expected: List[str], retrieved: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, retrieved_text in enumerate(retrieved):
            for expected_text in expected:
                if expected_text.lower() in retrieved_text.lower():
                    return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, expected: List[str], retrieved: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        def dcg(relevances: List[int], k: int) -> float:
            """Calculate DCG@k."""
            dcg_val = 0.0
            for i, rel in enumerate(relevances[:k]):
                dcg_val += rel / math.log2(i + 2)  # i+2 because index starts at 0
            return dcg_val
        
        # Create relevance scores (1 if match, 0 otherwise)
        relevances = []
        for retrieved_text in retrieved[:k]:
            rel = 0
            for expected_text in expected:
                if expected_text.lower() in retrieved_text.lower():
                    rel = 1
                    break
            relevances.append(rel)
        
        # Ideal relevance (all 1s for the number of expected results)
        ideal_relevances = [1] * min(len(expected), k) + [0] * (k - min(len(expected), k))
        
        dcg_val = dcg(relevances, k)
        idcg_val = dcg(ideal_relevances, k)
        
        return dcg_val / idcg_val if idcg_val > 0 else 0
    
    async def run_test_suite(self, test_cases: List[TestCase], **search_params) -> Dict[str, Any]:
        """Run a suite of test cases and aggregate metrics."""
        all_metrics = []
        
        for test_case in test_cases:
            print(f"\nTesting query: {test_case.query}")
            metrics = await self.analyze_query(
                query=test_case.query,
                expected_results=test_case.expected_results,
                **search_params
            )
            all_metrics.append(metrics)
            
            print(f"  Precision: {metrics.precision:.2%}")
            print(f"  Recall: {metrics.recall:.2%}")
            print(f"  F1 Score: {metrics.f1_score:.2%}")
            print(f"  MRR: {metrics.mrr:.3f}")
            print(f"  NDCG: {metrics.ndcg:.3f}")
            print(f"  Latency: {metrics.latency_ms:.1f}ms")
        
        # Aggregate metrics
        aggregated = {
            'total_queries': len(all_metrics),
            'avg_precision': sum(m.precision for m in all_metrics) / len(all_metrics),
            'avg_recall': sum(m.recall for m in all_metrics) / len(all_metrics),
            'avg_f1_score': sum(m.f1_score for m in all_metrics) / len(all_metrics),
            'avg_mrr': sum(m.mrr for m in all_metrics) / len(all_metrics) if all_metrics else 0,
            'avg_ndcg': sum(m.ndcg for m in all_metrics) / len(all_metrics) if all_metrics else 0,
            'avg_latency_ms': sum(m.latency_ms for m in all_metrics) / len(all_metrics),
            'avg_result_count': sum(m.result_count for m in all_metrics) / len(all_metrics),
            'threshold_used': search_params.get('threshold', 0.7),
            'timestamp': datetime.now().isoformat()
        }
        
        return aggregated
    
    def save_metrics(self, filepath: str = "search_quality_metrics.json"):
        """Save metrics history to file."""
        output_path = Path(filepath)
        with open(output_path, 'w') as f:
            json.dump(
                [asdict(m) for m in self.metrics_history],
                f,
                indent=2,
                default=str
            )
        print(f"\nMetrics saved to {output_path}")
    
    async def compare_thresholds(
        self, 
        test_cases: List[TestCase],
        thresholds: Optional[List[float]] = None
    ) -> Dict[float, Dict[str, Any]]:
        """Compare search quality across different similarity thresholds."""
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {}
        for threshold in thresholds:
            print(f"\n{'='*50}")
            print(f"Testing threshold: {threshold}")
            print('='*50)
            
            aggregated = await self.run_test_suite(
                test_cases,
                threshold=threshold
            )
            results[threshold] = aggregated
        
        # Find optimal threshold
        best_threshold = max(results.keys(), key=lambda k: results[k]['avg_f1_score'])
        
        print(f"\n{'='*50}")
        print("THRESHOLD COMPARISON RESULTS")
        print('='*50)
        
        for threshold, metrics in sorted(results.items()):
            marker = " <-- BEST" if threshold == best_threshold else ""
            print(f"\nThreshold {threshold}{marker}:")
            print(f"  Avg F1 Score: {metrics['avg_f1_score']:.2%}")
            print(f"  Avg Precision: {metrics['avg_precision']:.2%}")
            print(f"  Avg Recall: {metrics['avg_recall']:.2%}")
            print(f"  Avg MRR: {metrics['avg_mrr']:.3f}")
            print(f"  Avg Results: {metrics['avg_result_count']:.1f}")
        
        return results


# Default test cases for Claude Self Reflect
DEFAULT_TEST_CASES = [
    TestCase(
        query="vector database migration from Neo4j to Qdrant",
        expected_results=[
            "Neo4j", "Qdrant", "migration", "vector database"
        ]
    ),
    TestCase(
        query="how to implement memory decay",
        expected_results=[
            "memory decay", "exponential decay", "90-day half-life"
        ]
    ),
    TestCase(
        query="MCP server Python implementation",
        expected_results=[
            "MCP", "FastMCP", "Python", "server"
        ]
    ),
    TestCase(
        query="semantic search with embeddings",
        expected_results=[
            "semantic search", "embeddings", "FastEmbed", "all-MiniLM-L6-v2"
        ]
    ),
    TestCase(
        query="Docker compose setup for Qdrant",
        expected_results=[
            "Docker", "compose", "Qdrant", "container"
        ]
    )
]


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze search quality metrics')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection', default='conversations_local',
                        help='Collection name to search')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Similarity threshold for search')
    parser.add_argument('--limit', type=int, default=10,
                        help='Maximum number of results to return')
    parser.add_argument('--compare-thresholds', action='store_true',
                        help='Compare different similarity thresholds')
    parser.add_argument('--apply-decay', action='store_true',
                        help='Apply time-based decay to results')
    parser.add_argument('--save-metrics', type=str,
                        help='Save metrics to specified file')
    parser.add_argument('--test-file', type=str,
                        help='Load test cases from JSON file')
    
    args = parser.parse_args()
    
    analyzer = SearchQualityAnalyzer(qdrant_url=args.qdrant_url)
    
    # Load test cases
    test_cases = DEFAULT_TEST_CASES
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
            test_cases = [TestCase(**tc) for tc in test_data]
    
    try:
        if args.compare_thresholds:
            # Compare different thresholds
            await analyzer.compare_thresholds(test_cases)
        else:
            # Run with single threshold
            aggregated = await analyzer.run_test_suite(
                test_cases,
                collection_name=args.collection,
                threshold=args.threshold,
                limit=args.limit,
                apply_decay=args.apply_decay
            )
            
            print(f"\n{'='*50}")
            print("AGGREGATED METRICS")
            print('='*50)
            print(f"Total Queries: {aggregated['total_queries']}")
            print(f"Avg Precision: {aggregated['avg_precision']:.2%}")
            print(f"Avg Recall: {aggregated['avg_recall']:.2%}")
            print(f"Avg F1 Score: {aggregated['avg_f1_score']:.2%}")
            print(f"Avg MRR: {aggregated['avg_mrr']:.3f}")
            print(f"Avg NDCG: {aggregated['avg_ndcg']:.3f}")
            print(f"Avg Latency: {aggregated['avg_latency_ms']:.1f}ms")
            print(f"Avg Results: {aggregated['avg_result_count']:.1f}")
        
        # Save metrics if requested
        if args.save_metrics:
            analyzer.save_metrics(args.save_metrics)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())