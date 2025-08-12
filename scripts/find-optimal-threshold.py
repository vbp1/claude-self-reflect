#!/usr/bin/env python3
"""Find optimal similarity threshold for search quality."""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding
import argparse


@dataclass
class ThresholdTestResult:
    """Result of testing a specific threshold."""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    avg_results: float
    avg_latency_ms: float
    false_positives: int
    false_negatives: int
    
    @property
    def efficiency_score(self) -> float:
        """Combined score considering quality and performance."""
        # Weighted combination: F1 (60%), speed (20%), result count efficiency (20%)
        latency_score = 1.0 - min(self.avg_latency_ms / 1000, 1.0)  # Normalize to 0-1
        result_efficiency = 1.0 - abs(self.avg_results - 5) / 10  # Ideal is ~5 results
        
        return (self.f1_score * 0.6 + 
                latency_score * 0.2 + 
                result_efficiency * 0.2)


class OptimalThresholdFinder:
    """Find optimal similarity threshold for search."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = AsyncQdrantClient(url=qdrant_url)
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.model = TextEmbedding(model_name=embedding_model)
        self.results = []
        
    async def test_threshold(
        self,
        threshold: float,
        test_queries: List[Dict[str, Any]],
        collection_name: str = "conversations_local",
        limit: int = 20
    ) -> ThresholdTestResult:
        """Test a specific threshold value."""
        import time
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_results = 0
        total_latency = 0
        false_positives = 0
        false_negatives = 0
        
        for query_data in test_queries:
            query = query_data['query']
            expected = query_data.get('expected', [])
            
            # Generate embedding
            embedding = list(self.model.embed([query]))[0].tolist()
            
            # Search with threshold
            start_time = time.time()
            results = await self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit,
                score_threshold=threshold,
                with_payload=True
            )
            latency = (time.time() - start_time) * 1000
            
            # Calculate metrics
            retrieved_texts = [r.payload.get('content', '') if r.payload else '' for r in results]
            
            # Find true positives
            true_positives = 0
            for expected_text in expected:
                found = False
                for retrieved in retrieved_texts:
                    if expected_text.lower() in retrieved.lower():
                        true_positives += 1
                        found = True
                        break
                if not found:
                    false_negatives += 1
            
            # Calculate false positives (results above threshold but not expected)
            fp = max(0, len(results) - true_positives)
            false_positives += fp
            
            # Calculate metrics
            precision = true_positives / len(results) if results else 0
            recall = true_positives / len(expected) if expected else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_results += len(results)
            total_latency += latency
        
        num_queries = len(test_queries)
        
        result = ThresholdTestResult(
            threshold=threshold,
            precision=total_precision / num_queries,
            recall=total_recall / num_queries,
            f1_score=total_f1 / num_queries,
            avg_results=total_results / num_queries,
            avg_latency_ms=total_latency / num_queries,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
        
        self.results.append(result)
        return result
    
    async def find_optimal(
        self,
        test_queries: List[Dict[str, Any]],
        threshold_range: Tuple[float, float] = (0.3, 0.95),
        step: float = 0.05,
        collection_name: str = "conversations_local"
    ) -> ThresholdTestResult:
        """Find optimal threshold using grid search."""
        
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        
        print(f"Testing {len(thresholds)} threshold values from {threshold_range[0]} to {threshold_range[1]}")
        print("="*60)
        
        for threshold in thresholds:
            result = await self.test_threshold(
                threshold=float(threshold),
                test_queries=test_queries,
                collection_name=collection_name
            )
            
            print(f"Threshold {threshold:.2f}: F1={result.f1_score:.3f}, "
                  f"Precision={result.precision:.3f}, Recall={result.recall:.3f}, "
                  f"Avg Results={result.avg_results:.1f}")
        
        # Find optimal based on efficiency score
        optimal = max(self.results, key=lambda r: r.efficiency_score)
        
        print("\n" + "="*60)
        print(f"OPTIMAL THRESHOLD: {optimal.threshold:.2f}")
        print("="*60)
        print(f"F1 Score: {optimal.f1_score:.3f}")
        print(f"Precision: {optimal.precision:.3f}")
        print(f"Recall: {optimal.recall:.3f}")
        print(f"Avg Results: {optimal.avg_results:.1f}")
        print(f"Avg Latency: {optimal.avg_latency_ms:.1f}ms")
        print(f"Efficiency Score: {optimal.efficiency_score:.3f}")
        
        return optimal
    
    async def adaptive_search(
        self,
        test_queries: List[Dict[str, Any]],
        initial_range: Tuple[float, float] = (0.3, 0.95),
        tolerance: float = 0.01
    ) -> ThresholdTestResult:
        """Use adaptive search to find optimal threshold more efficiently."""
        
        print("Starting adaptive threshold search...")
        print("="*60)
        
        # Coarse search
        print("\nPhase 1: Coarse search (step=0.1)")
        coarse_step = 0.1
        coarse_optimal = await self.find_optimal(
            test_queries=test_queries,
            threshold_range=initial_range,
            step=coarse_step
        )
        
        # Clear results for fine search
        self.results = []
        
        # Fine search around optimal
        fine_range = (
            max(initial_range[0], coarse_optimal.threshold - coarse_step),
            min(initial_range[1], coarse_optimal.threshold + coarse_step)
        )
        
        print(f"\nPhase 2: Fine search around {coarse_optimal.threshold:.2f} (step={tolerance})")
        fine_optimal = await self.find_optimal(
            test_queries=test_queries,
            threshold_range=fine_range,
            step=tolerance
        )
        
        return fine_optimal
    
    def plot_results(self, save_path: str = None):  # type: ignore
        """Plot threshold analysis results."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("matplotlib not installed. Skipping visualization.")
            print("Install with: pip install matplotlib")
            return
            
        if not self.results:
            print("No results to plot")
            return
        
        thresholds = [r.threshold for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        precisions = [r.precision for r in self.results]
        recalls = [r.recall for r in self.results]
        avg_results = [r.avg_results for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # F1 Score
        axes[0, 0].plot(thresholds, f1_scores, 'b-', marker='o')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[0, 1].plot(thresholds, precisions, 'g-', marker='s', label='Precision')
        axes[0, 1].plot(thresholds, recalls, 'r-', marker='^', label='Recall')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision & Recall vs Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Results
        axes[1, 0].plot(thresholds, avg_results, 'm-', marker='d')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Average Results')
        axes[1, 0].set_title('Average Result Count vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency Score
        efficiency_scores = [r.efficiency_score for r in self.results]
        axes[1, 1].plot(thresholds, efficiency_scores, 'c-', marker='*')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].set_title('Overall Efficiency vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark optimal threshold
        optimal = max(self.results, key=lambda r: r.efficiency_score)
        for ax in axes.flat:
            ax.axvline(x=optimal.threshold, color='red', linestyle='--', alpha=0.5, label=f'Optimal: {optimal.threshold:.2f}')
        
        plt.suptitle('Threshold Optimization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str = "threshold_optimization_results.json"):
        """Save optimization results to file."""
        if not self.results:
            print("No results to save")
            return
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'optimal_threshold': max(self.results, key=lambda r: r.efficiency_score).threshold,
            'results': [
                {
                    'threshold': r.threshold,
                    'f1_score': r.f1_score,
                    'precision': r.precision,
                    'recall': r.recall,
                    'avg_results': r.avg_results,
                    'avg_latency_ms': r.avg_latency_ms,
                    'efficiency_score': r.efficiency_score,
                    'false_positives': r.false_positives,
                    'false_negatives': r.false_negatives
                }
                for r in sorted(self.results, key=lambda x: x.threshold)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {filepath}")


# Default test queries for optimization
DEFAULT_TEST_QUERIES = [
    {
        'query': 'implement vector database with Qdrant',
        'expected': ['Qdrant', 'vector', 'database', 'implementation']
    },
    {
        'query': 'Python MCP server FastMCP',
        'expected': ['MCP', 'Python', 'FastMCP', 'server']
    },
    {
        'query': 'semantic search memory decay',
        'expected': ['semantic search', 'memory decay', 'exponential']
    },
    {
        'query': 'Docker compose configuration',
        'expected': ['Docker', 'compose', 'configuration']
    },
    {
        'query': 'embedding model comparison FastEmbed',
        'expected': ['embedding', 'FastEmbed', 'model', 'comparison']
    },
    {
        'query': 'cross-collection search implementation',
        'expected': ['cross-collection', 'search', 'implementation']
    },
    {
        'query': 'similarity threshold optimization',
        'expected': ['similarity', 'threshold', 'optimization']
    },
    {
        'query': 'conversation chunking strategy',
        'expected': ['conversation', 'chunking', 'strategy']
    }
]


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Find optimal similarity threshold')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection', default='conversations_local',
                        help='Collection name to search')
    parser.add_argument('--min-threshold', type=float, default=0.3,
                        help='Minimum threshold to test')
    parser.add_argument('--max-threshold', type=float, default=0.95,
                        help='Maximum threshold to test')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Step size for threshold testing')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive search (coarse then fine)')
    parser.add_argument('--test-file', type=str,
                        help='Load test queries from JSON file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--save-plot', type=str,
                        help='Save plot to specified file')
    parser.add_argument('--save-results', type=str,
                        default='threshold_optimization_results.json',
                        help='Save results to specified file')
    
    args = parser.parse_args()
    
    # Load test queries
    test_queries = DEFAULT_TEST_QUERIES
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_queries = json.load(f)
    
    finder = OptimalThresholdFinder(qdrant_url=args.qdrant_url)
    
    try:
        if args.adaptive:
            # Use adaptive search
            optimal = await finder.adaptive_search(
                test_queries=test_queries,
                initial_range=(args.min_threshold, args.max_threshold),
                tolerance=args.step / 10  # Fine step is 1/10 of coarse
            )
        else:
            # Use grid search
            optimal = await finder.find_optimal(
                test_queries=test_queries,
                threshold_range=(args.min_threshold, args.max_threshold),
                step=args.step,
                collection_name=args.collection
            )
        
        # Save results
        finder.save_results(args.save_results)
        
        # Generate plots if requested
        if args.plot or args.save_plot:
            try:
                finder.plot_results(save_path=args.save_plot)
            except ImportError:
                print("\nNote: matplotlib not installed. Skipping visualization.")
                print("Install with: pip install matplotlib")
        
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"Set SIMILARITY_THRESHOLD={optimal.threshold:.2f} in your configuration")
        print("This provides the best balance of precision, recall, and performance")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())