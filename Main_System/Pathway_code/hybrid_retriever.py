"""
Hybrid Retriever: Combines Pathway semantic search with BM25 keyword search

This module provides hybrid retrieval (semantic + keyword) by:
1. Querying Pathway server for semantic results
2. Running BM25 keyword search on indexed documents
3. Combining results using Reciprocal Rank Fusion (RRF)
"""

import numpy as np
import requests
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Hybrid retriever combining Pathway semantic search with BM25 keyword search.
    """
    
    def __init__(
        self,
        pathway_url: str = "http://127.0.0.1:8745",
        bm25_index: Optional[BM25Okapi] = None,
        text_chunks: Optional[List[str]] = None,
        doc_id_mapping: Optional[Dict[int, str]] = None,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            pathway_url: URL of Pathway server
            bm25_index: Pre-built BM25 index (optional, will build if not provided)
            text_chunks: List of text chunks for BM25 (required if bm25_index not provided)
            doc_id_mapping: Mapping from array index to Pathway doc_id (optional)
            semantic_weight: Weight for semantic scores (0.0-1.0)
            keyword_weight: Weight for keyword scores (0.0-1.0)
            rrf_k: RRF constant (higher = less weight on rank differences)
        """
        self.pathway_url = pathway_url.rstrip('/')
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        
        # BM25 components
        self.bm25_index = bm25_index
        self.text_chunks = text_chunks or []
        self.doc_id_mapping = doc_id_mapping or {}
        
        # Build BM25 index if text chunks provided but index not provided
        if self.text_chunks and self.bm25_index is None:
            self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from text chunks."""
        if not self.text_chunks:
            raise ValueError("Cannot build BM25 index: no text chunks provided")
        
        # Tokenize texts for BM25
        tokenized_texts = [text.lower().split() for text in self.text_chunks]
        self.bm25_index = BM25Okapi(tokenized_texts)
        print(f"✓ Built BM25 index with {len(self.text_chunks)} documents", flush=True)
    
    def build_bm25_from_pathway(self, max_samples: int = 10000):
        """
        Build BM25 index by sampling documents from Pathway server.
        
        This is a workaround since Pathway doesn't expose all indexed documents directly.
        We sample by querying with diverse queries and collecting unique results.
        
        Args:
            max_samples: Maximum number of documents to sample
        """
        print("Building BM25 index from Pathway server (sampling documents)...", flush=True)
        
        # Sample queries to get diverse documents
        sample_queries = [
            "character", "story", "event", "time", "place", "action",
            "said", "went", "came", "saw", "knew", "thought", "felt",
            "first", "then", "after", "before", "during", "when",
            "where", "who", "what", "why", "how"
        ]
        
        all_texts = set()
        all_results = []
        
        # Query Pathway with diverse queries to collect documents
        for query in sample_queries:
            try:
                response = requests.post(
                    f"{self.pathway_url}/v1/retrieve",
                    json={"query": query, "k": 100},  # Get many results per query
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                response.raise_for_status()
                results = response.json()
                
                if "results" in results:
                    for result in results["results"]:
                        if "text" in result:
                            text = result["text"]
                            # Use text as key to avoid duplicates
                            if text not in all_texts:
                                all_texts.add(text)
                                all_results.append({
                                    "text": text,
                                    "metadata": result.get("metadata", {}),
                                    "score": result.get("score", 0.0)
                                })
                                
                                if len(all_results) >= max_samples:
                                    break
                
                if len(all_results) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"  Warning: Error sampling with query '{query}': {e}", flush=True)
                continue
        
        if not all_results:
            raise ValueError("Could not sample any documents from Pathway server. Is the server running?")
        
        # Store texts and build BM25 index
        self.text_chunks = [r["text"] for r in all_results]
        self.doc_id_mapping = {i: f"pathway_doc_{i}" for i in range(len(all_results))}
        
        self._build_bm25_index()
        print(f"✓ Built BM25 index from {len(self.text_chunks)} sampled documents", flush=True)
        
        return len(self.text_chunks)
    
    def query_semantic(self, query: str, k: int = 15) -> List[Dict]:
        """
        Query Pathway server for semantic search results.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of results with 'text', 'score', 'metadata' keys
        """
        try:
            response = requests.post(
                f"{self.pathway_url}/v1/retrieve",
                json={"query": query, "k": k},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            
            # Handle different response formats
            if "results" in results:
                return results["results"]
            elif isinstance(results, list):
                return results
            return []
        except Exception as e:
            print(f"  Warning: Pathway semantic search failed: {e}", flush=True)
            return []
    
    def query_bm25(self, query: str, k: int = 15) -> List[Dict]:
        """
        Query BM25 index for keyword search results.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of results with 'text', 'score', 'index' keys
        """
        if self.bm25_index is None:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            if idx < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[idx],
                    "score": float(scores[idx]),
                    "index": int(idx),
                    "doc_id": self.doc_id_mapping.get(idx, f"bm25_doc_{idx}")
                })
        
        return results
    
    def rrf_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 15
    ) -> List[Dict]:
        """
        Combine semantic and BM25 results using Reciprocal Rank Fusion (RRF).
        
        RRF doesn't require score normalization and works well with different score ranges.
        RRF_score(doc) = sum(1 / (k + rank)) across all result sets
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            k: RRF constant (default 60)
            
        Returns:
            Combined and ranked results
        """
        # Create text-to-result mapping to handle duplicates
        text_to_result = {}
        rrf_scores = {}
        
        # Add semantic ranks
        for rank, result in enumerate(semantic_results, 1):
            text = result.get("text", "")
            if text:
                rrf_score = 1.0 / (self.rrf_k + rank)
                rrf_scores[text] = rrf_scores.get(text, 0.0) + self.semantic_weight * rrf_score
                if text not in text_to_result:
                    text_to_result[text] = result
        
        # Add BM25 ranks
        for rank, result in enumerate(bm25_results, 1):
            text = result.get("text", "")
            if text:
                rrf_score = 1.0 / (self.rrf_k + rank)
                rrf_scores[text] = rrf_scores.get(text, 0.0) + self.keyword_weight * rrf_score
                if text not in text_to_result:
                    text_to_result[text] = result
        
        # Sort by RRF score
        sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        final_results = []
        for text, score in sorted_texts[:k]:
            result = text_to_result[text].copy()
            result["hybrid_score"] = score
            final_results.append(result)
        
        return final_results
    
    def weighted_combination(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 15
    ) -> List[Dict]:
        """
        Combine semantic and BM25 results using weighted linear combination.
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            k: Number of final results
            
        Returns:
            Combined and ranked results
        """
        # Normalize scores to [0, 1]
        semantic_max = max(r.get("score", 0.0) for r in semantic_results) if semantic_results else 1.0
        bm25_max = max(r.get("score", 0.0) for r in bm25_results) if bm25_results else 1.0
        
        # Create text-to-score mapping
        combined_scores = {}
        text_to_result = {}
        
        # Add semantic scores (normalized)
        for result in semantic_results:
            text = result.get("text", "")
            if text:
                normalized_score = result.get("score", 0.0) / semantic_max if semantic_max > 0 else 0.0
                combined_scores[text] = self.semantic_weight * normalized_score
                text_to_result[text] = result
        
        # Add BM25 scores (normalized)
        for result in bm25_results:
            text = result.get("text", "")
            if text:
                normalized_score = result.get("score", 0.0) / bm25_max if bm25_max > 0 else 0.0
                combined_scores[text] = combined_scores.get(text, 0.0) + self.keyword_weight * normalized_score
                if text not in text_to_result:
                    text_to_result[text] = result
        
        # Sort by combined score
        sorted_texts = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        final_results = []
        for text, score in sorted_texts[:k]:
            result = text_to_result[text].copy()
            result["hybrid_score"] = score
            final_results.append(result)
        
        return final_results
    
    def retrieve(
        self,
        query: str,
        k: int = 15,
        use_rrf: bool = True,
        character: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Search query
            k: Number of results to return
            use_rrf: If True, use RRF fusion; else use weighted combination
            character: Optional character name for boosting (future enhancement)
            
        Returns:
            List of retrieved documents with combined scores
        """
        # Get more candidates from each method for better fusion
        candidate_k = k * 2
        
        # Query both methods
        semantic_results = self.query_semantic(query, k=candidate_k)
        bm25_results = self.query_bm25(query, k=candidate_k) if self.bm25_index else []
        
        # Fallback to semantic only if BM25 not available
        if not bm25_results:
            return semantic_results[:k]
        
        # Combine results
        if use_rrf:
            combined_results = self.rrf_fusion(semantic_results, bm25_results, k=k)
        else:
            combined_results = self.weighted_combination(semantic_results, bm25_results, k=k)
        
        # Character boosting (optional enhancement)
        if character:
            character_lower = character.lower()
            for result in combined_results:
                text = result.get("text", "").lower()
                if character_lower in text:
                    mention_count = text.count(character_lower)
                    boost = min(mention_count * 0.1, 0.3)  # Up to 30% boost
                    result["hybrid_score"] = result.get("hybrid_score", 0.0) * (1 + boost)
            
            # Re-sort after boosting
            combined_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        
        return combined_results[:k]
    
    def get_texts(self) -> List[str]:
        """Get all indexed text chunks."""
        return self.text_chunks.copy()

