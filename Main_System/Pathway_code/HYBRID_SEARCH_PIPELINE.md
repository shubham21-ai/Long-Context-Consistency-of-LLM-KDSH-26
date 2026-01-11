# Hybrid Search Pipeline: Pathway + BM25

## Overview

Pathway's `VectorStoreServer` only supports semantic search (USearch). To add hybrid search (semantic + keyword/BM25), we need to implement a custom pipeline that combines both.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Request                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  1. Extract Query Text       │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼
┌───────────────┐            ┌──────────────────┐
│ 2a. Semantic  │            │  2b. BM25        │
│    Search     │            │   Keyword Search │
│  (Pathway/    │            │  (Custom Layer)  │
│   USearch)    │            │                  │
└───────┬───────┘            └────────┬─────────┘
        │                              │
        │ Top-K results                │ Top-K results
        │ (doc_ids + scores)           │ (doc_ids + scores)
        │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  3. Score Fusion/Combination │
        │     - Normalize scores       │
        │     - Weighted combination   │
        │     - Rank fusion (RRF)      │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  4. Re-rank & Merge Results  │
        │     - Deduplicate by doc_id  │
        │     - Sort by combined score │
        │     - Take top-K final       │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  5. Fetch Full Documents     │
        │     - Get text chunks        │
        │     - Get metadata           │
        └──────────────┬───────────────┘
                       │
                       ▼
              Final Results (Top-K)
```

## Implementation Approaches

### Approach 1: Custom Retrieval Endpoint (Recommended) ⭐

**Add a new endpoint to Pathway server that combines semantic + BM25:**

```python
# In server.py or custom wrapper

class HybridVectorStoreServer(VectorStoreServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build BM25 index from all indexed documents
        self.bm25_index = None  # Built after indexing
        self.text_chunks = []   # Store text for BM25
    
    def build_bm25_index(self):
        """Build BM25 index from all indexed chunks"""
        from rank_bm25 import BM25Okapi
        
        # Get all indexed documents (extract from Pathway table)
        # Tokenize texts
        tokenized_texts = [text.lower().split() for text in self.text_chunks]
        self.bm25_index = BM25Okapi(tokenized_texts)
    
    def hybrid_retrieve(self, query: str, k: int = 15, 
                       semantic_weight: float = 0.5,
                       keyword_weight: float = 0.5):
        """
        Hybrid retrieval combining semantic + BM25
        
        Steps:
        1. Get semantic results from Pathway (USearch)
        2. Get BM25 results (keyword search)
        3. Combine scores with weights
        4. Return top-K
        """
        # Step 1: Semantic search via Pathway's existing endpoint
        semantic_results = self.query_semantic(query, k*2)  # Get more candidates
        
        # Step 2: BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]  # Top candidates
        
        # Step 3: Combine scores
        combined_scores = {}
        
        # Add semantic scores
        for doc_id, score in semantic_results:
            combined_scores[doc_id] = semantic_weight * score
        
        # Add BM25 scores (normalized)
        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1
        for idx in bm25_indices:
            doc_id = self.doc_id_from_idx(idx)
            normalized_score = bm25_scores[idx] / max_bm25
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + keyword_weight * normalized_score
        
        # Step 4: Sort and return top-K
        top_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_docs
```

**New HTTP Endpoint:**
```python
# Add to server.py's run_server method

@app.post("/v1/hybrid-retrieve")
def hybrid_retrieve(request: HybridRetrieveRequest):
    """Hybrid search endpoint (semantic + BM25)"""
    results = server.hybrid_retrieve(
        query=request.query,
        k=request.k,
        semantic_weight=0.5,
        keyword_weight=0.5
    )
    return results
```

### Approach 2: Post-Processing Layer (Simpler) ⭐⭐

**Keep Pathway server as-is, add BM25 layer in test_framework.py:**

```python
# In test_framework.py or separate module

class HybridRetriever:
    def __init__(self, pathway_server_url, bm25_index=None):
        self.pathway_url = pathway_server_url
        self.bm25_index = bm25_index  # Built separately
    
    def retrieve(self, query: str, k: int = 15):
        # Step 1: Get semantic results from Pathway
        semantic_results = self.query_pathway(query, k*2)
        
        # Step 2: Get BM25 results
        bm25_results = self.query_bm25(query, k*2)
        
        # Step 3: Combine and re-rank
        combined = self.combine_results(semantic_results, bm25_results, k)
        return combined
```

### Approach 3: Custom Pathway UDF (Advanced)

**Create custom Pathway UDF that includes BM25 in the graph:**

```python
@pw.udf
def hybrid_search_score(query: str, text: str, semantic_score: float) -> float:
    """Custom UDF combining semantic + BM25 scores"""
    # BM25 calculation
    bm25_score = calculate_bm25(query, text)
    
    # Combine (requires normalization)
    combined = 0.5 * normalize(semantic_score) + 0.5 * normalize(bm25_score)
    return combined
```

## Detailed Pipeline Steps

### Step 1: Build BM25 Index

**When:** After Pathway indexes documents

**How:**
```python
def build_bm25_index(self):
    """
    Build BM25 index from Pathway's indexed documents
    
    Challenge: Need to extract all indexed text chunks from Pathway
    Solution: 
    - Store chunks in a separate list during indexing
    - Or query Pathway's internal state (if accessible)
    - Or maintain parallel index during Pathway indexing
    """
    from rank_bm25 import BM25Okapi
    
    # Extract all text chunks (need to sync with Pathway's indexing)
    all_texts = self.get_all_indexed_texts()  # Custom method needed
    
    # Tokenize
    tokenized_texts = [text.lower().split() for text in all_texts]
    
    # Build BM25 index
    self.bm25_index = BM25Okapi(tokenized_texts)
    self.text_to_doc_id = {...}  # Map text to Pathway doc_id
```

### Step 2: Semantic Search (Pathway)

```python
# Use Pathway's existing /v1/retrieve endpoint
response = requests.post(
    f"{pathway_url}/v1/retrieve",
    json={"query": query, "k": k*2}  # Get more candidates
)
semantic_results = response.json()["results"]
# Returns: [{"text": "...", "score": 0.85, "metadata": {...}}, ...]
```

### Step 3: BM25 Keyword Search

```python
def bm25_search(self, query: str, k: int):
    """Run BM25 search on indexed texts"""
    tokenized_query = query.lower().split()
    
    # Get BM25 scores for all documents
    scores = self.bm25_index.get_scores(tokenized_query)
    
    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][:k]
    
    # Map indices to doc_ids and return
    results = []
    for idx in top_indices:
        doc_id = self.idx_to_doc_id[idx]
        results.append({
            "doc_id": doc_id,
            "score": scores[idx],
            "text": self.text_chunks[idx]
        })
    return results
```

### Step 4: Score Fusion

**Options:**

#### 4a. Weighted Linear Combination
```python
def combine_scores(semantic_results, bm25_results, 
                  semantic_weight=0.5, keyword_weight=0.5):
    """Simple weighted combination"""
    combined = {}
    
    # Normalize semantic scores (0-1)
    max_semantic = max(r["score"] for r in semantic_results) if semantic_results else 1
    
    # Normalize BM25 scores (0-1)
    max_bm25 = max(r["score"] for r in bm25_results) if bm25_results else 1
    
    # Combine
    for result in semantic_results:
        doc_id = result["doc_id"]
        normalized_score = result["score"] / max_semantic
        combined[doc_id] = semantic_weight * normalized_score
    
    for result in bm25_results:
        doc_id = result["doc_id"]
        normalized_score = result["score"] / max_bm25
        combined[doc_id] = combined.get(doc_id, 0) + keyword_weight * normalized_score
    
    return combined
```

#### 4b. Reciprocal Rank Fusion (RRF) - Better
```python
def rrf_fusion(semantic_results, bm25_results, k: int = 60):
    """
    Reciprocal Rank Fusion - doesn't require score normalization
    RRF_score = sum(1 / (k + rank))
    """
    rrf_scores = {}
    
    # Add semantic ranks
    for rank, result in enumerate(semantic_results, 1):
        doc_id = result["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Add BM25 ranks
    for rank, result in enumerate(bm25_results, 1):
        doc_id = result["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

### Step 5: Final Retrieval

```python
def final_retrieval(combined_scores, top_k: int):
    """Get top-K documents with full metadata"""
    top_doc_ids = [doc_id for doc_id, _ in combined_scores[:top_k]]
    
    # Fetch full documents from Pathway (or cached index)
    results = []
    for doc_id in top_doc_ids:
        doc = self.get_document_by_id(doc_id)
        results.append(doc)
    
    return results
```

## Key Challenges

1. **Syncing BM25 index with Pathway indexing**
   - Pathway indexes dynamically (streaming)
   - Need to keep BM25 index in sync
   - Solution: Build BM25 index from Pathway's indexed documents

2. **Mapping doc_ids between systems**
   - Pathway has its own doc_id system
   - BM25 uses array indices
   - Solution: Maintain mapping table

3. **Performance**
   - BM25 adds computation overhead
   - Need efficient indexing and lookup
   - Solution: Cache BM25 index, use efficient data structures

4. **Score normalization**
   - Semantic scores (cosine similarity) vs BM25 scores (different ranges)
   - Solution: Use RRF (doesn't need normalization) or normalize to [0,1]

## Recommended Implementation Order

1. **Start with Approach 2** (Post-processing) - Simplest
2. **Build BM25 index** from Pathway's indexed documents
3. **Implement hybrid retrieval** in a wrapper class
4. **Test and tune weights** (semantic_weight, keyword_weight)
5. **Move to Approach 1** (Custom endpoint) if needed for performance

## Files to Modify

1. **server.py** - Add hybrid endpoint (if Approach 1)
2. **run-server-gdrive.py** - Build BM25 index after Pathway indexing
3. **test_framework.py** - Use hybrid retrieval (if Approach 2)
4. **New file: hybrid_retriever.py** - Hybrid search logic

## Example Usage (After Implementation)

```python
# In test_framework.py
from hybrid_retriever import HybridRetriever

hybrid_retriever = HybridRetriever(
    pathway_url="http://127.0.0.1:8745",
    bm25_index=bm25_index
)

results = hybrid_retriever.retrieve(
    query="What happened to Edmond Dantes?",
    k=15,
    semantic_weight=0.5,
    keyword_weight=0.5
)
```

