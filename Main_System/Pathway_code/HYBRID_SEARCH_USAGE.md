# Hybrid Search Usage Guide

## Overview

Hybrid search combines Pathway's semantic search (USearch) with BM25 keyword search for better retrieval accuracy.

## Installation

Make sure you have the required dependency:

```bash
pip install rank-bm25
```

## Usage

### Enable Hybrid Search in test_framework.py

Use the `--hybrid-search` flag:

```bash
# Run with hybrid search enabled
python3 test_framework.py --max-tests 5 --hybrid-search

# Run with semantic-only (default)
python3 test_framework.py --max-tests 5
```

### Programmatic Usage

```python
from Pathway_code.hybrid_retriever import HybridRetriever

# Initialize hybrid retriever
retriever = HybridRetriever(
    pathway_url="http://127.0.0.1:8745",
    semantic_weight=0.5,  # Weight for semantic scores
    keyword_weight=0.5,   # Weight for keyword scores
    rrf_k=60              # RRF constant
)

# Build BM25 index from Pathway server
retriever.build_bm25_from_pathway(max_samples=5000)

# Perform hybrid retrieval
results = retriever.retrieve(
    query="What happened to Edmond Dantes?",
    k=15,
    use_rrf=True,  # Use Reciprocal Rank Fusion
    character="Edmond Dantes"  # Optional character boosting
)

# Extract text chunks
chunks = [r["text"] for r in results]
```

## How It Works

1. **BM25 Index Building**: Samples documents from Pathway server by querying with diverse keywords
2. **Dual Retrieval**: 
   - Semantic search via Pathway `/v1/retrieve` endpoint
   - BM25 keyword search on built index
3. **Score Fusion**: Uses Reciprocal Rank Fusion (RRF) or weighted combination
4. **Final Ranking**: Returns top-K results sorted by combined score

## Performance

- **Initial Setup**: BM25 index building takes ~10-30 seconds (one-time per session)
- **Query Time**: Adds ~50-200ms overhead per query (BM25 search)
- **Memory**: BM25 index uses ~10-50MB depending on document count

## Advantages

✅ **Better keyword matching**: Finds exact character names, places, dates
✅ **Better semantic understanding**: Still benefits from embedding-based search
✅ **Best of both worlds**: Combines semantic and keyword strengths
✅ **Character boosting**: Optional boost for chunks containing character names

## Limitations

⚠️ **BM25 index building**: Requires sampling documents from Pathway (may miss some)
⚠️ **Index sync**: BM25 index is built from samples, not all documents
⚠️ **Memory usage**: Stores text chunks in memory for BM25
⚠️ **Additional dependency**: Requires `rank-bm25` package

## Troubleshooting

### "rank_bm25 not found"
```bash
pip install rank-bm25
```

### "Could not sample documents from Pathway server"
- Make sure Pathway server is running
- Check server URL is correct
- Ensure server has indexed some documents

### Hybrid search falls back to semantic-only
- Check error messages in console
- Verify BM25 index was built successfully
- Check Pathway server is responding

## Comparison with rag.py

| Feature | Pathway (semantic-only) | Pathway (hybrid) | rag.py |
|---------|------------------------|------------------|---------|
| Semantic Search | ✅ USearch | ✅ USearch | ✅ FAISS |
| Keyword Search | ❌ | ✅ BM25 | ✅ BM25 |
| Character Boosting | ❌ | ✅ | ✅ |
| Score Fusion | N/A | RRF/Weighted | Weighted |
| Index Building | Automatic | Sampling | Direct |

