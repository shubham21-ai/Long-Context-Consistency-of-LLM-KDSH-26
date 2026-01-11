# SOTA Improvements for Pathway RAG Server

## Current Issues

### 1. **No LLM in Pathway Server** ❌
- **Current**: Pathway server (`run-server-gdrive.py`) only does **RETRIEVAL** (vector search)
- **No LLM**: It doesn't generate answers, only returns chunks
- **Comparison**: `rag.py` uses Gemini LLM for answer generation
- **Impact**: Lower accuracy because retrieval-only, no answer synthesis

### 2. **Basic Embedding Model** ❌
- **Current**: `all-MiniLM-L6-v2` (384 dimensions, basic quality)
- **Issue**: Outdated model, not SOTA
- **Impact**: Lower retrieval quality = lower accuracy

### 3. **Only Semantic Search** ❌
- **Current**: Pathway uses USearch (semantic only)
- **Missing**: No hybrid search (semantic + keyword/BM25)
- **Comparison**: `rag.py` uses hybrid search (FAISS + BM25)
- **Impact**: Misses keyword matches (character names, specific terms)

### 4. **Basic Chunking** ⚠️
- **Current**: `chunk_size=1000, chunk_overlap=50`
- **Issue**: Small overlap, may lose context
- **Comparison**: `rag.py` uses `chunk_overlap=250` (better context preservation)

## Recommended SOTA Improvements

### Option 1: Improve Retrieval Only (Easier) ⭐ RECOMMENDED

**Changes to `run-server-gdrive.py`:**

1. **Better Embedding Model**
   ```python
   # Replace: all-MiniLM-L6-v2
   # With one of:
   embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # 768d, better quality
   # OR
   embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")  # 1024d, SOTA
   # OR (smaller, faster, still good)
   embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")  # 384d, better than L6
   ```

2. **Better Chunking**
   ```python
   text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
       chunk_size=1000,
       chunk_overlap=250,  # Increase from 50 to 250 (like rag.py)
   )
   ```

3. **Add Reranking** (Optional but SOTA)
   - Use Cohere rerank API or cross-encoder models
   - Rerank top-K results for better precision

### Option 2: Add Hybrid Search (Complex, Requires Custom Implementation)

Pathway doesn't natively support hybrid search. Options:
- Use Pathway for semantic + add BM25 layer separately
- Or use Pathway's metadata filtering for keyword matching

### Option 3: Use Better Embedding APIs (Best Quality, Costs Money)

```python
# Use OpenAI embeddings (best quality)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072d, SOTA

# OR Cohere embeddings
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")  # 1024d, SOTA
```

## My Recommendation: Quick Wins ⭐

1. **Upgrade embedding model** → `all-mpnet-base-v2` or `all-MiniLM-L12-v2`
2. **Increase chunk overlap** → 250 (like rag.py)
3. **Keep current architecture** (Pathway for retrieval, Groq for evaluation)

This gives you 80% of the benefit with minimal changes.

## Next Steps

Would you like me to:
1. ✅ Update `run-server-gdrive.py` with better embeddings and chunking?
2. ✅ Add embedding model options/config?
3. ✅ Document the improvements?

