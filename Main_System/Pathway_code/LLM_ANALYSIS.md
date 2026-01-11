# LLM Analysis: Pathway Server vs rag.py

## Key Finding: **No LLM in Pathway Server** ❌

### Current Architecture

#### Pathway Server (`run-server-gdrive.py`)
- **Purpose**: RETRIEVAL ONLY (vector search)
- **LLM Used**: ❌ **NONE**
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2` → now upgraded to `all-mpnet-base-v2`)
- **Functionality**: Returns relevant document chunks only
- **Answer Generation**: Not done here (handled separately)

#### rag.py (Traditional RAG)
- **Purpose**: Full RAG pipeline (retrieval + answer generation)
- **LLM Used**: ✅ **Gemini** (`ChatGoogleGenerativeAI`)
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Functionality**: 
  1. Retrieves chunks (hybrid: FAISS semantic + BM25 keyword)
  2. Generates answers using Gemini LLM
- **Answer Generation**: Done in `answer_query()` method

### Why Pathway Server is "Less Accurate"

1. **No Answer Generation**: Pathway only retrieves chunks, doesn't synthesize answers
2. **No Hybrid Search**: Only semantic search (no keyword/BM25 like rag.py)
3. **Basic Embeddings**: Was using `all-MiniLM-L6-v2` (now upgraded)
4. **No Character Boosting**: rag.py boosts chunks containing character names

### Current Workflow

```
Pathway Server:
  Query → Embedding → Vector Search → Return Chunks ❌ (No LLM)

rag.py:
  Query → Hybrid Search (FAISS + BM25) → Character Boost → Gemini LLM → Answer ✅
```

### Where LLM is Actually Used

1. **test_framework.py** → Uses Pathway server for retrieval, then:
   - `evaluator.py` → Uses **Groq** (`qwen/qwen3-32b`) for evaluation
   - `question_generator.py` → Uses **Groq** for question generation
   - `claim_extractor.py` → Uses **Groq** for claim extraction

2. **rag.py** → Uses **Gemini** for answer generation (not used by test_framework.py anymore)

## Improvements Made

✅ **Upgraded Embeddings**: `all-MiniLM-L6-v2` → `all-mpnet-base-v2` (better retrieval quality)
✅ **Better Chunking**: Overlap 50 → 250 (better context preservation)

## Future SOTA Improvements (Optional)

1. **Add Reranking**: Cohere rerank or cross-encoder models
2. **Hybrid Search**: Add BM25 keyword search layer (complex, requires custom implementation)
3. **Better Embeddings**: OpenAI embeddings (`text-embedding-3-large`) or Cohere (`embed-english-v3.0`)
4. **Answer Generation**: Add LLM layer to Pathway server (but currently handled separately)

## Summary

- **Pathway server is NOT supposed to use LLM** - it's a retrieval-only service
- **LLM is used separately** in evaluator/question_generator for evaluation
- **Accuracy depends on retrieval quality**, not answer generation (since answers are generated separately)
- **Improvements made**: Better embeddings + better chunking = better retrieval quality

