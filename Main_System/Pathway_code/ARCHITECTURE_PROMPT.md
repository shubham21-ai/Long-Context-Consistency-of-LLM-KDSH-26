# Pathway RAG Architecture Diagram Prompt for Erase.ai

Use this prompt with Erase.ai to generate a detailed architecture diagram of the Pathway RAG system.

---

## Prompt for Erase.ai

```
Create a detailed system architecture diagram for a Pathway RAG (Retrieval-Augmented Generation) system with the following components and data flow:

SYSTEM OVERVIEW:
A real-time document indexing and semantic search system that processes documents from Google Drive and local filesystem, converts them to vector embeddings, and serves queries via HTTP API.

DATA SOURCES LAYER (Top):
- Google Drive folder (connected via service account authentication)
  - Continuous monitoring (30-second refresh)
  - Binary file stream with metadata
- Local filesystem directory
  - Binary file stream with metadata
- Both sources feed into Pathway I/O connectors (pw.io.gdrive.read and pw.io.fs.read)

PARSING LAYER:
- CustomParser component
  - Input: Binary files + metadata
  - Primary: unstructured.partition.auto.partition() (handles PDF, DOCX, HTML, Markdown, CSV, Excel, PowerPoint, Images)
  - Fallback: UTF-8 decoding for plain text
  - Output: Structured text elements with metadata (JSON format)
  - Metadata includes: filepath, page numbers, element types

CHUNKING LAYER:
- RecursiveCharacterTextSplitter (LangChain)
  - Input: Parsed document elements (full text)
  - Configuration: 1000 char chunks, 250 char overlap
  - Process: Hierarchical splitting (paragraphs → sentences → words)
  - Output: Text chunks with overlapping context + metadata

EMBEDDING LAYER:
- SentenceTransformerEmbeddings
  - Model: all-mpnet-base-v2
  - Input: Text chunks
  - Process: Batch encoding to 768-dimensional vectors
  - Output: Vector embeddings

INDEXING LAYER:
- VectorStoreServer (Pathway)
  - Input: Vector embeddings + metadata
  - Component: UsearchKnnFactory (Approximate Nearest Neighbor index)
  - Process: Builds ANN index for fast similarity search
  - Storage: In-memory index with caching
  - Output: Indexed vector store ready for queries

SERVER LAYER:
- HTTP Server (FastAPI/Uvicorn via Pathway)
  - Port: 8745
  - Host: 127.0.0.1
  - Mode: Threaded (non-blocking)
  - Endpoints:
    - GET /v1/retrieve: Semantic search (query → top-k chunks)
    - GET /v1/inputs: Document metadata
    - GET /v1/statistics: System statistics

QUERY/RETRIEVAL LAYER (Client Side):
- Client application (test_framework.py)
  - Input: Question string
  - Process: HTTP POST to /v1/retrieve endpoint
  - Server processes: Query embedding → Vector similarity search → Top-k results
  - Output: Retrieved text chunks

DATA FLOW:
1. Documents flow from sources → Parsing → Chunking → Embedding → Indexing → Server
2. Queries flow from Client → Server → Embedding → Vector Search → Results → Client

KEY FEATURES TO HIGHLIGHT:
- Real-time streaming updates (30-second Google Drive refresh)
- Incremental indexing (no full rebuild needed)
- Batch processing for embeddings
- Approximate nearest neighbor search for scalability
- Metadata preservation throughout pipeline
- Caching for performance

STYLING:
- Use different colors for each layer
- Show data flow with arrows
- Include component names and technologies
- Show HTTP endpoints clearly
- Highlight the vector embeddings as 768-dimensional vectors
- Show the continuous/streaming nature with appropriate symbols
```

---

## Alternative Simplified Prompt

If the above is too complex, use this simplified version:

```
Create an architecture diagram showing a Pathway RAG system with these layers:

1. DATA SOURCES: Google Drive + Local Filesystem → Pathway I/O Connectors
2. PARSING: CustomParser → Structured Text + Metadata
3. CHUNKING: TextSplitter → Text Chunks (1000 chars, 250 overlap)
4. EMBEDDING: SentenceTransformer (all-mpnet-base-v2) → 768-dim Vectors
5. INDEXING: VectorStoreServer + USearch Index → Indexed Vectors
6. SERVER: HTTP API (Port 8745) with /v1/retrieve endpoint
7. QUERY: Client → Server → Vector Search → Retrieved Chunks

Show data flow from top to bottom, with query flow from client to server and back.
Include key technologies: Pathway, SentenceTransformers, USearch, FastAPI.
Highlight real-time streaming and incremental indexing capabilities.
```

---

## Manual Architecture Description (Text Format)

If you prefer a text-based description for manual diagram creation:

### System Architecture Layers:

**Layer 1: Data Ingestion**
- Components: Google Drive Connector, Local Filesystem Connector
- Technology: Pathway I/O (pw.io.gdrive.read, pw.io.fs.read)
- Output: Binary data streams with metadata

**Layer 2: Document Parsing**
- Component: CustomParser
- Technology: unstructured library, UTF-8 fallback
- Input: Binary files
- Output: Structured text elements with metadata

**Layer 3: Text Chunking**
- Component: RecursiveCharacterTextSplitter
- Technology: LangChain
- Configuration: 1000 chars/chunk, 250 chars overlap
- Output: Overlapping text chunks

**Layer 4: Embedding Generation**
- Component: SentenceTransformerEmbeddings
- Model: all-mpnet-base-v2
- Output: 768-dimensional vectors

**Layer 5: Vector Indexing**
- Component: VectorStoreServer
- Index: USearch (ANN)
- Storage: In-memory with caching
- Output: Searchable vector index

**Layer 6: HTTP Server**
- Framework: FastAPI/Uvicorn (via Pathway)
- Port: 8745
- Endpoints: /v1/retrieve, /v1/inputs, /v1/statistics

**Layer 7: Query Client**
- Component: test_framework.py
- Process: HTTP requests → Vector search → Results

### Data Flow Direction:
- **Ingestion Flow**: Sources → Parse → Chunk → Embed → Index → Server
- **Query Flow**: Client → Server → Search → Results → Client

### Key Characteristics:
- Real-time/streaming processing
- Incremental updates
- Scalable ANN search
- Metadata preservation
- Caching for performance

