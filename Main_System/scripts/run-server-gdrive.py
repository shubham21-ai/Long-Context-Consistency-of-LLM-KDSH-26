"""
Pathway RAG Server with Google Drive Support

This version includes both local filesystem and Google Drive connectors.
"""

import os
import pathway as pw
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "integration"))
from server import VectorStoreServer
from langchain_text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# Simple SentenceTransformer wrapper for LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

# Configuration
GDRIVE_FOLDER_ID = "18Eqn1Yr9u-GsRv5ypt0OsjQoi54OlMD0"
GDRIVE_SERVICE_ACCOUNT = "../service_account.json.json"  # Adjust path as needed

data_sources = []

# Option 1: Local filesystem (comment out if not needed)
data_sources.append(
    pw.io.fs.read(
        "../data",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

# Option 2: Google Drive (uncomment to enable)
data_sources.append(
    pw.io.gdrive.read(
        object_id=GDRIVE_FOLDER_ID,
        service_user_credentials_file=GDRIVE_SERVICE_ACCOUNT,
        refresh_interval=30,  # Check for updates every 30 seconds
        format="binary",
        with_metadata=True,
    )
)

sys.path.insert(0, str(Path(__file__).parent.parent / "integration"))
from custom_parser import CustomParse
parser = CustomParse()

# Embeddings - SentenceTransformer (Upgraded to better model for SOTA performance)
# Options:
# - "all-MiniLM-L6-v2" (384d, fast, basic) - OLD
# - "all-MiniLM-L12-v2" (384d, faster, better than L6) - RECOMMENDED for speed
# - "all-mpnet-base-v2" (768d, better quality) - RECOMMENDED for quality
# - "BAAI/bge-large-en-v1.5" (1024d, SOTA) - BEST quality but slower
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Changed from all-MiniLM-L6-v2 for better accuracy
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# Text splitter (Improved overlap for better context preservation, matching rag.py)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=250  # Increased from 50 to 250 for better context (matches rag.py)
)

# Create vector server with both sources
vector_server = VectorStoreServer.from_langchain_components(
    *data_sources,
    embedder=embeddings,
    splitter=text_splitter,
    parser=parser
)

print("=" * 80)
print("Pathway RAG Server with Google Drive Support")
print("=" * 80)
print(f"✓ Local files: ../data")
print(f"✓ Google Drive folder: {GDRIVE_FOLDER_ID}")
print(f"✓ Service account: {GDRIVE_SERVICE_ACCOUNT}")
print(f"✓ Embedding model: {EMBEDDING_MODEL} (SOTA upgrade)")
print(f"✓ Chunk size: 1000, overlap: 250 (improved context)")
print("=" * 80)
print("NOTE: This server only does RETRIEVAL (vector search).")
print("Answer generation/LLM is handled separately in main.py")
print("=" * 80)

vector_server.run_server(host="127.0.0.1", port=8745, threaded=True, with_cache=True)

