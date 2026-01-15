
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

data_sources = []

# Local filesystem source
data_sources.append(
    pw.io.fs.read(
        "../data",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

# Google Drive source (optional - uncomment to enable)
# Replace with your folder ID and service account path
# GDRIVE_FOLDER_ID = "18Eqn1Yr9u-GsRv5ypt0OsjQoi54OlMD0"  # Your folder ID
# GDRIVE_SERVICE_ACCOUNT = "../service_account.json.json"  # Path to service account JSON
# 
# data_sources.append(
#     pw.io.gdrive.read(
#         object_id=GDRIVE_FOLDER_ID,
#         service_user_credentials_file=GDRIVE_SERVICE_ACCOUNT,
#         refresh_interval=30,  # Check for updates every 30 seconds
#         format="binary",
#         with_metadata=True,
#     )
# )

sys.path.insert(0, str(Path(__file__).parent.parent / "integration"))
from custom_parser import CustomParse
parser = CustomParse()

# Embeddings - SentenceTransformer (replacing VoyageAI)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=50
)

vector_server = VectorStoreServer.from_langchain_components(
    *data_sources,
    embedder=embeddings,
    splitter=text_splitter,
    parser = parser
)

vector_server.run_server(host="127.0.0.1", port=8745, threaded=True, with_cache=True)
