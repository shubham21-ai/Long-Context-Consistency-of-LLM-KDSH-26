import numpy as np
from rank_bm25 import BM25Okapi
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from sentence_transformers import SentenceTransformer
import faiss
import os
from config import load_gemini_api_key

class RAG:
    def __init__(self, model_llm: str, model_embedding: str, data_dir:str):
        self.model_llm = model_llm
        self.model_embedding = model_embedding
        self.data_path = data_dir
        self.bm25_index = None  # For keyword search
        self.text_strings = []  # Store text strings for BM25
        # Set API key for langchain
        api_key = load_gemini_api_key()
        os.environ['GOOGLE_API_KEY'] = api_key

    def load_model(self):
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_llm,
            google_api_key=os.environ.get('GOOGLE_API_KEY')
        )
        # Use sentence-transformers for embeddings (free, no quota)
        self.embedding = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self):
        self.loader = DirectoryLoader(self.data_path, glob="*.txt", loader_cls=TextLoader)
        self.documents = self.loader.load()

    def split_data(self):
        # IMPROVEMENT 1: Larger chunks with more overlap for better context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # CHANGED from 500 - gives more context
            chunk_overlap=250,  # CHANGED from 100 - preserves context across chunks
            separators=["\n\n", "\n", ". ", " ", ""]  # Split by paragraphs, then sentences
        )
        self.texts = self.text_splitter.split_documents(self.documents)

        # Storing text strings for BM25 keyword search
        self.text_strings = [doc.page_content for doc in self.texts]
        print(f"  → Created {len(self.text_strings)} chunks (size: 1000, overlap: 250)", flush=True)

    def embed_data(self):
        # Use sentence-transformers encode method
        self.embeddings = self.embedding.encode(self.text_strings, show_progress_bar=False)
        self.embeddings = np.array(self.embeddings).astype('float32')

    def create_index(self):
        # Creating FAISS index for semantic search
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        # Creating BM25 index for keyword search
        # Tokenizing texts for BM25
        tokenized_texts = [text.lower().split() for text in self.text_strings]
        self.bm25_index = BM25Okapi(tokenized_texts)

    def retrieve_data(self, query: str, k: int = 15, character: str = ""):
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query
            k: Number of chunks to return (default: 15)
            character: Character name to boost in results (optional)
        
        Returns:
            List of retrieved document chunks
        """
        # IMPROVEMENT 2: Retrieve MORE candidates for better recall
        num_candidates = 50  # CHANGED from 20
        
        # Semantic search using embeddings (FAISS)
        query_embed = self.embedding.encode([query], show_progress_bar=False)
        query_embed = np.array(query_embed).astype('float32')
        semantic_distances, semantic_indices = self.index.search(query_embed, k=num_candidates)
        
        # Keyword search using BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        keyword_indices = np.argsort(bm25_scores)[::-1][:num_candidates]
        
        # IMPROVEMENT 3: Balanced weights (50/50) for character-specific queries
        # For character backstories, keywords (names) are as important as semantics
        semantic_weight = 0.5  # CHANGED from 0.6
        keyword_weight = 0.5   # CHANGED from 0.4
        
        # Convert FAISS distances to similarity scores (lower distance = higher similarity)
        semantic_scores = 1 / (1 + semantic_distances[0])  # Normalize distances to [0,1]
        
        # Normalize BM25 scores to [0,1]
        if bm25_scores.max() > 0:
            keyword_scores = bm25_scores / bm25_scores.max()
        else:
            keyword_scores = bm25_scores
        
        # Create combined score dictionary
        combined_scores = {}
        for idx, score in zip(semantic_indices[0], semantic_scores):
            combined_scores[idx] = combined_scores.get(idx, 0) + semantic_weight * score
        
        for idx in keyword_indices:
            combined_scores[idx] = combined_scores.get(idx, 0) + keyword_weight * keyword_scores[idx]
        
        # IMPROVEMENT 4: Boost chunks that mention the character
        if character:
            character_lower = character.lower()
            for idx in combined_scores:
                chunk_text = self.text_strings[idx].lower()
                if character_lower in chunk_text:
                    # Count mentions
                    mention_count = chunk_text.count(character_lower)
                    # Boost score by 20% per mention (up to 60% total boost)
                    boost = min(mention_count * 0.2, 0.6)
                    combined_scores[idx] = combined_scores[idx] * (1 + boost)
        
        # IMPROVEMENT 5: Return MORE chunks (15 instead of 10)
        top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        final_indices = [idx for idx, _ in top_indices]
        
        # Return retrieved documents
        retrieved_docs = [self.texts[idx] for idx in final_indices]
        
        # DIAGNOSTIC: Show retrieval quality
        if character:
            chunks_with_char = sum(1 for idx in final_indices if character.lower() in self.text_strings[idx].lower())
            if chunks_with_char > 0:
                print(f"    → Character '{character}' found in {chunks_with_char}/{k} retrieved chunks", flush=True)
        
        self.retriever = retrieved_docs
        return self.retriever

    def answer_query(self, query: str):
        self.retrieved_data = self.retrieve_data(query)
        self.answer = self.llm.invoke(self.retrieved_data)
        return self.answer

    def save_index(self):
        self.index.save("index.faiss")
        self.embeddings.save("embeddings.npy")
        self.texts.save("texts.pkl")