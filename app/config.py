import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Remove this in production

class Settings:
    # Model configuration
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.4))
    STREAMING = bool(os.getenv("STREAMING", True))
    
    # Path configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "shared_chroma_db")
    SHARED_DOCS_PATH = os.getenv("SHARED_DOCS_PATH", "shared_docs")
    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "bm25_index.pkl")
    
    # Chroma configuration
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "main_collection")
    
    # Embedding model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    
    # BM25 configuration
    IS_USING_BM25 = os.getenv("IS_USING_BM25", "False").lower() == "true"
    BM25_K = int(os.getenv("BM25_K", "3"))
    
    # Text splitting configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval configuration
    VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "10"))
    ENSEMBLE_VECTOR_WEIGHT = float(os.getenv("ENSEMBLE_VECTOR_WEIGHT", "0.6"))
    ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.4"))
    
    @property
    def chroma_db_path(self) -> Path:
        """Get Chroma DB path as Path object"""
        return Path(self.CHROMA_DB_PATH)
    
    @property
    def shared_docs_path(self) -> Path:
        """Get shared docs path as Path object"""
        return Path(self.SHARED_DOCS_PATH)
    
    @property
    def bm25_index_path(self) -> Path:
        """Get BM25 index path as Path object"""
        return Path(self.BM25_INDEX_PATH)

settings = Settings()
