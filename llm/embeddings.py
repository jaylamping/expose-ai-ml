"""
Embedding management with Ollama for local embeddings.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

from config.settings import settings


class EmbeddingManager:
    """
    Manages text embeddings and vector stores.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize embedding manager with Ollama.
        
        Args:
            model_name: Name of the Ollama embedding model
            **kwargs: Additional arguments
        """
        self.model_name = model_name or settings.default_embedding_model
        self.embeddings = self._create_embeddings(**kwargs)
        self.vector_stores: Dict[str, Any] = {}
    
    def _create_embeddings(self, **kwargs):
        """Create the Ollama embeddings instance."""
        return OllamaEmbeddings(
            model=self.model_name,
            base_url=settings.ollama_base_url,
            **kwargs
        )
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding)
    
    def create_vector_store(self, 
                           documents: List[Document],
                           store_name: str = "default",
                           store_type: str = "faiss",
                           **kwargs) -> Any:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to store
            store_name: Name for the vector store
            store_type: Type of vector store ('faiss', 'chroma')
            **kwargs: Additional arguments
            
        Returns:
            Vector store instance
        """
        if store_type == "faiss":
            vector_store = FAISS.from_documents(documents, self.embeddings, **kwargs)
        elif store_type == "chroma":
            vector_store = Chroma.from_documents(documents, self.embeddings, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        self.vector_stores[store_name] = vector_store
        return vector_store
    
    def similarity_search(self, 
                         query: str,
                         store_name: str = "default",
                         k: int = 4,
                         **kwargs) -> List[Document]:
        """
        Perform similarity search in a vector store.
        
        Args:
            query: Search query
            store_name: Name of the vector store
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        if store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{store_name}' not found")
        
        vector_store = self.vector_stores[store_name]
        return vector_store.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(self, 
                                   query: str,
                                   store_name: str = "default",
                                   k: int = 4,
                                   **kwargs) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            store_name: Name of the vector store
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        if store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{store_name}' not found")
        
        vector_store = self.vector_stores[store_name]
        return vector_store.similarity_search_with_score(query, k=k, **kwargs)
    
    def save_vector_store(self, store_name: str, path: str):
        """Save a vector store to disk."""
        if store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{store_name}' not found")
        
        vector_store = self.vector_stores[store_name]
        vector_store.save_local(path)
    
    def load_vector_store(self, store_name: str, path: str):
        """Load a vector store from disk."""
        vector_store = FAISS.load_local(path, self.embeddings)
        self.vector_stores[store_name] = vector_store
