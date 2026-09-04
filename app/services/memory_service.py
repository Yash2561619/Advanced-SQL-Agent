# app/services/memory_service.py

import os
import logging
from app.config import Config

logger = logging.getLogger(__name__)

# Modern imports with fallback
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        OllamaEmbeddings = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        Chroma = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None


class MemoryService:
    def __init__(self):
        self.embedding_function = None
        self.vector_store = None
        self.memory_buffer = []
        self.buffer_size = 10
        self.initialize()

    def initialize(self):
        try:
            self.embedding_function = self._create_embedding_function()
            self.vector_store = self._create_vector_store()
            logger.info("MemoryService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService: {str(e)}", exc_info=True)
            self.vector_store = None

    def _create_embedding_function(self):
        # 1. HuggingFace (Ideal for Render free tier: <150MB RAM, CPU-only)
        if getattr(Config, "USE_HUGGINGFACE", False) or getattr(Config, "USE_HF", False):
            if HuggingFaceEmbeddings is None:
                raise ImportError("HuggingFaceEmbeddings not installed. Run: pip install langchain-huggingface sentence-transformers")
            model_name = getattr(Config, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            logger.info(f"Using HuggingFace embeddings with model: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

        # 2. Local Ollama instance
        elif getattr(Config, "USE_OLLAMA", False):
            if OllamaEmbeddings is None:
                raise ImportError("OllamaEmbeddings is not installed.")
            logger.info(f"Using Ollama embeddings with model: {Config.OLLAMA_MODEL}")
            return OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_MODEL
            )

        # 3. OpenAI Embeddings
        elif getattr(Config, "USE_OPENAI", False):
            if OpenAIEmbeddings is None:
                raise ImportError("OpenAIEmbeddings is not installed.")
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)

        # Fallback default: If none explicitly set, default safely to HuggingFace
        elif HuggingFaceEmbeddings is not None:
            logger.info("No provider explicitly set. Defaulting to lightweight HuggingFace embeddings.")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )

        else:
            logger.error("No valid embedding configuration or package found")
            raise ValueError("No valid embedding configuration found")

    def _create_vector_store(self):
        if getattr(Config, "USE_CHROMADB", True):
            if Chroma is None:
                raise ImportError("Chroma is not installed. Run: pip install langchain-chroma chromadb")
            os.makedirs(Config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            logger.info(f"Initializing ChromaDB at {Config.CHROMA_PERSIST_DIRECTORY}")
            return Chroma(
                collection_name=Config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function,
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY
            )
        elif getattr(Config, "USE_MILVUS", False):
            logger.info("Initializing Milvus")
            from langchain_community.vectorstores import Milvus
            return Milvus(
                embedding_function=self.embedding_function,
                collection_name=Config.MILVUS_COLLECTION,
                connection_args={"host": Config.MILVUS_HOST, "port": Config.MILVUS_PORT}
            )
        else:
            logger.error("No valid vector store configuration found")
            raise ValueError("No valid vector store configuration found")

    def add_memory(self, text, metadata=None):
        if self.vector_store is None:
            logger.warning("Vector store is not available. Skipping memory addition.")
            return

        try:
            self.vector_store.add_texts([text], metadatas=[metadata] if metadata else None)
            logger.info(f"Successfully added memory: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add memory: {str(e)}", exc_info=True)

    def search_memory(self, query, k=5):
        if self.vector_store is None:
            logger.warning("Vector store not initialized. Returning empty memory list.")
            return []
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Successfully searched memory for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Failed to search memory: {str(e)}", exc_info=True)
            return []

    def clear_memory(self):
        try:
            if self.vector_store is not None:
                if hasattr(self.vector_store, "delete_collection"):
                    self.vector_store.delete_collection()
                elif hasattr(self.vector_store, "drop"):
                    self.vector_store.drop()
            self.vector_store = self._create_vector_store()
            logger.info("Successfully cleared memory")
        except Exception as e:
            logger.error(f"Failed to clear memory: {str(e)}", exc_info=True)
            raise