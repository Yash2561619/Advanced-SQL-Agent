import os
from dotenv import load_dotenv
from cachelib import FileSystemCache

load_dotenv()


class Config:
    # Project Root
    APP_DIR = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))

    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    # Vector store selection
    USE_CHROMADB = os.getenv('USE_CHROMADB', 'True').lower() == 'true'
    USE_MILVUS = os.getenv('USE_MILVUS', 'False').lower() == 'true'

    # Embeddings (HuggingFace runs in-process on CPU under Render's 512MB RAM)
    USE_HUGGINGFACE = os.getenv('USE_HUGGINGFACE', 'True').lower() == 'true'
    USE_OLLAMA = os.getenv('USE_OLLAMA', 'False').lower() == 'true'
    USE_OPENAI = os.getenv('USE_OPENAI', 'False').lower() == 'true'
    HF_EMBEDDING_MODEL = os.getenv('HF_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

    # ChromaDB configurations
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'sql_agent_memory')
    CHROMA_PERSIST_DIRECTORY = os.getenv(
        'CHROMA_PERSIST_DIRECTORY',
        os.path.join(PROJECT_ROOT, 'chroma_db')
    )

    # Ollama configurations
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')

    # Milvus configurations
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
    MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'sql_agent_memory')

    # Database paths
    DB_PATH = os.path.join(PROJECT_ROOT, 'ecommerce.db')
    DATABASE_URL = os.getenv('DATABASE_URL', f"sqlite:///{DB_PATH}")
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session configurations
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_DIR = os.getenv(
        'SESSION_FILE_DIR',
        os.path.join(PROJECT_ROOT, 'flask_session')
    )
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_CACHELIB = FileSystemCache(SESSION_FILE_DIR)

    # LLM Settings (Locked to openai/gpt-oss-120b)
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq')
    LLM_MODEL = os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))

    # Application & Graph Settings
    MAX_TABLES_TO_SELECT = int(os.getenv('MAX_TABLES_TO_SELECT', '5'))
    MAX_SQL_REFINEMENT_ATTEMPTS = int(os.getenv('MAX_SQL_REFINEMENT_ATTEMPTS', '3'))
    GRAPH_RECURSION_LIMIT = int(os.getenv('GRAPH_RECURSION_LIMIT', '20'))

    # LangChain / LangSmith Tracing
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')