from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.config import Config

def get_llm(provider: str, model_name: str):
    temperature = Config.LLM_TEMPERATURE
    provider = provider.lower().strip()

    if provider == 'openai':
        return ChatOpenAI(
            model=model_name,
            api_key=Config.OPENAI_API_KEY,
            temperature=temperature
        )
    elif provider == 'anthropic':
        return ChatAnthropic(
            model=model_name,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature
        )
    elif provider == 'groq':
        return ChatGroq(
            model=model_name,
            api_key=Config.GROQ_API_KEY,
            temperature=temperature
        )
    elif provider == 'ollama':
        return ChatOllama(
            model=model_name,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")