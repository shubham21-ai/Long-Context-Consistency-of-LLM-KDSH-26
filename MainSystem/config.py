import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (check parent directory too)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Try current directory


def load_openrouter_api_key() -> str:
    """
    Load the OpenRouter API key from environment variable.
    
    Returns:
        The API key string.
        
    Raises:
        RuntimeError: if no API key can be found.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key.strip()
    
    raise RuntimeError(
        "Missing OpenRouter API key. Set OPENROUTER_API_KEY in .env file or as environment variable."
    )


def load_huggingface_api_key() -> str:
    """
    Load the Hugging Face API key from environment variable.
    
    Returns:
        The API key string.
        
    Raises:
        RuntimeError: if no API key can be found.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key:
        return api_key.strip()
    
    raise RuntimeError(
        "Missing Hugging Face API key. Set HUGGINGFACE_API_KEY in .env file or as environment variable."
    )


def load_groq_api_key() -> str:
    """
    Load the Groq API key from environment variable.
    
    Returns:
        The API key string.
        
    Raises:
        RuntimeError: if no API key can be found.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key.strip()
    
    raise RuntimeError(
        "Missing Groq API key. Set GROQ_API_KEY in .env file or as environment variable."
    )


def load_gemini_api_key() -> str:
    """
    Load the Google Gemini API key from environment variable.
    
    Returns:
        The API key string.
        
    Raises:
        RuntimeError: if no API key can be found.
    """
    api_key = os.getenv("GEMINII_API_KEY")
    if api_key:
        return api_key.strip()
    
    raise RuntimeError(
        "Missing Gemini API key. Set GEMINI_API_KEY in .env file or as environment variable."
    )


def load_hf_token(token_path: str | Path | None = None) -> str:
    """
    Load the Hugging Face token from env var HF_TOKEN or from a local file.

    Args:
        token_path: Optional override for the token file path (defaults to ".hf_token").

    Returns:
        The token string.

    Raises:
        RuntimeError: if no token can be found.
    """
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token.strip()

    path = Path(token_path or ".hf_token")
    if path.exists():
        return path.read_text().strip()

    raise RuntimeError(
        "Missing Hugging Face token. Set HF_TOKEN env var or create a '.hf_token' file."
    )
