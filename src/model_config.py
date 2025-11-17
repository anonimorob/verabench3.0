"""
Configurazione dei modelli disponibili (Cerebras, OpenAI, OpenRouter, Google AI Studio, NVIDIA).

Cerebras Free Tier:
- 30 requests/minute, 60K tokens/minute
- 14,400 requests/day, 1M tokens/day

OpenAI API:
- Richiede OPENAI_API_KEY in .env

OpenRouter Free Tier:
- 20 requests/minute
- 50 requests/day (1000 requests/day con $10 lifetime topup)
- Richiede OPENROUTER_API_KEY in .env

Google AI Studio:
- Free tier: 15 requests/minute, 1M tokens/minute
- Richiede GOOGLE_API_KEY in .env

NVIDIA NIM:
- Free tier per development  max 40 req per min
- Richiede NVIDIA_API_KEY in .env
"""

MODELS = {
    # Modelli Cerebras

    "llama3.1-8b": {
        "id": "llama3.1-8b",
        "name": "Llama 3.1 8B Instruct",
        "params": "8B",
        "provider": "cerebras",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
    },
    
    "llama-3.3-70b": {
        "id": "llama-3.3-70b",
        "name": "Llama 3.3 70B Instruct",
        "params": "70B",
        "provider": "cerebras",
        "input_price_per_1m": 0.85,
        "output_price_per_1m": 1.20,
    },
  
    # Modelli OpenAI
    "gpt-4o-mini": {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "params": "Unknown",
        "provider": "openai",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
    },
    
    "gpt-4o": {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "params": "Unknown",
        "provider": "openai",
        "input_price_per_1m": 2.50,
        "output_price_per_1m": 10.00,
    },
         
    # Modelli OpenRouter
    "gemma-3-12b": {
        "id": "google/gemma-3-12b-it",
        "name": "Gemma 3 12B Instruct",
        "params": "12B",
        "provider": "openrouter",
        "input_price_per_1m": 0.03,
        "output_price_per_1m": 0.10,
    },

    "gemma-3-27b": {
        "id": "google/gemma-3-27b-it",
        "name": "Gemma 3 27B Instruct",
        "params": "27B",
        "provider": "openrouter",
        "input_price_per_1m": 0.09,
        "output_price_per_1m": 0.16,
    },
 
    # Modelli Google AI Studio
    "gemini-2.5-flash-lite": {
        "id": "gemini-2.5-flash-lite",
        "name": "Gemini 2.5 Flash-Lite",
        "params": "Unknown",
        "provider": "google",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "gemini-2.5-flash": {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "params": "Unknown",
        "provider": "google",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    # Modelli NVIDIA NIM
    "deepseek-r1-distill-llama-8b": {
        "id": "deepseek-ai/deepseek-r1-distill-llama-8b",
        "name": "DeepSeek R1 Distill Llama 8B",
        "params": "8B",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "gpt-oss-20b": {
        "id": "openai/gpt-oss-20b",
        "name": "OpenAI GPT-OSS 20B",
        "params": "20B (MoE)",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "name": "OpenAI GPT-OSS 120B",
        "params": "120B (MoE)",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "phi-4-mini": {
        "id": "microsoft/phi-4-mini-instruct",
        "name": "Phi-4 Mini Instruct",
        "params": "14B",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "phi-4-mini-flash-reasoning": {
        "id": "microsoft/phi-4-mini-flash-reasoning",
        "name": "Phi-4 Mini Flash Reasoning",
        "params": "14B",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "mistral-nemo": {
        "id": "mistralai/mistral-nemotron",
        "name": "Mistral Nemotron",
        "params": "12B",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

    "qwen3-next-80b": {
        "id": "qwen/qwen3-next-80b-a3b-instruct",
        "name": "Qwen3 Next 80B A3B Instruct",
        "params": "80B (MoE)",
        "provider": "nvidia",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
    },

 
}


def get_model_config(model_key: str) -> dict:
    """Restituisce la configurazione per un modello specifico."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato.")
    return MODELS[model_key]

def get_all_models() -> list:
    """Restituisce la lista di tutti i modelli configurati."""
    return list(MODELS.keys())
