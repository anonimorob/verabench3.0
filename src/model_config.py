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
    # Modelli TogetherAI

    "mistralai/Mistral-7B-Instruct-v0.1": {
        "id": "mistralai/Mistral-7B-Instruct-v0.1",
        "name": "Mistral 7B Instruct v0.1",
        "params": "7B",
        "provider": "togetherai",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.20,
    },

    "mistralai/Mistral-7B-Instruct-v0.3": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral 7B Instruct v0.3",
        "params": "7B",
        "provider": "togetherai",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.20,
    },

    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "name": "Mistral Small 24B Instruct 2501",
        "params": "24B",
        "provider": "togetherai",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 0.80,
    },

    "pangram/mistral-small-2501": {
        "id": "pangram/mistral-small-2501",
        "name": "Pangram Mistral Small 2501",
        "params": "Unknown",
        "provider": "togetherai",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 0.80,
    },

    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "name": "Qwen3 Next 80B A3B Instruct",
        "params": "80B",
        "provider": "togetherai",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 1.50,
    },

    "Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "id": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "name": "Qwen2.5 7B Instruct Turbo",
        "params": "7B",
        "provider": "togetherai",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 0.30,
    },

    "deepseek-ai/DeepSeek-V3": {
        "id": "deepseek-ai/DeepSeek-V3",
        "name": "DeepSeek V3",
        "params": "Unknown",
        "provider": "togetherai",
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 1.25,
    },

    "openai/gpt-oss-20b": {
        "id": "openai/gpt-oss-20b",
        "name": "GPT-OSS 20B",
        "params": "20B",
        "provider": "togetherai",
        "input_price_per_1m": 0.05,
        "output_price_per_1m": 0.20,
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

    

 
}


def get_model_config(model_key: str) -> dict:
    """Restituisce la configurazione per un modello specifico."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato.")
    return MODELS[model_key]

def get_all_models() -> list:
    """Restituisce la lista di tutti i modelli configurati."""
    return list(MODELS.keys())
