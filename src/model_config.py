"""
Configurazione dei modelli disponibili (Cerebras, OpenAI, OpenRouter).

Cerebras Free Tier:
- 30 requests/minute, 60K tokens/minute
- 14,400 requests/day, 1M tokens/day

OpenAI API:
- Richiede OPENAI_API_KEY in .env

OpenRouter Free Tier:
- 20 requests/minute
- 50 requests/day (1000 requests/day con $10 lifetime topup)
- Richiede OPENROUTER_API_KEY in .env
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
 
}


def get_model_config(model_key: str) -> dict:
    """Restituisce la configurazione per un modello specifico."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato.")
    return MODELS[model_key]

def get_all_models() -> list:
    """Restituisce la lista di tutti i modelli configurati."""
    return list(MODELS.keys())
