"""
Configurazione dei modelli disponibili su Cerebras (Free Tier).

Limiti Cerebras Free Tier:
- 30 requests/minute, 60K tokens/minute
- 14,400 requests/day, 1M tokens/day

"""

MODELS = {
    "llama3.1-8b": {
        "id": "llama3.1-8b",
        "name": "Llama 3.1 8B Instruct",
        "params": "8B",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
    },
    
    "llama-3.3-70b": {
        "id": "llama-3.3-70b",
        "name": "Llama 3.3 70B Instruct",
        "params": "70B",
        "input_price_per_1m": 0.85,
        "output_price_per_1m": 1.20,
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
