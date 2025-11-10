# Configurazione del progetto

# Provider: Cerebras Cloud (100% Gratuito)
# Limiti Free Tier:
# - 30 requests/minute, 60K tokens/minute
# - 14,400 requests/day, 1M tokens/day
# - Info: https://cloud.cerebras.ai/

# Con 50 test cases e 4 modelli = 200 requests totali
# Ben dentro i limiti giornalieri!

# File di configurazione da creare:
# 1. .env (copiare da .env.example)
# 2. Configurare CEREBRAS_API_KEY e WANDB_API_KEY

# Per ottenere la Cerebras API Key:
# 1. Vai su https://cloud.cerebras.ai/
# 2. Crea un account gratuito
# 3. Vai su API Keys e crea una nuova chiave

# Modelli disponibili su Cerebras:
# - llama3.1-8b (8B parametri)
# - llama3.3-70b (70B parametri)
# - qwen3-32b (32B parametri)
# - llama-4-scout (17B parametri)

# Tutti i modelli sono GRATIS!
# Prezzi configurati a $0.00 in src/model_config.py
