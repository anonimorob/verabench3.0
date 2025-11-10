# Limiti e Ottimizzazione per Cerebras

## Limiti Free Tier Cerebras

### Per Request
- **30 requests/minute** (1 request ogni 2 secondi)
- **60,000 tokens/minute** per modelli piccoli (8B, 17B, 32B)
- **64,000 tokens/minute** per Llama 3.3 70B

### Per Ora
- **900 requests/hour**
- **1,000,000 tokens/hour**

### Per Giorno
- **14,400 requests/day**
- **1,000,000 tokens/day**

## Analisi del Benchmark

### Configurazione Attuale
- **50 test cases** nel dataset
- **4 modelli** da testare
- **Totale richieste**: 50 × 4 = **200 requests**

### Stima Tempi
Con rate limiter a 2.5 secondi tra richieste:
- **Per modello**: 50 × 2.5s = ~125 secondi (~2 minuti)
- **Totale 4 modelli**: ~500 secondi (~8-10 minuti)

### Uso Risorse
- **Requests usate**: 200 / 14,400 = **1.4% del limite giornaliero**
- **Puoi eseguire il benchmark completo ~72 volte al giorno!**

## Opzioni di Ottimizzazione

### Se Vuoi Velocizzare
Riduci il delay nel rate limiter (main.py):
```python
rate_limiter = RateLimiter(requests_per_minute=28, min_delay_seconds=2.2)
```

### Se Hai Rate Limit Errors
Aumenta il delay:
```python
rate_limiter = RateLimiter(requests_per_minute=20, min_delay_seconds=3.0)
```

### Se Vuoi Testare Più Velocemente
Riduci il numero di test cases modificando il dataset o filtrando:
```python
# In main.py, dopo get_test_cases():
test_cases = test_cases[:10]  # Solo primi 10 test
```

## Raccomandazioni

1. **Configurazione attuale (2.5s delay)**: Bilanciamento ottimale tra velocità e sicurezza
2. **Non modificare** se non hai problemi di rate limiting
3. **Cerebras è generoso**: Con 50 test puoi eseguire molti esperimenti al giorno
4. **Per sperimentazione rapida**: Usa un subset del dataset (es. 10-20 test)

## Confronto con Altri Provider

| Provider | Limite Giornaliero | Test Completi Possibili |
|----------|-------------------|-------------------------|
| Cerebras | 14,400 req/day | ~72 volte |
| Groq (Llama 3.3 70B) | 1,000 req/day | ~5 volte |
| OpenRouter Free | 50 req/day | 0 (troppo basso) |

**Cerebras è la scelta migliore per il tuo caso d'uso!**
