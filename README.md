# VERABENCH

Sistema di benchmark per testare modelli LLM sul routing degli agenti Vera AI.

## Caratteristiche

- Test di modelli LLM su dataset di routing degli agenti
- Supporto per inference tramite Cerebras Cloud (100% gratuito!)
- Metriche: accuratezza, latenza, costo
- Logging locale (JSON) e su Weights & Biases
- Sistema riproducibile con seed configurabile
- Test a parità di condizioni (temperatura = 0.0)

## Modelli Testati

Il benchmark include 4 modelli gratuiti su Cerebras Cloud:

1. **Llama 3.1 8B Instruct** (8B) - Veloce ed efficiente
2. **Llama 3.3 70B Instruct** (70B) - Massima performance
3. **Qwen 3 32B** (32B) - Ottimo bilanciamento
4. **Llama 4 Scout** (17B) - Nuova generazione

## Limiti Cerebras Free Tier

- **30 requests/minute**, 60K tokens/minute
- **14,400 requests/day**, 1M tokens/day
- **Completamente gratuito!**

Con 50 test cases e 4 modelli (200 requests totali), il benchmark completo usa solo **1.4% del limite giornaliero**. Puoi eseguire il benchmark ~72 volte al giorno!

**Tempo di esecuzione**: ~8-10 minuti per tutti i 4 modelli (con rate limiting per sicurezza).

Per dettagli completi sui limiti e ottimizzazioni, vedi [LIMITS.md](LIMITS.md).

## Setup

### 1. Installazione con uv

Assicurati di avere `uv` installato:

```powershell
# Installa uv se non lo hai già
pip install uv
```

### 2. Crea e attiva l'ambiente virtuale

```powershell
# uv creerà automaticamente il venv e installerà le dipendenze
uv sync
```

### 3. Configurazione delle API Keys

Crea un file `.env` nella root del progetto:

```powershell
Copy-Item .env.example .env
```

Modifica `.env` e aggiungi le tue API keys:

```env
CEREBRAS_API_KEY=your_cerebras_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
```

**Come ottenere le API keys:**

- **Cerebras API Key**: 
  1. Vai su https://cloud.cerebras.ai/
  2. Crea un account gratuito
  3. Vai su API Keys e crea una nuova chiave
  4. **100% gratuito** con limiti generosi per testing

- **Weights & Biases API Key**: 
  1. Vai su https://wandb.ai/settings
  2. Copia la tua API key dalla sezione "Danger Zone"

## Utilizzo

### Eseguire il benchmark completo

```powershell
uv run python main.py
```

Questo eseguirà il benchmark su tutti i 4 modelli configurati.

### Struttura dei risultati

I risultati vengono salvati in due formati:

1. **File JSON locali** nella directory `results/`:
   - `{model_key}_{timestamp}.json` - Metriche aggregate
   - `{model_key}_{timestamp}_predictions.json` - Predizioni dettagliate

2. **Dashboard Weights & Biases**:
   - Ogni modello viene loggato come un run separato
   - Visualizzazione interattiva delle metriche
   - Tabelle con tutte le predizioni

## Metriche Calcolate

### Accuratezza
- Percentuale di predizioni corrette
- Calcolata come: `correct_predictions / total_examples`

### Latenza
- **Mean**: Latenza media in secondi
- **Median**: Latenza mediana
- **Min/Max**: Valori minimo e massimo
- **Std**: Deviazione standard

### Costo
- **Total**: Costo totale del run in USD
- **Mean**: Costo medio per esempio
- **Median**: Costo mediano per esempio

**Nota**: Cerebras è completamente gratuito, quindi il costo sarà sempre $0.00!

## Riproducibilità

Il sistema garantisce la riproducibilità attraverso:

1. **Seed fisso** (default: 42)
   - Controllo del random seed
   - Stesso ordinamento dei test cases

2. **Temperatura = 0.0**
   - Output deterministico dai modelli
   - Stesse condizioni per tutti i test

3. **Configurazione salvata**
   - Ogni run salva la configurazione usata
   - Tracciabilità completa dei parametri

Per testare la stabilità, esegui il benchmark più volte e confronta i risultati:

```powershell
# Run 1
uv run python main.py

# Run 2
uv run python main.py

# Confronta i file JSON in results/
```

## Struttura del Progetto

```
VERABENCH/
├── main.py                                    # Script principale
├── pyproject.toml                             # Configurazione uv e dipendenze
├── .python-version                            # Versione Python richiesta
├── .env                                       # API keys (non versionato)
├── .env.example                               # Template per .env
├── router_prompt.json                         # Configurazione prompt
├── vera_agent_router_dataset_v0.1.json       # Dataset di test
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Caricamento dataset e prompt
│   ├── model_config.py                       # Configurazione modelli
│   ├── inference_client.py                   # Client per inferenza
│   ├── metrics.py                            # Calcolo metriche
│   └── logger.py                             # Logging locale e W&B
└── results/                                   # Risultati salvati (generato)
```

## Best Practices Implementate

1. **Modularità**: Codice organizzato in moduli separati per responsabilità
2. **Configurabilità**: Parametri configurabili tramite costanti e argomenti
3. **Logging completo**: Doppio sistema di logging (locale + W&B)
4. **Error handling**: Gestione degli errori per continuare anche in caso di fallimenti
5. **Documentazione**: Docstring complete per tutte le funzioni
6. **Type hints**: Annotazioni di tipo per maggiore chiarezza
7. **Riproducibilità**: Seed fisso e temperatura 0.0
8. **Tracciabilità**: Ogni run salvato con timestamp e configurazione

## Estensioni Future

Il sistema è progettato per essere esteso facilmente:

- **Più modelli**: Aggiungi nuovi modelli in `src/model_config.py`
- **Più prompt**: Testa diverse configurazioni di prompt
- **Più metriche**: Aggiungi nuove metriche in `src/metrics.py`
- **A/B testing**: Confronta diverse configurazioni

## Troubleshooting

### Errore: "CEREBRAS_API_KEY non trovato"
Assicurati di aver creato il file `.env` con la tua API key di Cerebras.

### Errore: "WANDB_API_KEY non trovato"
Il sistema funziona anche senza W&B, ma stamperà un warning. Aggiungi la key per il logging completo.

### Errore: Rate limit exceeded
Cerebras ha limiti di 30 req/min. Il benchmark aggiunge automaticamente pause tra le richieste. Se hai problemi, riduci il numero di test cases.

### Modello non risponde
Verifica che la tua API key di Cerebras sia valida e attiva su https://cloud.cerebras.ai/

## Licenza

Questo progetto è per uso interno/ricerca.
