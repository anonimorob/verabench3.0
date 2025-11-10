# VERABENCH

Sistema di benchmark per testare modelli LLM sul routing degli agenti Vera AI.

## Caratteristiche

- Test di modelli LLM su dataset di routing degli agenti
- Supporto per inference tramite Cerebras Cloud (100% gratuito!)
- Metriche: accuratezza, latenza, costo
- Logging locale (JSON) e su Weights & Biases
- Sistema riproducibile con seed configurabile
- Test a parità di condizioni (temperatura = 0.0)

## Limiti Cerebras Free Tier

- **30 requests/minute**, 60K tokens/minute
- **14,400 requests/day**, 1M tokens/day
- **Completamente gratuito!**

Con 50 test cases e 4 modelli (200 requests totali), benchmark eseguibile ~72 volte al giorno


## Setup

### 1. Installazione con uv

Assicurati di avere `uv` installato:

```powershell

pip install uv
```

### 2. Crea e attiva l'ambiente virtuale

```powershell

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

