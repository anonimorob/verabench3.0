# VERABENCH# VERABENCH



Sistema modulare di benchmark per testare modelli LLM su task specifiche degli agenti Vera AI.Sistema di benchmark per testare modelli LLM sul routing degli agenti Vera AI.



## Overview## Caratteristiche



VERABENCH è un framework per valutare le performance di LLM su 3 task critiche:- Test di modelli LLM su dataset di routing degli agenti

- Supporto per inference tramite Cerebras Cloud (100% gratuito!)

1. **Routing**: Selezionare l'agente corretto per una richiesta utente- Metriche: accuratezza, latenza, costo

2. **Tool Calling**: Scegliere il tool giusto con parametri corretti  - Logging locale (JSON) e su Weights & Biases

3. **Judge**: Validare output prima di inviarli all'utente (quality gate)- Sistema riproducibile con seed configurabile

- Test a parità di condizioni (temperatura = 0.0)

Ogni task ha dataset dedicato, metriche specifiche e script indipendente.

## Limiti Cerebras Free Tier

---

- **30 requests/minute**, 60K tokens/minute

## Architettura- 



```

tasks/Con 50 test cases e 4 modelli (200 requests totali), benchmark eseguibile ~72 volte al giorno

  routing/          → Dataset, prompt, metriche routing

  tool_calling/     → Dataset, prompt, metriche tool calling

  judge/            → Dataset, prompt, metriche judge## Setup

  

main_routing.py     → Benchmark routing### 1. Installazione con uv

main_tool_calling.py → Benchmark tool calling  

main_judge.py       → Benchmark judgeAssicurati di avere `uv` installato:

```

```powershell

### Inference Providers

pip install uv
Chiamiamo direttamente i provider con SDK OpenAI:```


- **Cerebras**: `base_url="https://api.cerebras.ai/v1"` (free tier)
### 2. Crea e attiva l'ambiente virtuale

- **OpenAI**: base_url default (proprietario)

- **OpenRouter**: `base_url="https://openrouter.ai/api/v1"` (gateway multi-modello)```powershell



Stesso SDK, cambia solo `base_url`.uv sync

```

---

### 3. Configurazione delle API Keys

## Setup

Crea un file `.env` nella root del progetto:

### 1. Installazione

```powershell

```bashCopy-Item .env.example .env

# Installa uv```

pip install uv

Modifica `.env` e aggiungi le tue API keys:

# Sincronizza dipendenze

uv sync```env

```CEREBRAS_API_KEY=your_cerebras_api_key_here

WANDB_API_KEY=your_wandb_api_key_here

### 2. Configurazione API Keys```



```bash**Come ottenere le API keys:**

# Copia template

cp .env.example .env- **Cerebras API Key**: 

  1. Vai su https://cloud.cerebras.ai/

# Modifica .env con le tue keys  2. Crea un account gratuito

```  3. Vai su API Keys e crea una nuova chiave



File `.env`:- **Weights & Biases API Key**: 

```env  1. Vai su https://wandb.ai/settings

CEREBRAS_API_KEY=your_key_here  2. Copia la tua API key dalla sezione "Danger Zone"

OPENAI_API_KEY=your_key_here  

OPENROUTER_API_KEY=your_key_here## Utilizzo

WANDB_API_KEY=your_key_here

```### Eseguire il benchmark completo



**Dove ottenere le keys:**```powershell

- Cerebras: https://cloud.cerebras.ai/ (free tier)uv run python main.py

- OpenAI: https://platform.openai.com/api-keys```

- OpenRouter: https://openrouter.ai/keys

- W&B: https://wandb.ai/settingsQuesto eseguirà il benchmark su tutti i 4 modelli configurati.



---### Struttura dei risultati



## Task 1: RoutingI risultati vengono salvati in due formati:



**Obiettivo:** Dato un messaggio utente, selezionare l'agente giusto.1. **File JSON locali** nella directory `results/`:

   - `{model_key}_{timestamp}.json` - Metriche aggregate

### Come funziona   - `{model_key}_{timestamp}_predictions.json` - Predizioni dettagliate



1. Dataset: 50 test con richieste utente + agente corretto2. **Dashboard Weights & Biases**:

2. Modello riceve: system prompt (lista agenti) + user request   - Ogni modello viene loggato come un run separato

3. Deve rispondere con: nome dell'agente   - Visualizzazione interattiva delle metriche

   - Tabelle con tutte le predizioni

### Esecuzione

## Metriche Calcolate

```bash

uv run python main_routing.py### Accuratezza

```- Percentuale di predizioni corrette

- Calcolata come: `correct_predictions / total_examples`

Modelli testati (configurabili in `MODELS_TO_TEST`):

- gpt-4o-mini### Latenza

- gpt-4o- **Mean**: Latenza media in secondi

- **Median**: Latenza mediana

### Metriche- **Min/Max**: Valori minimo e massimo

- **Std**: Deviazione standard

- **routing_accuracy**: % routing corretti

- **total_cost**: Costo totale in USD### Costo

- **total_latency**: Latenza totale in secondi- **Total**: Costo totale del run in USD



### Risultati

## Riproducibilità

Salvati in `results/routing/TIMESTAMP/`:

- `{model}_results.json`: Metriche aggregateIl sistema garantisce la riproducibilità attraverso:

- Grafici: accuracy vs cost, comparison bars

- W&B dashboard: `verabench-routing`1. **Seed fisso** (default: 42)

   - Controllo del random seed

---   - Stesso ordinamento dei test cases



## Task 2: Tool Calling2. **Temperatura = 0.0**

   - Output deterministico dai modelli

**Obiettivo:** Dato un messaggio utente, selezionare tool corretto con parametri esatti.   - Stesse condizioni per tutti i test



### Come funziona3. **Configurazione salvata**

   - Ogni run salva la configurazione usata

1. Dataset: 50 test con richieste → expected tool + parameters   - Tracciabilità completa dei parametri

2. Modello riceve: system prompt (tools disponibili) + user request

3. Deve rispondere JSON:
   ```json
   {
     "tool": "nome_tool",
     "parameters": { "param1": "value1", ... }
   }
   ```

### Esecuzione

```bash
uv run python main_tool_calling.py
```

### Metriche

- **tool_selection_accuracy**: % tool corretti
- **parameter_correctness**: Media di:
  - **parameter_name_accuracy**: % nomi parametri corretti
  - **parameter_value_correctness**: % valori parametri corretti
  - **parameter_type_accuracy**: % tipi parametri corretti
- **total_cost**, **total_latency**

### Note tecniche

- GPT-4o risponde con markdown code blocks → cleanup automatico prima del parsing
- Parametri non specificati devono essere `null`
- Date relative vengono calcolate (oggi, domani, settimana prossima)

### Risultati

Salvati in `results/tool_calling/TIMESTAMP/`

---

## Task 3: Judge/Validator

**Obiettivo:** Validare se l'output di un agente è sicuro da inviare all'utente (quality gate finale).

### Come funziona

1. Dataset: 70 test con user request + tool call eseguito + risultato + ground truth
2. Modello valuta 4 criteri:
   - **appropriate_actions**: Tool/azioni sensate?
   - **relevant_data**: Dati pertinenti?
   - **coherent_response**: Risposta coerente?
   - **complete**: Tutti i dati richiesti?
3. Risponde JSON:
   ```json
   {
     "approved": true/false,
     "appropriate_actions": true/false,
     "relevant_data": true/false,
     "coherent_response": true/false,
     "complete": true/false,
     "reasoning": "..."
   }
   ```

**Regola critica:** Se anche UN SOLO criterio è false → approved = false

### Consistency Tests

8 test vengono eseguiti **5 volte ciascuno** per verificare consistenza:

- Stesso input → stessa decisione?
- Con temperature=0.0 dovrebbe essere ≥90%
- Formula: decisione_più_frequente / 5 run
- Media su tutti gli 8 test

### Esecuzione

```bash
uv run python main_judge.py
```

**Totale richieste:** 70 test normali + 8 test × 5 run = 98 richieste

### Metriche

- **judgment_accuracy**: % decisioni approve/reject corrette
- **false_positive_rate**: % approva quando dovrebbe rigettare (CRITICO, target <10%)
  - Esempio: Approva fattura con importo sbagliato
- **false_negative_rate**: % rigetta quando dovrebbe approvare (meno critico)
  - Esempio: Blocca operazione valida
- **consistency_score**: % consistenza su test ripetuti (target ≥90%)
- **total_cost**, **total_latency**

### Perché FPR è critico?

- **False Positive**: Approva dati sbagliati → rischio finanziario, errori critici, dati sensibili esposti
- **False Negative**: Rigetta dati validi → utente bloccato ma nessun danno reale

### Risultati

Salvati in `results/judge/TIMESTAMP/`

---

## Modelli Configurati

File: `src/model_config.py`

### Cerebras (Free Tier)
- llama3.1-8b: $0.10/$0.10 per 1M tokens
- llama-3.3-70b: $0.85/$1.20 per 1M tokens
- qwen-3-32b: $0.40/$0.80 per 1M tokens

### OpenAI
- gpt-4o-mini: $0.15/$0.60 per 1M tokens
- gpt-4o: $2.50/$10.00 per 1M tokens

### OpenRouter
- gemma-3-12b: $0.03/$0.10 per 1M tokens
- gemma-3-27b: $0.09/$0.16 per 1M tokens
- mistral-7b: (vedi pricing su OpenRouter)

**Per modificare modelli testati:** Edita lista `MODELS_TO_TEST` in ogni `main_*.py`

---

## Riproducibilità

Garantita tramite:

1. **Seed fisso** (default: 42)
   - Controllo del random seed
   - Ordinamento deterministico del dataset per ID

2. **Temperature = 0.0**
   - Output deterministico dai modelli
   - Stesse condizioni per tutti i test

3. **Configurazione salvata**
   - Ogni run salva la configurazione usata
   - Tracciabilità completa dei parametri

---

## Logging

### Local JSON
- `results/{task}/{timestamp}/{model}_results.json`
- Metriche aggregate + configurazione

### Weights & Biases
- Progetto separato per task: `verabench-routing`, `verabench-tool-calling`, `verabench-judge`
- Ogni modello = 1 run
- Dashboard interattive con grafici e tabelle

---

## Visualizzazioni

Script automatico crea:

1. **accuracy_vs_cost.png**: Scatter plot accuracy vs costo
2. **comparison_bars.png**: Bar chart multi-metrica

Supporta automaticamente nomi metriche diversi per task:
- Routing: `routing_accuracy`
- Tool Calling: `tool_selection_accuracy`
- Judge: `judgment_accuracy`

---

## Note Tecniche

### Cleanup Markdown
GPT-4o risponde con ` ```json...``` ` → parsing automatico rimuove i marker

### Rate Limiting
**Rimosso completamente.** Nessun limite artificiale, solo limiti naturali dei provider.

### Consistency Tests
Eseguiti SOLO su test marcati `"category": "consistency_test_X"` nel dataset judge.

---

## Aggiungere Nuove Task

1. Crea `tasks/nuova_task/`:
   - `dataset.json`
   - `prompt.json`
   - `metrics.py` (class MetricsCalculator)

2. Crea `main_nuova_task.py`:
   - Importa `from tasks.nuova_task.metrics import MetricsCalculator`
   - Imposta `MODELS_TO_TEST`
   - Implementa logica di esecuzione

3. Esegui: `uv run python main_nuova_task.py`

---

## License

MIT
