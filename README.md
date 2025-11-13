# VERABENCH# VERABENCH# VERABENCH



Sistema modulare di benchmark per testare modelli LLM su task specifiche degli agenti Vera AI.



## OverviewSistema modulare di benchmark per testare modelli LLM su task specifiche degli agenti Vera AI.Sistema di benchmark per testare modelli LLM sul routing degli agenti Vera AI.



VERABENCH è un framework per valutare le performance di LLM su 4 task critiche:



1. **Routing**: Selezionare l'agente corretto per una richiesta utente## Overview## Caratteristiche

2. **Tool Calling**: Scegliere il tool giusto con parametri corretti

3. **Judge**: Validare output prima di inviarli all'utente (quality gate)

4. **RAG**: Recuperare dati dal database interno verificando permessi e completeness

VERABENCH è un framework per valutare le performance di LLM su 3 task critiche:- Test di modelli LLM su dataset di routing degli agenti

Ogni task ha dataset dedicato, metriche specifiche e script indipendente.

- Supporto per inference tramite Cerebras Cloud 

---

1. **Routing**: Selezionare l'agente corretto per una richiesta utente- Metriche: accuratezza, latenza, costo

## Architettura

2. **Tool Calling**: Scegliere il tool giusto con parametri corretti  - Logging locale (JSON) e su Weights & Biases

```

tasks/3. **Judge**: Validare output prima di inviarli all'utente (quality gate)- Sistema riproducibile con seed configurabile

  routing/          → Dataset, prompt, metriche routing

  tool_calling/     → Dataset, prompt, metriche tool calling- Test a parità di condizioni (temperatura = 0.0)

  judge/            → Dataset, prompt, metriche judge

  rag/              → Dataset, prompt, metriche RAG, mock databaseOgni task ha dataset dedicato, metriche specifiche e script indipendente.

  

main_routing.py     → Benchmark routing## Limiti Cerebras Free Tier

main_tool_calling.py → Benchmark tool calling  

main_judge.py       → Benchmark judge---

main_rag.py         → Benchmark RAG

```- **30 requests/minute**, 60K tokens/minute



### Inference Providers## Architettura- 



Chiamiamo direttamente i provider con SDK OpenAI:



- **Cerebras**: `base_url="https://api.cerebras.ai/v1"` (free tier)```

- **OpenAI**: base_url default (proprietario)

- **OpenRouter**: `base_url="https://openrouter.ai/api/v1"` (gateway multi-modello)tasks/Con 50 test cases e 4 modelli (200 requests totali), benchmark eseguibile ~72 volte al giorno



Stesso SDK, cambia solo `base_url`.  routing/          → Dataset, prompt, metriche routing



---  tool_calling/     → Dataset, prompt, metriche tool calling



## Setup  judge/            → Dataset, prompt, metriche judge## Setup



### 1. Installazione  



```powershellmain_routing.py     → Benchmark routing### 1. Installazione con uv

# Installa uv

pip install uvmain_tool_calling.py → Benchmark tool calling  



# Sincronizza dipendenzemain_judge.py       → Benchmark judgeAssicurati di avere `uv` installato:

uv sync

``````



### 2. Configurazione delle API Keys```powershell



Crea un file `.env` nella root del progetto:### Inference Providers



```bashpip install uv

# Cerebras (free tier)Chiamiamo direttamente i provider con SDK OpenAI:```

CEREBRAS_API_KEY=your_cerebras_api_key_here



# OpenAI (proprietario)- **Cerebras**: `base_url="https://api.cerebras.ai/v1"` (free tier)

OPENAI_API_KEY=your_openai_api_key_here### 2. Crea e attiva l'ambiente virtuale



# OpenRouter (gateway)- **OpenAI**: base_url default (proprietario)

OPENROUTER_API_KEY=your_openrouter_api_key_here

- **OpenRouter**: `base_url="https://openrouter.ai/api/v1"` (gateway multi-modello)```powershell

# Weights & Biases (opzionale)

WANDB_API_KEY=your_wandb_api_key_here

```

Stesso SDK, cambia solo `base_url`.uv sync

---

```

## Task 1: Routing

---

**Obiettivo**: Testare la capacità del modello di selezionare l'agente corretto per ogni richiesta utente.

### 3. Configurazione delle API Keys

**Dataset**: 50 test cases in `tasks/routing/dataset.json`  

**Script**: `main_routing.py`## Setup



### MetricheCrea un file `.env` nella root del progetto:

- **routing_accuracy**: % agenti corretti selezionati

- **total_cost**: Costo totale in USD### 1. Installazione

- **total_latency**: Latenza totale in secondi

```powershell

### Esecuzione

```powershell```bashCopy-Item .env.example .env

uv run python main_routing.py

```# Installa uv```



---pip install uv



## Task 2: Tool CallingModifica `.env` e aggiungi le tue API keys:



**Obiettivo**: Testare la capacità del modello di selezionare il tool corretto e specificare i parametri giusti.# Sincronizza dipendenze



**Dataset**: 50 test cases in `tasks/tool_calling/dataset.json`  uv sync```env

**Script**: `main_tool_calling.py`

```CEREBRAS_API_KEY=your_cerebras_api_key_here

### Metriche

- **tool_selection_accuracy**: % tool correttiWANDB_API_KEY=your_wandb_api_key_here

- **parameter_name_accuracy**: % nomi parametri corretti

- **parameter_value_correctness**: % valori parametri corretti### 2. Configurazione API Keys```

- **parameter_type_accuracy**: % tipi parametri corretti

- **parameter_correctness**: Media delle 3 metriche parametri

- **total_cost**: Costo totale

- **total_latency**: Latenza totale```bash**Come ottenere le API keys:**



### Esecuzione# Copia template

```powershell

uv run python main_tool_calling.pycp .env.example .env- **Cerebras API Key**: 

```

  1. Vai su https://cloud.cerebras.ai/

### Note Tecniche

- Rimuove automaticamente markdown code blocks (```json...```) prima del parsing# Modifica .env con le tue keys  2. Crea un account gratuito

- GPT-4o tende a rispondere con markdown, pulizia automatica garantisce compatibilità

```  3. Vai su API Keys e crea una nuova chiave

---



## Task 3: Judge

File `.env`:- **Weights & Biases API Key**: 

**Obiettivo**: Validare output degli agenti prima dell'invio all'utente (quality gate).

```env  1. Vai su https://wandb.ai/settings

**Dataset**: 70 test cases in `tasks/judge/dataset.json`

- 62 test normaliCEREBRAS_API_KEY=your_key_here  2. Copia la tua API key dalla sezione "Danger Zone"

- 8 test di consistency (eseguiti 5 volte ciascuno)

OPENAI_API_KEY=your_key_here  

**Script**: `main_judge.py`

OPENROUTER_API_KEY=your_key_here## Utilizzo

### Metriche

- **judgment_accuracy**: % decisioni corrette (approve/reject)WANDB_API_KEY=your_key_here

- **false_positive_rate**: % approva quando dovrebbe rifiutare (**CRITICO - target <10%**)

- **false_negative_rate**: % rifiuta quando dovrebbe approvare```### Eseguire il benchmark completo

- **consistency_score**: % consistency su test ripetuti (target ≥90%)

- **total_cost**: Costo totale

- **total_latency**: Latenza totale

**Dove ottenere le keys:**```powershell

### Consistency Tests

8 test vengono eseguiti 5 volte ciascuno per verificare consistenza delle decisioni:- Cerebras: https://cloud.cerebras.ai/ (free tier)uv run python main.py

- Ogni test deve produrre la stessa decisione (approve/reject) in tutte le 5 run

- Consistency score = media delle % di consistency per gli 8 test- OpenAI: https://platform.openai.com/api-keys```

- Identifica non-determinismo anche con temperature=0.0

- OpenRouter: https://openrouter.ai/keys

### Perché FPR è Critico?

- **False Positive**: Approva dati errati → cliente riceve info sbagliate → rischio finanziario- W&B: https://wandb.ai/settingsQuesto eseguirà il benchmark su tutti i 4 modelli configurati.

- **False Negative**: Rifiuta dati corretti → cliente bloccato → disagio ma no rischio finanziario



### Esecuzione

```powershell---### Struttura dei risultati

uv run python main_judge.py

```



---## Task 1: RoutingI risultati vengono salvati in due formati:



## Task 4: RAG (Retrieval Augmented Generation)



**Obiettivo**: Testare la capacità del modello di recuperare dati dal database interno di Vera AI verificando permessi, completeness e accuracy.**Obiettivo:** Dato un messaggio utente, selezionare l'agente giusto.1. **File JSON locali** nella directory `results/`:



**Dataset**: 25 test cases in `tasks/rag/dataset.json`   - `{model_key}_{timestamp}.json` - Metriche aggregate

- 9 test su permission retrieval

- 7 test su security (access control)### Come funziona   - `{model_key}_{timestamp}_predictions.json` - Predizioni dettagliate

- 6 test su user preferences

- 5 test su conversation history



**Mock Database**: `tasks/rag/mock_database.json`1. Dataset: 50 test con richieste utente + agente corretto2. **Dashboard Weights & Biases**:

- 5 utenti con diversi ruoli e permessi (Sales Manager, HR Director, Inventory Analyst, CEO, Junior Sales Rep)

- 2 companies (Alpha Trading LLC, Beta Enterprises Inc)2. Modello riceve: system prompt (lista agenti) + user request   - Ogni modello viene loggato come un run separato

- Permessi su 3 sistemi: ERP, CRM, HR

3. Deve rispondere con: nome dell'agente   - Visualizzazione interattiva delle metriche

**Script**: `main_rag.py`

   - Tabelle con tutte le predizioni

### Metriche

- **security_score**: Score binario 0.0-1.0 (**CRITICO - target 1.0**)### Esecuzione

  - 1.0 se TUTTI i security check passano

  - 0.0 se anche UN SOLO check fallisce## Metriche Calcolate

- **retrieval_accuracy**: % accuratezza dati recuperati (exact match)

- **completeness_score**: % completezza risposta (tutti i campi necessari presenti)```bash

- **total_cost**: Costo totale

- **total_latency**: Latenza totaleuv run python main_routing.py### Accuratezza



### Security Checks```- Percentuale di predizioni corrette

La security è binaria e verifica:

1. **Access Denial**: Se `should_deny_access=true`, deve negare accesso con `access_granted: false`- Calcolata come: `correct_predictions / total_examples`

2. **Forbidden Fields**: Non deve restituire campi forbidden (cross-user data)

3. **Critical Permissions**: Campi `can_read`, `can_write`, `modules` devono essere ESATTIModelli testati (configurabili in `MODELS_TO_TEST`):



Se **anche un solo check fallisce**, security_score = 0.0 per quel test.- gpt-4o-mini### Latenza



### Come Funziona- gpt-4o- **Mean**: Latenza media in secondi

Il database mock viene passato nel prompt come contesto in formato JSON, simulando l'accesso a dati interni:

- **Median**: Latenza mediana

```json

DATABASE CONTEXT:### Metriche- **Min/Max**: Valori minimo e massimo

{

  "users": [...],- **Std**: Deviazione standard

  "companies": [...]

}- **routing_accuracy**: % routing corretti



USER PHONE NUMBER: +971501234567- **total_cost**: Costo totale in USD### Costo

USER QUERY: "Can I write to the ERP sales module?"

```- **total_latency**: Latenza totale in secondi- **Total**: Costo totale del run in USD



Il modello deve:

1. Identificare l'utente dal phone number

2. Verificare i permessi### Risultati

3. Recuperare solo i dati pertinenti alla query

4. Rispondere in JSON con `access_granted` e `retrieved_data`## Riproducibilità



### Perché Security è Critico?Salvati in `results/routing/TIMESTAMP/`:

- **Errato can_write=true**: Utente può modificare dati che non dovrebbe → corruzione database

- **Errato can_read=true**: Utente vede dati riservati (es. payroll) → violazione privacy- `{model}_results.json`: Metriche aggregateIl sistema garantisce la riproducibilità attraverso:

- **Cross-user data**: Accesso a dati di altri utenti → security breach

- Grafici: accuracy vs cost, comparison bars

### Test Categories

- **permission_retrieval**: Verifica read/write access su ERP, CRM, HR- W&B dashboard: `verabench-routing`1. **Seed fisso** (default: 42)

- **security**: Test con `should_deny_access=true` per accessi non autorizzati

- **user_preferences**: Retrieval di lingua, timezone, currency, notifications   - Controllo del random seed

- **conversation_history**: Recupero storico conversazioni con timestamp ed entities

---   - Stesso ordinamento dei test cases

### Esecuzione

```powershell

uv run python main_rag.py

```## Task 2: Tool Calling2. **Temperatura = 0.0**



---   - Output deterministico dai modelli



## Modelli Configurati**Obiettivo:** Dato un messaggio utente, selezionare tool corretto con parametri esatti.   - Stesse condizioni per tutti i test



In `src/model_config.py`:



```python### Come funziona3. **Configurazione salvata**

MODELS = {

    "llama-3.3-70b": {   - Ogni run salva la configurazione usata

        "name": "Llama 3.3 70B",

        "id": "meta-llama/Llama-3.3-70B-Instruct",1. Dataset: 50 test con richieste → expected tool + parameters   - Tracciabilità completa dei parametri

        "provider": "cerebras",

        "input_price_per_1m": 0.0,2. Modello riceve: system prompt (tools disponibili) + user request

        "output_price_per_1m": 0.0,

    },3. Deve rispondere JSON:

    "gpt-4o-mini": {   ```json

        "name": "GPT-4o mini",   {

        "id": "gpt-4o-mini",     "tool": "nome_tool",

        "provider": "openai",     "parameters": { "param1": "value1", ... }

        "input_price_per_1m": 0.15,   }

        "output_price_per_1m": 0.60,   ```

    },

    "gpt-4o": {### Esecuzione

        "name": "GPT-4o",

        "id": "gpt-4o",```bash

        "provider": "openai",uv run python main_tool_calling.py

        "input_price_per_1m": 2.50,```

        "output_price_per_1m": 10.00,

    },### Metriche

    # ... altri modelli

}- **tool_selection_accuracy**: % tool corretti

```- **parameter_correctness**: Media di:

  - **parameter_name_accuracy**: % nomi parametri corretti

---  - **parameter_value_correctness**: % valori parametri corretti

  - **parameter_type_accuracy**: % tipi parametri corretti

## Riproducibilità- **total_cost**, **total_latency**



- **Seed**: 42 (configurabile)### Note tecniche

- **Temperature**: 0.0 (deterministica)

- **Ordinamento**: Dataset ordinato per ID test case- GPT-4o risponde con markdown code blocks → cleanup automatico prima del parsing

- Parametri non specificati devono essere `null`

---- Date relative vengono calcolate (oggi, domani, settimana prossima)



## Logging### Risultati



### Locale (JSON)Salvati in `results/tool_calling/TIMESTAMP/`

Risultati salvati in:

```---

results/

  routing/YYYYMMDD_HHMMSS/## Task 3: Judge/Validator

    model_name.json

  tool_calling/YYYYMMDD_HHMMSS/**Obiettivo:** Validare se l'output di un agente è sicuro da inviare all'utente (quality gate finale).

    model_name.json

  judge/YYYYMMDD_HHMMSS/### Come funziona

    model_name.json

  rag/YYYYMMDD_HHMMSS/1. Dataset: 70 test con user request + tool call eseguito + risultato + ground truth

    model_name.json2. Modello valuta 4 criteri:

```   - **appropriate_actions**: Tool/azioni sensate?

   - **relevant_data**: Dati pertinenti?

### Weights & Biases   - **coherent_response**: Risposta coerente?

Run automatici con config e metriche per ogni modello.   - **complete**: Tutti i dati richiesti?

3. Risponde JSON:

---   ```json

   {

## Visualizzazioni     "approved": true/false,

     "appropriate_actions": true/false,

Grafici generati automaticamente in `results/task_name/timestamp/`:     "relevant_data": true/false,

     "coherent_response": true/false,

- `accuracy_vs_cost.png`: Scatter plot accuracy vs costo     "complete": true/false,

- `comparison_bars.png`: Bar chart con accuracy, latenza, costo     "reasoning": "..."

   }

Il visualizzatore supporta automaticamente tutte e 4 le task con fallback per nomi metriche diversi.   ```



---**Regola critica:** Se anche UN SOLO criterio è false → approved = false



## Note Tecniche### Consistency Tests



### Markdown Cleanup (GPT-4o)8 test vengono eseguiti **5 volte ciascuno** per verificare consistenza:

GPT-4o risponde spesso con markdown code blocks:

```json- Stesso input → stessa decisione?

{...}- Con temperature=0.0 dovrebbe essere ≥90%

```- Formula: decisione_più_frequente / 5 run

- Media su tutti gli 8 test

Tutti i metrics calculator rimuovono automaticamente ` ```json` e ` ``` ` prima del parsing.

### Esecuzione

### Rate Limiting

Rimosso completamente da tutte le task. Gestione errori HTTP 429 delegata al client.```bash

uv run python main_judge.py

### Provider Selection```

Il client sceglie il `base_url` in base al campo `provider` nel model config.

**Totale richieste:** 70 test normali + 8 test × 5 run = 98 richieste

---

### Metriche

## Aggiungere Nuove Task

- **judgment_accuracy**: % decisioni approve/reject corrette

1. Crea cartella `tasks/nuova_task/`- **false_positive_rate**: % approva quando dovrebbe rigettare (CRITICO, target <10%)

2. Aggiungi `dataset.json` e `prompt.json`  - Esempio: Approva fattura con importo sbagliato

3. Crea `metrics.py` con classe `NuovaTaskMetricsCalculator`- **false_negative_rate**: % rigetta quando dovrebbe approvare (meno critico)

4. Crea `main_nuova_task.py` basandoti sui template esistenti  - Esempio: Blocca operazione valida

5. Aggiorna `src/visualizer.py` per supportare nuove metriche (fallback)- **consistency_score**: % consistenza su test ripetuti (target ≥90%)

- **total_cost**, **total_latency**

---

### Perché FPR è critico?

## License

- **False Positive**: Approva dati sbagliati → rischio finanziario, errori critici, dati sensibili esposti

MIT- **False Negative**: Rigetta dati validi → utente bloccato ma nessun danno reale


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

