"""
Sistema di logging per salvare risultati localmente e su Weights & Biases.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import wandb

class ResultLogger:
    """Gestisce il salvataggio dei risultati in locale."""
    
    def __init__(self, results_dir: str = "results", run_timestamp: str = None):
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_timestamp = run_timestamp
        self.results_dir = Path(results_dir) / run_timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Risultati verranno salvati in: {self.results_dir}")
    
    def save_results(self, results: Dict[str, Any], model_name: str):
        """Salva i risultati in un file JSON locale."""
        # Sostituisce / con _ per evitare sottodirectory
        safe_model_name = model_name.replace('/', '_')
        filename = self.results_dir / f"{safe_model_name}_results.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Risultati salvati in: {filename}")


class WandBLogger:
    """Gestisce il logging su Weights & Biases."""
    
    def __init__(self, project_name: str = "verabench"):
        self.project_name = project_name
        self.current_run = None
    
    def start_run(self, model_name: str, config: Dict[str, Any]):
        """Inizia un nuovo run su W&B."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        
        self.current_run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=config,
            reinit=True,
        )
        
        print(f"W&B run iniziato: {run_name}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Logga le metriche su W&B."""
        if self.current_run:
            wandb.log(metrics)
    
    def finish_run(self):
        """Chiude il run corrente su W&B."""
        if self.current_run:
            wandb.finish()
            self.current_run = None
