"""
Benchmark per la task di Tool Calling.

Testa i modelli selezionati sulla capacità di selezionare tool e parametri corretti.
"""
import random
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.data_loader import load_dataset, load_prompt
from src.model_config import get_model_config
from src.inference_client import ModelInferenceClient
from src.metrics import calculate_cost
from src.logger import ResultLogger, WandBLogger
from src.result_aggregator import aggregate_task_results
from src.bubble_visualizer import visualize_results
from tasks.tool_calling.metrics import ToolCallingMetricsCalculator

# PHASE 1: Screening iniziale su dataset ridotto (dataset_short.json)
MODELS_PHASE_1 = [
    "gpt-4o-mini",
    "gpt-4o",
    "openai/gpt-oss-20b"
    # Aggiungere qui tutti i modelli da testare in fase 1
]

# PHASE 2: Valutazione completa su dataset intero (dataset.json)
MODELS_TO_TEST = [
    "openai/gpt-oss-20b"
]


class ToolCallingBenchmarkRunner:
    """Esegue il benchmark per la task di Tool Calling."""

    def __init__(self, seed: int = 42, use_short_dataset: bool = False):
        load_dotenv()
        random.seed(seed)
        self.seed = seed
        self.use_short_dataset = use_short_dataset

        # Carica dataset e prompt dalla cartella task
        dataset_file = "tasks/tool_calling/dataset_short.json" if use_short_dataset else "tasks/tool_calling/dataset.json"
        self.test_cases = load_dataset(dataset_file)
        self.system_prompt = load_prompt("tasks/tool_calling/prompt.json")
        
        # Setup logging
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_logger = ResultLogger("results/tool_calling", run_timestamp)
        self.wandb_logger = WandBLogger("verabench-tool-calling")
        
        print(f"Task: Tool Calling")
        print(f"Dataset: {len(self.test_cases)} esempi")
        print(f"Seed: {seed}\n")
    
    def run_single_model(
        self,
        model_key: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Esegue il benchmark su un singolo modello."""
        model_config = get_model_config(model_key)
        model_id = model_config['id']
        model_name = model_config['name']
        provider = model_config['provider']
        
        print(f"\n{'='*60}")
        print(f"Modello: {model_name} ({provider.upper()})")
        print(f"{'='*60}\n")
        
        # Inizializza
        client = ModelInferenceClient(model_id, provider=provider)
        metrics = ToolCallingMetricsCalculator()
        
        # Configura W&B
        config = {
            "task": "tool_calling",
            "model_id": model_id,
            "model_name": model_name,
            "provider": provider,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": self.seed,
            "total_examples": len(self.test_cases),
        }
        self.wandb_logger.start_run(f"tool_calling_{model_key}", config)
        
        # Esegui inferenza
        for i, test_case in enumerate(self.test_cases, 1):
            try:
                predicted_response, latency, token_usage = client.generate(
                    system_prompt=self.system_prompt,
                    user_prompt=test_case['user_request'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                
                cost = calculate_cost(
                    token_usage['prompt_tokens'],
                    token_usage['completion_tokens'],
                    model_config['input_price_per_1m'],
                    model_config['output_price_per_1m'],
                )
                
                # Print risposta modello
                print(f"\n[{i}/{len(self.test_cases)}] Request: {test_case['user_request'][:60]}...")
                print(f"    Expected Tool: {test_case['expected_tool']}")
                print(f"    Model Response:\n{predicted_response[:200]}{'...' if len(predicted_response) > 200 else ''}")
                
                metrics.add_prediction(
                    predicted_response=predicted_response,
                    expected_tool=test_case['expected_tool'],
                    expected_parameters=test_case['expected_parameters'],
                    latency=latency,
                    cost=cost,
                )
                
                if i % 10 == 0:
                    current_metrics = metrics.get_metrics()
                    print(f"  → Tool Acc: {current_metrics['tool_selection_accuracy']:.3f} | Param Correct: {current_metrics['parameter_correctness']:.3f}\n")
                
            except Exception as e:
                print(f"ERRORE test {test_case['id']}: {str(e)}")
                continue
        
        # Metriche finali
        final_metrics = metrics.get_metrics()
        
        # Log W&B
        self.wandb_logger.log_metrics(final_metrics)
        self.wandb_logger.finish_run()
        
        # Salva localmente
        results = {"config": config, "metrics": final_metrics}
        self.result_logger.save_results(results, model_key)
        
        # Stampa riepilogo
        print(f"\n{'='*60}")
        print(f"RISULTATI {model_name}:")
        print(f"Tool Selection Accuracy: {final_metrics['tool_selection_accuracy']:.2%}")
        print(f"Parameter Correctness: {final_metrics['parameter_correctness']:.2%}")
        print(f"  - Name Accuracy: {final_metrics['parameter_name_accuracy']:.2%}")
        print(f"  - Value Correctness: {final_metrics['parameter_value_correctness']:.2%}")
        print(f"  - Type Accuracy: {final_metrics['parameter_type_accuracy']:.2%}")
        print(f"Total Cost: ${final_metrics['total_cost']:.6f}")
        print(f"Total Latency: {final_metrics['total_latency']:.3f}s")
        print(f"{'='*60}\n")
        
        return results
    
    def run_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Esegue il benchmark su tutti i modelli configurati."""
        all_results = {}
        
        for model_key in MODELS_TO_TEST:
            try:
                results = self.run_single_model(model_key)
                all_results[model_key] = results
            except Exception as e:
                print(f"ERRORE {model_key}: {str(e)}")
                continue
        
        return all_results


def main():
    """Funzione principale."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Tool Calling")
    parser.add_argument("--phase1", action="store_true", help="Esegui Phase 1: screening su dataset ridotto")
    args = parser.parse_args()

    if args.phase1:
        models = MODELS_PHASE_1
        use_short = True
        phase_name = "PHASE 1 - SCREENING"
        print("="*60)
        print(f"{phase_name}")
        print("Dataset: dataset_short.json (10 esempi)")
        print(f"Modelli da testare: {len(models)}")
        print("="*60 + "\n")
    else:
        models = MODELS_TO_TEST
        use_short = False
        phase_name = "PHASE 2 - VALUTAZIONE COMPLETA"
        print("="*60)
        print(f"{phase_name}")
        print("Dataset: dataset.json (completo)")
        print(f"Modelli da testare: {len(models)}")
        print("="*60 + "\n")

    runner = ToolCallingBenchmarkRunner(use_short_dataset=use_short)

    # Esegui solo i modelli selezionati
    def run_selected_models():
        results = {}
        for model_key in models:
            try:
                result = runner.run_single_model(model_key)
                results[model_key] = result
            except Exception as e:
                print(f"ERRORE {model_key}: {str(e)}")
                continue
        return results

    all_results = run_selected_models()

    print("\n" + "="*60)
    print(f"{phase_name} COMPLETATO")
    print(f"Modelli testati: {len(all_results)}")
    print(f"Risultati: {runner.result_logger.results_dir}/")
    print("="*60)

    # Aggrega e visualizza automaticamente
    if all_results:
        print("\n[*] Aggregando risultati e generando visualizzazioni...")
        aggregated = aggregate_task_results(runner.result_logger.results_dir, "tool_calling")
        if aggregated:
            visualize_results(aggregated, "tool_calling", runner.result_logger.results_dir / "visualizations")

if __name__ == "__main__":
    main()
