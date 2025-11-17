"""
Benchmark per la task di Judge/Validator.

Testa i modelli selezionati sulla capacità di validare output prima di inviarli all'utente.
Include test di consistency (5 run per consistency_test).
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
from src.visualizer import BenchmarkVisualizer
from tasks.judge.metrics import JudgeMetricsCalculator


# Modelli da testare per questa task
MODELS_TO_TEST = [
    "gpt-4o-mini",
    "gpt-4o",
    "llama-3.3-70b",
    "llama3.1-8b",
    "gemma-3-12b",
    "gemma-3-27b",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash"
]

# Numero di run per consistency tests
CONSISTENCY_RUNS = 5


class JudgeBenchmarkRunner:
    """Esegue il benchmark per la task di Judge/Validator."""
    
    def __init__(self, seed: int = 42):
        load_dotenv()
        random.seed(seed)
        self.seed = seed
        
        # Carica dataset e prompt dalla cartella task
        self.test_cases = load_dataset("tasks/judge/dataset.json")
        self.system_prompt = load_prompt("tasks/judge/prompt.json")
        
        # Setup logging
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_logger = ResultLogger("results/judge", run_timestamp)
        self.wandb_logger = WandBLogger("verabench-judge")
        
        print(f"Task: Judge/Validator")
        print(f"Dataset: {len(self.test_cases)} esempi")
        print(f"Consistency runs: {CONSISTENCY_RUNS} per test")
        print(f"Seed: {seed}\n")
    
    def _format_user_prompt(self, test_case: Dict[str, Any]) -> str:
        """Formatta il prompt utente con i dati del test case."""
        # Il prompt template richiede: user_request, tool_name, tool_parameters, tool_result
        import json
        
        user_request = test_case['user_request']
        tool_name = test_case['tool_call']['name']
        tool_parameters = json.dumps(test_case['tool_call']['parameters'], indent=2, ensure_ascii=False)
        tool_result = json.dumps(test_case['tool_result'], indent=2, ensure_ascii=False)
        
        # Il template è in prompt.json come user_prompt_template
        return f"""USER REQUEST:
{user_request}

TOOL CALL ESEGUITO:
Tool: {tool_name}
Parametri: {tool_parameters}

RISULTATO OTTENUTO:
{tool_result}

Valuta se questo output è appropriato e può essere inoltrato all'utente."""
    
    def run_single_model(
        self,
        model_key: str,
        max_new_tokens: int = 300,
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
        metrics = JudgeMetricsCalculator()
        
        # Configura W&B
        config = {
            "task": "judge",
            "model_id": model_id,
            "model_name": model_name,
            "provider": provider,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": self.seed,
            "total_examples": len(self.test_cases),
            "consistency_runs": CONSISTENCY_RUNS,
        }
        self.wandb_logger.start_run(f"judge_{model_key}", config)
        
        # Esegui inferenza
        total_requests = 0
        for i, test_case in enumerate(self.test_cases, 1):
            category = test_case.get('category', '')
            is_consistency_test = 'consistency_test' in category
            
            # Determina quante volte eseguire questo test
            num_runs = CONSISTENCY_RUNS if is_consistency_test else 1
            
            for run_idx in range(num_runs):
                total_requests += 1
                
                try:
                    user_prompt = self._format_user_prompt(test_case)
                    
                    predicted_response, latency, token_usage = client.generate(
                        system_prompt=self.system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    
                    cost = calculate_cost(
                        token_usage['prompt_tokens'],
                        token_usage['completion_tokens'],
                        model_config['input_price_per_1m'],
                        model_config['output_price_per_1m'],
                    )
                    
                    metrics.add_prediction(
                        predicted_response=predicted_response,
                        ground_truth=test_case['ground_truth'],
                        latency=latency,
                        cost=cost,
                        test_case_id=test_case['id'] if is_consistency_test else None,
                    )
                    
                except Exception as e:
                    print(f"ERRORE test {test_case['id']} run {run_idx+1}/{num_runs}: {str(e)}")
                    continue
            
            # Progress update ogni 10 test cases
            if i % 10 == 0:
                current_metrics = metrics.get_metrics()
                print(f"Progresso: {i}/{len(self.test_cases)} test cases | "
                      f"Accuracy: {current_metrics['judgment_accuracy']:.3f} | "
                      f"Requests: {total_requests}")
        
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
        print(f"Judgment Accuracy: {final_metrics['judgment_accuracy']:.2%}")
        print(f"False Positive Rate: {final_metrics['false_positive_rate']:.2%} (target <10%)")
        print(f"False Negative Rate: {final_metrics['false_negative_rate']:.2%}")
        print(f"Consistency Score: {final_metrics['consistency_score']:.2%} (target ≥90%)")
        print(f"Total Cost: ${final_metrics['total_cost']:.6f}")
        print(f"Total Latency: {final_metrics['total_latency']:.3f}s")
        print(f"Total Requests: {total_requests}")
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
    runner = JudgeBenchmarkRunner()
    
    print("="*60)
    print("BENCHMARK: Judge/Validator")
    print("="*60 + "\n")
    
    all_results = runner.run_all_models()
    
    # Crea grafici
    if all_results:
        visualizer = BenchmarkVisualizer(runner.result_logger.results_dir)
        visualizer.create_all_plots(all_results)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETATO")
    print(f"Modelli testati: {len(all_results)}")
    print(f"Risultati: {runner.result_logger.results_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
