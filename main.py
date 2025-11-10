"""
Orchestratore principale per eseguire i benchmark.
"""
import random
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.data_loader import load_dataset, load_prompt
from src.model_config import get_model_config, get_all_models
from src.inference_client import ModelInferenceClient
from src.metrics import MetricsCalculator, calculate_cost
from src.logger import ResultLogger, WandBLogger
from src.rate_limiter import RateLimiter
from src.visualizer import BenchmarkVisualizer


class BenchmarkRunner:
    """Esegue il benchmark su uno o più modelli."""
    
    def __init__(
        self,
        dataset_path: str,
        prompt_path: str,
        results_dir: str = "results",
        wandb_project: str = "verabench",
        seed: int = 42,
    ):
        load_dotenv()
        
        # Imposta seed per riproducibilità
        random.seed(seed)
        self.seed = seed
        
        # Carica dataset e prompt (dataset già ordinato per ID)
        self.test_cases = load_dataset(dataset_path)
        self.system_prompt = load_prompt(prompt_path)
        
        # Setup logging
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_logger = ResultLogger(results_dir, run_timestamp)
        self.wandb_logger = WandBLogger(wandb_project)
        
        print(f"Dataset caricato: {len(self.test_cases)} esempi")
        print(f"Seed impostato: {seed}")
        print(f"Ordine test: deterministico (ordinato per ID)")
    
    def run_single_model(
        self,
        model_key: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Esegue il benchmark su un singolo modello."""
        model_config = get_model_config(model_key)
        model_id = model_config['id']
        model_name = model_config['name']
        provider = model_config['provider']
        
        print(f"\n{'='*60}")
        print(f"Benchmark: {model_name} ({provider.upper()})")
        print(f"{'='*60}\n")
        
        # Inizializza
        client = ModelInferenceClient(model_id, provider=provider)
        metrics = MetricsCalculator()
        # Rate limiter: 25 req/min per Cerebras, 20 req/min per OpenRouter, nessun limite per OpenAI
        if provider == "cerebras":
            rate_limiter = RateLimiter(requests_per_minute=25)
        elif provider == "openrouter":
            rate_limiter = RateLimiter(requests_per_minute=20)
        else:
            rate_limiter = None
        
        # Configura W&B
        config = {
            "model_id": model_id,
            "model_name": model_name,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": self.seed,
            "total_examples": len(self.test_cases),
        }
        self.wandb_logger.start_run(model_key, config)
        
        # Esegui inferenza
        for i, test_case in enumerate(self.test_cases, 1):
            # Rispetta rate limit (solo per Cerebras)
            if rate_limiter:
                rate_limiter.wait_if_needed()
            
            try:
                predicted_agent, latency, token_usage = client.generate(
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
                
                metrics.add_prediction(
                    predicted=predicted_agent,
                    expected=test_case['correct_agent'],
                    latency=latency,
                    cost=cost,
                )
                
                if i % 10 == 0:
                    current_metrics = metrics.get_metrics()
                    print(f"Progresso: {i}/{len(self.test_cases)} | Accuracy: {current_metrics['accuracy']:.3f}")
                
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
        print(f"Accuracy: {final_metrics['accuracy']:.2%}")
        print(f"Latenza media: {final_metrics['latency_mean']:.3f}s")
        print(f"Costo totale: ${final_metrics['cost_total']:.6f}")
        print(f"{'='*60}\n")
        
        return results
    
    def run_all_models(self, model_keys: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Esegue il benchmark su tutti i modelli configurati."""
        if model_keys is None:
            model_keys = get_all_models()
        
        all_results = {}
        
        for model_key in model_keys:
            try:
                results = self.run_single_model(model_key)
                all_results[model_key] = results
            except Exception as e:
                print(f"ERRORE {model_key}: {str(e)}")
                continue
        
        return all_results


def main():
    """Funzione principale."""
    runner = BenchmarkRunner(
        dataset_path="vera_agent_router_dataset_v0.1.json",
        prompt_path="router_prompt.json",
        results_dir="results",
        wandb_project="verabench",
    )
    
    print("Inizio benchmark...\n")
    
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
