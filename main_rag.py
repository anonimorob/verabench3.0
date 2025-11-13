"""
Benchmark per la task di RAG (Retrieval Augmented Generation).

Testa i modelli selezionati sulla capacità di recuperare dati dal database interno
di Vera AI, verificando permissions, preferences e conversation history.
"""
import json
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
from tasks.rag.metrics import RAGMetricsCalculator


# Modelli da testare per questa task
MODELS_TO_TEST = [
    "gpt-4o-mini",
    #"gpt-4o",
    #"gemma-3-12b",
]


class RAGBenchmarkRunner:
    """Esegue il benchmark per la task di RAG."""
    
    def __init__(self, seed: int = 42):
        load_dotenv()
        random.seed(seed)
        self.seed = seed
        
        # Carica dataset e prompt dalla cartella task
        self.test_cases = load_dataset("tasks/rag/dataset.json")
        self.system_prompt = load_prompt("tasks/rag/prompt.json")
        
        # Carica prompt config completo per user_prompt_template
        with open("tasks/rag/prompt.json", "r", encoding="utf-8") as f:
            prompt_config = json.load(f)
            self.user_prompt_template = prompt_config['user_prompt_template']
        
        # Carica mock database
        with open("tasks/rag/mock_database.json", "r", encoding="utf-8") as f:
            self.mock_database = json.load(f)
        
        # Setup logging
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_logger = ResultLogger("results/rag", run_timestamp)
        self.wandb_logger = WandBLogger("verabench-rag")
        
        print(f"Task: RAG (Retrieval Augmented Generation)")
        print(f"Dataset: {len(self.test_cases)} esempi")
        print(f"Mock Database: {len(self.mock_database['users'])} utenti, {len(self.mock_database['companies'])} companies")
        print(f"Seed: {seed}\n")
    
    def _format_user_prompt(self, test_case: Dict[str, Any]) -> str:
        """
        Formatta lo user prompt inserendo il database completo come contesto.
        
        Il database viene passato in JSON per simulare l'accesso a dati interni.
        """
        database_json = json.dumps(self.mock_database, indent=2, ensure_ascii=False)
        user_phone = test_case['user_phone']
        user_query = test_case['user_query']
        
        return self.user_prompt_template.format(
            database_json=database_json,
            user_phone=user_phone,
            user_query=user_query,
        )
    
    def run_single_model(
        self,
        model_key: str,
        max_new_tokens: int = 500,
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
        metrics = RAGMetricsCalculator()
        
        # Configura W&B
        config = {
            "task": "rag",
            "model_id": model_id,
            "model_name": model_name,
            "provider": provider,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": self.seed,
            "total_examples": len(self.test_cases),
        }
        self.wandb_logger.start_run(f"rag_{model_key}", config)
        
        # Esegui inferenza
        for i, test_case in enumerate(self.test_cases, 1):
            try:
                # Formatta prompt con database context
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
                    test_case=test_case,
                    latency=latency,
                    cost=cost,
                )
                
                if i % 5 == 0:
                    current_metrics = metrics.get_metrics()
                    print(f"Progresso: {i}/{len(self.test_cases)} | "
                          f"Security: {current_metrics['security_score']:.3f} | "
                          f"Accuracy: {current_metrics['retrieval_accuracy']:.3f}")
                
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
        print(f"Security Score: {final_metrics['security_score']:.2%} {'✓ PASS' if final_metrics['security_score'] >= 0.95 else '✗ FAIL'}")
        print(f"Retrieval Accuracy: {final_metrics['retrieval_accuracy']:.2%}")
        print(f"Completeness Score: {final_metrics['completeness_score']:.2%}")
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
    runner = RAGBenchmarkRunner()
    
    print("="*60)
    print("BENCHMARK: RAG (Retrieval Augmented Generation)")
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
