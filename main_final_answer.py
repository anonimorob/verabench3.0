"""
Benchmark per la task di Final Answer (Response Generation).

Testa i modelli selezionati sulla capacità di generare risposte user-friendly
partendo da dati degli agenti upstream (retrieval, tool calling).
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
from tasks.final_answer.metrics import FinalAnswerMetricsCalculator


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

LLM_JUDGE_MODEL = "gpt-4o-mini"


class FinalAnswerBenchmarkRunner:
    """Esegue il benchmark per la task di Final Answer."""
    
    def __init__(self, seed: int = 42):
        load_dotenv()
        random.seed(seed)
        self.seed = seed
        
        # Carica dataset e prompt dalla cartella task
        self.test_cases = load_dataset("tasks/final_answer/dataset.json")
        self.system_prompt = load_prompt("tasks/final_answer/prompt.json")
        
        # Carica prompt config completo per user_prompt_template
        with open("tasks/final_answer/prompt.json", "r", encoding="utf-8") as f:
            prompt_config = json.load(f)
            self.user_prompt_template = prompt_config['user_prompt_template']
        
        # Setup logging
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_logger = ResultLogger("results/final_answer", run_timestamp)
        self.wandb_logger = WandBLogger("verabench-final-answer")
        
        print(f"Task: Final Answer (Response Generation)")
        print(f"Dataset: {len(self.test_cases)} esempi")
        print(f"LLM Judge: {LLM_JUDGE_MODEL} (per DeepEval)")
        print(f"Seed: {seed}\n")
    
    def _format_user_prompt(self, test_case: Dict[str, Any]) -> str:
        """
        Formatta lo user prompt inserendo query, preferences e context.
        
        Args:
            test_case: Test case con user_query, user_preferences, retrieved_context
        
        Returns:
            User prompt formattato
        """
        user_query = test_case['user_query']
        user_preferences = json.dumps(test_case['user_preferences'], indent=2, ensure_ascii=False)
        retrieved_context = json.dumps(test_case['retrieved_context'], indent=2, ensure_ascii=False)
        
        return self.user_prompt_template.format(
            user_query=user_query,
            user_preferences=user_preferences,
            retrieved_context=retrieved_context,
        )
    
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
        metrics = FinalAnswerMetricsCalculator(llm_judge_model=LLM_JUDGE_MODEL)
        
        # Configura W&B
        config = {
            "task": "final_answer",
            "model_id": model_id,
            "model_name": model_name,
            "provider": provider,
            "llm_judge_model": LLM_JUDGE_MODEL,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": self.seed,
            "total_examples": len(self.test_cases),
        }
        self.wandb_logger.start_run(f"final_answer_{model_key}", config)
        
        # Esegui inferenza
        for i, test_case in enumerate(self.test_cases, 1):
            try:
                # Formatta prompt con query + preferences + context
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
                
                # Aggiungi predizione (include chiamate DeepEval)
                print(f"[{i}/{len(self.test_cases)}] Evaluating {test_case['id']}...", end=" ")
                metrics.add_prediction(
                    predicted_response=predicted_response,
                    test_case=test_case,
                    latency=latency,
                    cost=cost,
                )
                print("✓")
                
                if i % 5 == 0:
                    current_metrics = metrics.get_metrics()
                    print(f"  → Faithfulness: {current_metrics['faithfulness_score']:.3f} | "
                          f"Relevancy: {current_metrics['answer_relevancy_score']:.3f} | "
                          f"Conciseness: {current_metrics['conciseness_score']:.3f}")
                
            except Exception as e:
                print(f"✗ ERRORE test {test_case['id']}: {str(e)}")
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
        print(f"Faithfulness Score: {final_metrics['faithfulness_score']:.2%}")
        print(f"Answer Relevancy Score: {final_metrics['answer_relevancy_score']:.2%}")
        print(f"Conciseness Score: {final_metrics['conciseness_score']:.2%}")
        print(f"Overall Quality: {final_metrics['overall_quality']:.2%}")
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
    runner = FinalAnswerBenchmarkRunner()
    
    print("="*60)
    print("BENCHMARK: Final Answer (Response Generation)")
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
