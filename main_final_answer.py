"""
Benchmark per la task di Final Answer (Response Generation).

Testa i modelli selezionati sulla capacità di generare risposte user-friendly.
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
from src.result_aggregator import aggregate_task_results
from src.bubble_visualizer import visualize_results
from tasks.final_answer.metrics import FinalAnswerMetricsCalculator
import argparse


# PHASE 1: Screening iniziale su dataset ridotto (dataset_short.json)
MODELS_PHASE_1 = [
    "openai/gpt-oss-20b",
    "gpt-4o-mini",
    # Aggiungere qui tutti i modelli da testare in fase 1
]

# PHASE 2: Valutazione completa su dataset intero (dataset.json)
MODELS_TO_TEST = [
    # Aggiungere qui solo i migliori modelli selezionati in fase 1
]

LLM_JUDGE_MODEL = "gpt-4o-mini" # Modello LLM usato per DeepEval come giudice delle risposte generate

class FinalAnswerBenchmarkRunner:
    """Esegue il benchmark per Final Answer."""
    
    def __init__(self, seed: int = 42, use_short_dataset: bool = False):
        load_dotenv()
        random.seed(seed)
        self.seed = seed
        self.use_short_dataset = use_short_dataset

        # Carica dataset e prompt dalla cartella tasks
        dataset_file = "tasks/final_answer/dataset_short.json" if use_short_dataset else "tasks/final_answer/dataset.json"
        self.test_cases = load_dataset(dataset_file)
        self.system_prompt = load_prompt("tasks/final_answer/prompt.json")
        
        # Carica prompt config completo per user_prompt_template
        with open("tasks/final_answer/prompt.json", "r", encoding="utf-8") as f:
            prompt_config = json.load(f)
            self.user_prompt_template = prompt_config['user_prompt_template']
        
        # Setup logging
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_logger = ResultLogger("results/final_answer", run_timestamp)
        self.wandb_logger = WandBLogger("verabench-final-answer")
        
        print(f"Task: Final Answer")
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
        
        # Inizializza modello e metriche
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
                
                # Debug: stampa risposta modello
                print(f"\n[{i}/{len(self.test_cases)}] Query: {test_case['user_query'][:60]}...")
                print(f"    Category: {test_case['category']}")
                print(f"    Model Response:\n{predicted_response[:200]}{'...' if len(predicted_response) > 200 else ''}")
                print(f"    Evaluating with DeepEval...", end=" ")
                
                # Aggiungi predizione (include chiamate DeepEval)
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
                          f"Conciseness: {current_metrics['conciseness_score']:.3f}\n")
                
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
    parser = argparse.ArgumentParser(description="Benchmark Final Answer")
    parser.add_argument("--phase1", action="store_true", help="Esegui Phase 1: screening su dataset ridotto (dataset_short.json)")
    args = parser.parse_args()

    # Seleziona modelli e dataset in base alla fase
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

    runner = FinalAnswerBenchmarkRunner(use_short_dataset=use_short)

    # Esegui solo i modelli selezionati per questa fase
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
        print("\n[*] Aggregando risultati e generando visualizzazioni")
        aggregated = aggregate_task_results(runner.result_logger.results_dir, "final_answer")
        if aggregated:
            visualize_results(aggregated, "final_answer", runner.result_logger.results_dir / "visualizations")


if __name__ == "__main__":
    main()
