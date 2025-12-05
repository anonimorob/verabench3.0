"""
Metriche specifiche per la task di Final Answer.

Valuta:
- Faithfulness: Fedeltà ai fatti tramite DeepEval 
- Answer Relevancy: Pertinenza della risposta tramite DeepEval
- Conciseness: Concisione per WhatsApp (rule-based: max caratteri e linee)
"""
import os
from typing import Dict, Any
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import json


class FinalAnswerMetricsCalculator:
    """Calcola le metriche per la task di Final Answer."""
    
    def __init__(self, llm_judge_model: str = "gpt-4o-mini"):
        """
        Inizializza il calculator con modello LLM judge.
        Args:
            llm_judge_model: Modello da usare per DeepEval metrics (default: gpt-4o-mini)
        """
        self.llm_judge_model = llm_judge_model
        self.reset()
        
        # Verifica che OPENAI_API_KEY sia disponibile per DeepEval
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY non trovata. DeepEval richiede OpenAI API key.")
    
    def reset(self):
        """Reset delle metriche accumulate."""
        self.faithfulness_scores = []
        self.answer_relevancy_scores = []
        self.conciseness_scores = []
        self.latencies = []
        self.costs = []
    
    def add_prediction(
        self,
        predicted_response: str,
        test_case: Dict[str, Any],
        latency: float,
        cost: float,
    ):
        """
        Aggiunge una singola predizione e valuta correttezza.
        Args:
            predicted_response: Risposta generata dal modello
            test_case: Test case completo con user_query, retrieved_context, evaluation_config
            latency: Latenza in secondi
            cost: Costo in USD
        """
        self.latencies.append(latency)
        self.costs.append(cost)
        
        user_query = test_case['user_query']
        retrieved_context = test_case['retrieved_context']
        evaluation_config = test_case.get('evaluation_config', {})
        
        # Estrai context come stringa per DeepEval
        context_str = self._format_context(retrieved_context)
        
        # 1. FAITHFULNESS (DeepEval)
        faithfulness_threshold = evaluation_config.get('faithfulness_threshold', 0.85)
        faithfulness_score = self._evaluate_faithfulness(
            actual_output=predicted_response,
            retrieval_context=[context_str],
            threshold=faithfulness_threshold,
        )
        self.faithfulness_scores.append(faithfulness_score)
        
        # 2. ANSWER RELEVANCY (DeepEval)
        relevancy_threshold = evaluation_config.get('answer_relevancy_threshold', 0.85)
        relevancy_score = self._evaluate_answer_relevancy(
            input_query=user_query,
            actual_output=predicted_response,
            retrieval_context=[context_str],
            threshold=relevancy_threshold,
        )
        self.answer_relevancy_scores.append(relevancy_score)
        
        # 3. CONCISENESS (Rule-Based)
        max_characters = evaluation_config.get('max_characters', 500)
        max_lines = evaluation_config.get('max_lines', 10)
        conciseness_score = self._evaluate_conciseness(
            response=predicted_response,
            max_characters=max_characters,
            max_lines=max_lines,
        )
        self.conciseness_scores.append(conciseness_score)
    
    def _format_context(self, retrieved_context: Dict[str, Any]) -> str:
        """
        Formatta il context per DeepEval in stringa leggibile.
        Args:
            retrieved_context: Context da agenti upstream
        Returns:
            Stringa formattata del context
        """
        if 'data' in retrieved_context:
            # Context da retrieval_agent
            return json.dumps(retrieved_context['data'], indent=2, ensure_ascii=False)
        elif 'tool_result' in retrieved_context:
            # Context da tool_calling_agent
            return json.dumps(retrieved_context['tool_result'], indent=2, ensure_ascii=False)
        else:
            # Fallback: converti tutto a stringa
            return json.dumps(retrieved_context, indent=2, ensure_ascii=False)
    
    def _evaluate_faithfulness(
        self,
        actual_output: str,
        retrieval_context: list,
        threshold: float,
    ) -> float:
        """
        Valuta faithfulness usando DeepEval FaithfulnessMetric.
        Returns:
            Score 0.0-1.0 (1.0 = completamente fedele ai fatti)
        """
        try:
            # Crea test case per DeepEval
            test_case = LLMTestCase(
                input="N/A",  # Non serve per faithfulness
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )
            
            # Crea metrica con threshold
            metric = FaithfulnessMetric(
                threshold=threshold,
                model=self.llm_judge_model,
                include_reason=False,  # Non serve reason per benchmark
            )
            
            # Misura (sincrono)
            metric.measure(test_case)
            
            # Ritorna score normalizzato 0-1
            return metric.score if metric.score is not None else 0.0
            
        except Exception as e:
            print(f"ERRORE DeepEval Faithfulness: {str(e)}")
            return 0.0
    
    def _evaluate_answer_relevancy(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: list,
        threshold: float,
    ) -> float:
        """
        Valuta answer relevancy usando DeepEval AnswerRelevancyMetric.
        Returns:
            Score 0.0-1.0 (1.0 = completamente rilevante)
        """
        try:
            # Crea test case per DeepEval
            test_case = LLMTestCase(
                input=input_query,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )
            # Crea metrica con threshold
            metric = AnswerRelevancyMetric(
                threshold=threshold,
                model=self.llm_judge_model,
                include_reason=False,
            )
            # Misura
            metric.measure(test_case)
            # Ritorna score normalizzato 0-1
            return metric.score if metric.score is not None else 0.0
            
        except Exception as e:
            print(f"ERRORE DeepEval Answer Relevancy: {str(e)}")
            return 0.0
    
    def _evaluate_conciseness(
        self,
        response: str,
        max_characters: int,
        max_lines: int,
    ) -> float:
        """
        Valuta conciseness con regole per WhatsApp (rule-based).
        Returns:
            Score 0.0-1.0 (1.0 = perfettamente conciso)
        """
        # Conta caratteri
        char_count = len(response)
        char_score = 1.0 if char_count <= max_characters else max(0.0, 1.0 - (char_count - max_characters) / max_characters)
        
        # Conta linee
        line_count = len(response.split('\n'))
        line_score = 1.0 if line_count <= max_lines else max(0.0, 1.0 - (line_count - max_lines) / max_lines)
        # Pondera 70% caratteri, 30% linee
        conciseness_score = (char_score * 0.7) + (line_score * 0.3)
        
        return conciseness_score
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcola le metriche per Final Answer.
        
        Returns:
            - faithfulness_score: Media score fedeltà ai fatti (0-1)
            - answer_relevancy_score: Media score pertinenza (0-1)
            - conciseness_score: Media score concisione (0-1)
            - overall_quality: Media delle 3 metriche
            - total_cost: Costo totale della task (escluso costo judge LLM)
            - total_latency: Latenza totale della task (esclusa latenza judge)
        """
        if not self.faithfulness_scores:
            return {
                "faithfulness_score": 0.0,
                "answer_relevancy_score": 0.0,
                "conciseness_score": 0.0,
                "overall_quality": 0.0,
                "total_cost": 0.0,
                "total_latency": 0.0,
                "total_examples": 0,
            }
        
        faithfulness_avg = sum(self.faithfulness_scores) / len(self.faithfulness_scores)
        relevancy_avg = sum(self.answer_relevancy_scores) / len(self.answer_relevancy_scores)
        conciseness_avg = sum(self.conciseness_scores) / len(self.conciseness_scores)
        
        # Overall quality: media delle 3 metriche
        overall_quality = (faithfulness_avg + relevancy_avg + conciseness_avg) / 3
        
        return {
            "faithfulness_score": faithfulness_avg,
            "answer_relevancy_score": relevancy_avg,
            "conciseness_score": conciseness_avg,
            "overall_quality": overall_quality,
            "total_cost": sum(self.costs),
            "total_latency": sum(self.latencies),
            "total_examples": len(self.faithfulness_scores),
        }
