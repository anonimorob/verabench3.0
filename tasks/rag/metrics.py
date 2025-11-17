"""
Metriche specifiche per la task di RAG (Retrieval Augmented Generation).

Valuta:
- Retrieval Accuracy: Correttezza dei dati recuperati
- Completeness: Completezza della risposta (via LLM judge se configurato)
"""
import json
from typing import Dict, Any, List


class RAGMetricsCalculator:
    """Calcola le metriche per la task di RAG."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset delle metriche accumulate."""
        self.retrieval_accuracy_scores = []
        self.completeness_scores = []
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
            predicted_response: Risposta JSON del modello
            test_case: Test case completo con expected_output e evaluation_config
            latency: Latenza in secondi
            cost: Costo in USD
        """
        self.latencies.append(latency)
        self.costs.append(cost)
        
        expected_output = test_case['expected_output']
        evaluation_config = test_case.get('evaluation_config', {})
        should_deny_access = test_case.get('should_deny_access', False)
        
        # Parsing della risposta - rimuovi markdown code blocks se presenti
        cleaned_response = predicted_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        try:
            predicted = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Se non è JSON valido, tutto fallisce
            self.retrieval_accuracy_scores.append(0.0)
            self.completeness_scores.append(0.0)
            return
        
        # 1. RETRIEVAL ACCURACY (solo se non security violation)
        if not should_deny_access:
            accuracy_score = self._evaluate_retrieval_accuracy(
                predicted=predicted,
                expected_output=expected_output,
            )
            self.retrieval_accuracy_scores.append(accuracy_score)
        else:
            # Per security violations, accuracy non applicabile
            self.retrieval_accuracy_scores.append(0.0)
        
        # 2. COMPLETENESS (rule-based o LLM judge)
        use_llm_judge = evaluation_config.get('use_llm_judge', False)
        if use_llm_judge:
            # TODO: Implementare LLM judge per semantic evaluation
            # Per ora usiamo rule-based
            completeness_score = self._evaluate_completeness_rule_based(
                predicted=predicted,
                expected_output=expected_output,
                should_deny_access=should_deny_access,
            )
        else:
            completeness_score = self._evaluate_completeness_rule_based(
                predicted=predicted,
                expected_output=expected_output,
                should_deny_access=should_deny_access,
            )
        self.completeness_scores.append(completeness_score)
    
    def _evaluate_retrieval_accuracy(
        self,
        predicted: Dict[str, Any],
        expected_output: Dict[str, Any],
    ) -> float:
        """
        Valuta l'accuratezza dei dati recuperati (exact match per campi critici).
        
        Calcola % di campi corretti rispetto ai campi expected.
        """
        retrieved_data = predicted.get('retrieved_data', {})
        if not retrieved_data:
            return 0.0
        
        # Campi da verificare (solo quelli in expected_output)
        total_fields = 0
        correct_fields = 0
        
        def compare_nested(pred_dict: Any, exp_dict: Any) -> tuple:
            """Compara ricorsivamente nested dictionaries."""
            if not isinstance(exp_dict, dict):
                # Valori primitivi o liste
                return (1, 1) if pred_dict == exp_dict else (1, 0)
            
            fields_total = 0
            fields_correct = 0
            for key, exp_value in exp_dict.items():
                fields_total += 1
                pred_value = pred_dict.get(key) if isinstance(pred_dict, dict) else None
                
                if isinstance(exp_value, dict):
                    # Nested dict
                    nested_total, nested_correct = compare_nested(pred_value, exp_value)
                    fields_total += nested_total - 1  # -1 perché già contato sopra
                    fields_correct += nested_correct
                elif isinstance(exp_value, list):
                    # Liste: set comparison per ordine-indipendente
                    if isinstance(pred_value, list):
                        if set(pred_value) == set(exp_value):
                            fields_correct += 1
                else:
                    # Valori primitivi
                    if pred_value == exp_value:
                        fields_correct += 1
            
            return (fields_total, fields_correct)
        
        total_fields, correct_fields = compare_nested(retrieved_data, expected_output)
        
        return correct_fields / total_fields if total_fields > 0 else 0.0
    
    def _evaluate_completeness_rule_based(
        self,
        predicted: Dict[str, Any],
        expected_output: Dict[str, Any],
        should_deny_access: bool,
    ) -> float:
        """
        Valuta la completeness: % di campi expected presenti nella risposta.
        
        Diverso da accuracy: qui verifichiamo che ci siano tutti i campi,
        non che i valori siano corretti (quello è accuracy).
        """
        if should_deny_access:
            # Per security violations, completeness = presence of error field
            return 1.0 if 'error' in predicted else 0.0
        
        retrieved_data = predicted.get('retrieved_data', {})
        if not retrieved_data:
            return 0.0
        
        def count_expected_fields(exp_dict: Any) -> int:
            """Conta ricorsivamente tutti i campi expected."""
            if not isinstance(exp_dict, dict):
                return 1
            
            count = 0
            for value in exp_dict.values():
                if isinstance(value, dict):
                    count += count_expected_fields(value)
                else:
                    count += 1
            return count
        
        def count_present_fields(pred_dict: Any, exp_dict: Any) -> int:
            """Conta quanti campi expected sono presenti in predicted."""
            if not isinstance(exp_dict, dict):
                return 1 if pred_dict is not None else 0
            
            count = 0
            for key, exp_value in exp_dict.items():
                pred_value = pred_dict.get(key) if isinstance(pred_dict, dict) else None
                if isinstance(exp_value, dict):
                    count += count_present_fields(pred_value, exp_value)
                else:
                    count += 1 if key in pred_dict else 0
            return count
        
        total_expected = count_expected_fields(expected_output)
        total_present = count_present_fields(retrieved_data, expected_output)
        
        return total_present / total_expected if total_expected > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcola le metriche per RAG.
        
        Returns:
            - retrieval_accuracy: % accuratezza dei dati recuperati (exact match)
            - completeness_score: % completezza risposta (campi presenti)
            - total_cost: Costo totale
            - total_latency: Latenza totale
        """
        if not self.retrieval_accuracy_scores:
            return {
                "retrieval_accuracy": 0.0,
                "completeness_score": 0.0,
                "total_cost": 0.0,
                "total_latency": 0.0,
                "total_examples": 0,
            }
        
        # Retrieval accuracy: media
        retrieval_accuracy = sum(self.retrieval_accuracy_scores) / len(self.retrieval_accuracy_scores)
        
        # Completeness: media
        completeness_score = sum(self.completeness_scores) / len(self.completeness_scores)
        
        return {
            "retrieval_accuracy": retrieval_accuracy,
            "completeness_score": completeness_score,
            "total_cost": sum(self.costs),
            "total_latency": sum(self.latencies),
            "total_examples": len(self.retrieval_accuracy_scores),
        }
