"""
Metriche specifiche per la task di Tool Calling.
"""
import json
from typing import Dict, Any

class ToolCallingMetricsCalculator:
    """Calcola le metriche per la task di Tool Calling."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset delle metriche accumulate."""
        self.tool_correct = []
        self.param_name_correct = []
        self.param_value_correct = []
        self.param_type_correct = []
        self.latencies = []
        self.costs = []
    
    def add_prediction(
        self,
        predicted_response: str,
        expected_tool: str,
        expected_parameters: Dict[str, Any],
        latency: float,
        cost: float,
    ):
        """
        Aggiunge una singola predizione e valuta correttezza.
        
        Args:
            predicted_response: Risposta JSON del modello
            expected_tool: Tool corretto
            expected_parameters: Parametri corretti
            latency: Latenza in secondi
            cost: Costo in USD
        """
        self.latencies.append(latency)
        self.costs.append(cost)
        
        # Parsing della risposta - rimuovi markdown code blocks se presenti
        cleaned_response = predicted_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Rimuovi ```json
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  # Rimuovi ```
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Rimuovi ```
        cleaned_response = cleaned_response.strip()
        
        try:
            predicted = json.loads(cleaned_response)
            predicted_tool = predicted.get("tool", "")
            predicted_params = predicted.get("parameters", {})
        except json.JSONDecodeError:
            # Se non è JSON valido, tutto è sbagliato
            self.tool_correct.append(False)
            self.param_name_correct.append(0.0)
            self.param_value_correct.append(0.0)
            self.param_type_correct.append(0.0)
            return
        
        # 1. Tool Selection Accuracy
        tool_match = predicted_tool == expected_tool
        self.tool_correct.append(tool_match)
        
        # 2. Parameter Correctness (solo se tool corretto)
        if tool_match and expected_parameters:
            # Parameter Name Accuracy
            expected_keys = set(expected_parameters.keys())
            predicted_keys = set(predicted_params.keys())
            name_accuracy = len(expected_keys & predicted_keys) / len(expected_keys) if expected_keys else 1.0
            self.param_name_correct.append(name_accuracy)
            
            # Parameter Value Correctness
            value_matches = 0
            for key in expected_keys:
                if key in predicted_params:
                    if self._values_match(predicted_params[key], expected_parameters[key]):
                        value_matches += 1
            value_accuracy = value_matches / len(expected_keys) if expected_keys else 1.0
            self.param_value_correct.append(value_accuracy)
            
            # Parameter Type Validation
            type_matches = 0
            for key in expected_keys:
                if key in predicted_params:
                    if type(predicted_params[key]) == type(expected_parameters[key]):
                        type_matches += 1
            type_accuracy = type_matches / len(expected_keys) if expected_keys else 1.0
            self.param_type_correct.append(type_accuracy)
        else:
            # Se tool sbagliato o no parametri attesi
            self.param_name_correct.append(0.0)
            self.param_value_correct.append(0.0)
            self.param_type_correct.append(0.0)
    
    def _values_match(self, predicted, expected) -> bool:
        """Verifica se due valori sono equivalenti."""
        if predicted == expected:
            return True
        # Gestisci confronto case-insensitive per stringhe
        if isinstance(predicted, str) and isinstance(expected, str):
            return predicted.lower().strip() == expected.lower().strip()
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcola le metriche per tool calling.
        
        Returns:
            - tool_selection_accuracy: % tool corretti
            - parameter_name_accuracy: % nomi parametri corretti
            - parameter_value_correctness: % valori parametri corretti
            - parameter_type_accuracy: % tipi parametri corretti
            - parameter_correctness: Media delle 3 metriche parametri
            - total_cost: Costo totale
            - total_latency: Latenza totale
        """
        if not self.tool_correct:
            return {
                "tool_selection_accuracy": 0.0,
                "parameter_name_accuracy": 0.0,
                "parameter_value_correctness": 0.0,
                "parameter_type_accuracy": 0.0,
                "parameter_correctness": 0.0,
                "total_cost": 0.0,
                "total_latency": 0.0,
                "total_examples": 0,
            }
        
        tool_acc = sum(self.tool_correct) / len(self.tool_correct)
        param_name_acc = sum(self.param_name_correct) / len(self.param_name_correct)
        param_value_acc = sum(self.param_value_correct) / len(self.param_value_correct)
        param_type_acc = sum(self.param_type_correct) / len(self.param_type_correct)
        param_correctness = (param_name_acc + param_value_acc + param_type_acc) / 3
        
        return {
            "tool_selection_accuracy": tool_acc,
            "parameter_name_accuracy": param_name_acc,
            "parameter_value_correctness": param_value_acc,
            "parameter_type_accuracy": param_type_acc,
            "parameter_correctness": param_correctness,
            "total_cost": sum(self.costs),
            "total_latency": sum(self.latencies),
            "total_examples": len(self.tool_correct),
        }
