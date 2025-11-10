"""
Sistema di metriche per il benchmark.
"""
from typing import Dict, Any


class MetricsCalculator:
    """Calcola le metriche essenziali per valutare le performance dei modelli."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset delle metriche accumulate."""
        self.predictions = []
        self.ground_truth = []
        self.latencies = []
        self.costs = []
    
    def add_prediction(
        self,
        predicted: str,
        expected: str,
        latency: float,
        cost: float,
    ):
        """
        Aggiunge una singola predizione alle metriche.
        
        Args:
            predicted: Agente predetto dal modello
            expected: Agente corretto
            latency: Latenza in secondi
            cost: Costo in USD
        """
        self.predictions.append(predicted)
        self.ground_truth.append(expected)
        self.latencies.append(latency)
        self.costs.append(cost)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcola e restituisce le metriche essenziali.
        
        Returns:
            Dizionario con accuracy, latenza media e costo totale
        """
        if not self.predictions:
            return {
                "accuracy": 0.0,
                "latency_mean": 0.0,
                "cost_total": 0.0,
                "total_examples": 0,
            }
        
        correct = sum(1 for pred, truth in zip(self.predictions, self.ground_truth) if pred == truth)
        
        return {
            "accuracy": correct / len(self.predictions),
            "latency_mean": sum(self.latencies) / len(self.latencies),
            "cost_total": sum(self.costs),
            "total_examples": len(self.predictions),
        }


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    """
    Calcola il costo di una singola inferenza.
    
    Args:
        prompt_tokens: Numero di token nel prompt
        completion_tokens: Numero di token nella risposta
        input_price_per_1m: Prezzo per 1M token di input in USD
        output_price_per_1m: Prezzo per 1M token di output in USD
    
    Returns:
        Costo totale in USD
    """
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_1m
    output_cost = (completion_tokens / 1_000_000) * output_price_per_1m
    return input_cost + output_cost
