"""
Metriche specifiche per la task di Routing.
"""
from typing import Dict, Any


class RoutingMetricsCalculator:
    """Calcola le metriche per la task di Agent Routing."""
    
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
        Aggiunge una singola predizione.
        
        Args:
            predicted: Agente predetto dal modello
            expected: Agente corretto (ground truth)
            latency: Latenza in secondi
            cost: Costo in USD
        """
        self.predictions.append(predicted)
        self.ground_truth.append(expected)
        self.latencies.append(latency)
        self.costs.append(cost)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcola le metriche per routing.
        
        Returns:
            - routing_accuracy: Percentuale di routing corretti
            - total_cost: Costo totale in USD
            - total_latency: Latenza totale in secondi
        """
        if not self.predictions:
            return {
                "routing_accuracy": 0.0,
                "total_cost": 0.0,
                "total_latency": 0.0,
                "total_examples": 0,
            }
        
        correct = sum(1 for pred, truth in zip(self.predictions, self.ground_truth) if pred == truth)
        
        return {
            "routing_accuracy": correct / len(self.predictions),
            "total_cost": sum(self.costs),
            "total_latency": sum(self.latencies),
            "total_examples": len(self.predictions),
        }
