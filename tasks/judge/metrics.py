"""
Metriche specifiche per la task di Judge/Validator.
"""
import json
from typing import Dict, Any, List
from collections import Counter


class JudgeMetricsCalculator:
    """Calcola le metriche per la task di Judge/Validator."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset delle metriche accumulate."""
        self.predictions = []  # Lista di dict con prediction e ground_truth
        self.latencies = []
        self.costs = []
        self.consistency_results = {}  # {test_case_id: [approved_bool, approved_bool, ...]}
    
    def add_prediction(
        self,
        predicted_response: str,
        ground_truth: Dict[str, Any],
        latency: float,
        cost: float,
        test_case_id: str = None,
    ):
        """
        Aggiunge una singola predizione.
        
        Args:
            predicted_response: Risposta JSON del modello
            ground_truth: Ground truth con should_approve e altri campi
            latency: Latenza in secondi
            cost: Costo in USD
            test_case_id: ID del test case (per consistency tracking)
        """
        self.latencies.append(latency)
        self.costs.append(cost)
        
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
            approved = predicted.get("approved", None)
        except json.JSONDecodeError:
            # Se non parsabile, considera come rejected
            approved = False
        
        # Salva per calcolo metriche
        self.predictions.append({
            "predicted_approved": approved,
            "should_approve": ground_truth.get("should_approve", None),
        })
        
        # Track per consistency se è un consistency test
        if test_case_id and "consistency_test" in test_case_id:
            if test_case_id not in self.consistency_results:
                self.consistency_results[test_case_id] = []
            self.consistency_results[test_case_id].append(approved)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calcola le metriche per judge.
        
        Returns:
            - judgment_accuracy: % decisioni approve/reject corrette
            - false_positive_rate: % approva cose da rigettare (CRITICO)
            - false_negative_rate: % rigetta cose valide
            - consistency_score: % consistenza su test ripetuti
            - total_cost: Costo totale
            - total_latency: Latenza totale
        """
        if not self.predictions:
            return {
                "judgment_accuracy": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "consistency_score": 0.0,
                "total_cost": 0.0,
                "total_latency": 0.0,
                "total_examples": 0,
            }
        
        # 1. Judgment Accuracy
        correct = sum(
            1 for p in self.predictions
            if p["predicted_approved"] == p["should_approve"]
        )
        judgment_accuracy = correct / len(self.predictions)
        
        # 2. False Positive Rate (approva quando dovrebbe rigettare)
        should_reject = [p for p in self.predictions if not p["should_approve"]]
        if should_reject:
            false_positives = sum(
                1 for p in should_reject
                if p["predicted_approved"] == True
            )
            fpr = false_positives / len(should_reject)
        else:
            fpr = 0.0
        
        # 3. False Negative Rate (rigetta quando dovrebbe approvare)
        should_approve = [p for p in self.predictions if p["should_approve"]]
        if should_approve:
            false_negatives = sum(
                1 for p in should_approve
                if p["predicted_approved"] == False
            )
            fnr = false_negatives / len(should_approve)
        else:
            fnr = 0.0
        
        # 4. Consistency Score
        consistency_score = self._calculate_consistency()
        
        return {
            "judgment_accuracy": judgment_accuracy,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "consistency_score": consistency_score,
            "total_cost": sum(self.costs),
            "total_latency": sum(self.latencies),
            "total_examples": len(self.predictions),
        }
    
    def _calculate_consistency(self) -> float:
        """
        Calcola consistency score: stesso input → stessa decisione?
        
        Per ogni test case consistency, calcola:
        - Quante volte la decisione più frequente appare / totale run
        - Media su tutti i test case
        """
        if not self.consistency_results:
            return 1.0  # Nessun test consistency, assume perfetto
        
        consistency_scores = []
        
        for test_id, decisions in self.consistency_results.items():
            if len(decisions) < 2:
                continue  # Serve almeno 2 run
            
            # Trova la decisione più comune
            counter = Counter(decisions)
            most_common_count = counter.most_common(1)[0][1]
            
            # Score = quante volte appare la decisione più comune / totale
            score = most_common_count / len(decisions)
            consistency_scores.append(score)
        
        if not consistency_scores:
            return 1.0
        
        return sum(consistency_scores) / len(consistency_scores)
