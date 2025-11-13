"""
Modulo per creare visualizzazioni dei risultati del benchmark.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

class BenchmarkVisualizer:
    """Crea grafici per confrontare i risultati dei modelli."""
    
    def __init__(self, output_dir: Path):
        """
        Inizializza il visualizzatore.
        
        Args:
            output_dir: Directory dove salvare i grafici
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_accuracy_vs_cost(self, results: Dict[str, Dict[str, Any]]):
        """Crea un grafico accuracy vs costo totale."""
        if not results:
            return
        
        model_names = []
        accuracies = []
        costs = []
        
        for model_key, result in results.items():
            metrics = result['metrics']
            model_names.append(result['config']['model_name'])
            
            # Supporta diversi nomi di accuracy in base alla task
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'] * 100)
            elif 'routing_accuracy' in metrics:
                accuracies.append(metrics['routing_accuracy'] * 100)
            elif 'tool_selection_accuracy' in metrics:
                accuracies.append(metrics['tool_selection_accuracy'] * 100)
            elif 'judgment_accuracy' in metrics:
                accuracies.append(metrics['judgment_accuracy'] * 100)
            elif 'retrieval_accuracy' in metrics:
                accuracies.append(metrics['retrieval_accuracy'] * 100)
            else:
                accuracies.append(0.0)
            
            # Costo totale
            if 'cost_total' in metrics:
                costs.append(metrics['cost_total'])
            elif 'total_cost' in metrics:
                costs.append(metrics['total_cost'])
            else:
                costs.append(0.0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(costs, accuracies, s=200, alpha=0.6, c=range(len(model_names)), cmap='viridis')
        
        for i, name in enumerate(model_names):
            ax.annotate(name, (costs[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        if max(costs) / min(costs) > 10:
            ax.set_xscale('log')
        
        ax.set_xlabel('Costo Totale (USD)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Confronto Modelli: Accuracy vs Costo', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'accuracy_vs_cost.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafico salvato in: {output_path}")
    
    def plot_comparison_bars(self, results: Dict[str, Dict[str, Any]]):
        """Crea grafici a barre per confrontare accuracy, latenza e costo."""
        if not results:
            return
        
        model_names = []
        accuracies = []
        latencies = []
        costs = []
        
        for model_key, result in results.items():
            metrics = result['metrics']
            model_names.append(result['config']['model_name'])
            
            # Supporta diversi nomi di accuracy
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'] * 100)
            elif 'routing_accuracy' in metrics:
                accuracies.append(metrics['routing_accuracy'] * 100)
            elif 'tool_selection_accuracy' in metrics:
                accuracies.append(metrics['tool_selection_accuracy'] * 100)
            elif 'judgment_accuracy' in metrics:
                accuracies.append(metrics['judgment_accuracy'] * 100)
            elif 'retrieval_accuracy' in metrics:
                accuracies.append(metrics['retrieval_accuracy'] * 100)
            else:
                accuracies.append(0.0)
            
            # Latenza
            if 'latency_mean' in metrics:
                latencies.append(metrics['latency_mean'])
            elif 'total_latency' in metrics:
                latencies.append(metrics['total_latency'] / metrics.get('total_examples', 1))
            else:
                latencies.append(0.0)
            
            # Costo
            if 'cost_total' in metrics:
                costs.append(metrics['cost_total'])
            elif 'total_cost' in metrics:
                costs.append(metrics['total_cost'])
            else:
                costs.append(0.0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0].set_ylabel('Accuracy (%)', fontsize=11)
        axes[0].set_title('Accuracy per Modello', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(model_names, latencies, color='lightcoral', alpha=0.7)
        axes[1].set_ylabel('Latenza Media (s)', fontsize=11)
        axes[1].set_title('Latenza Media per Modello', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(model_names, costs, color='lightgreen', alpha=0.7)
        axes[2].set_ylabel('Costo Totale (USD)', fontsize=11)
        axes[2].set_title('Costo Totale per Modello', fontsize=12, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        if max(costs) / min(costs) > 10:
            axes[2].set_yscale('log')
        
        # Salva il grafico
        output_path = self.output_dir / 'comparison_bars.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafico salvato in: {output_path}")
    
    def create_all_plots(self, results: Dict[str, Dict[str, Any]]):
        """
        Crea tutti i grafici disponibili.
        
        Args:
            results: Dizionario con i risultati di tutti i modelli
        """
        print("\nGenerazione grafici...")
        self.plot_accuracy_vs_cost(results)
        self.plot_comparison_bars(results)
        print("Grafici completati!\n")
