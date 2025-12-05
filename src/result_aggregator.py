"""
Aggregatore automatico dei risultati per visualizzazione.
"""
import json
from pathlib import Path
from typing import List, Dict, Any


def aggregate_task_results(results_dir: Path, task_name: str) -> List[Dict[str, Any]]:
    """
    Aggrega tutti i file *_results.json in una directory.

    Args:
        results_dir: Directory contenente i file JSON dei risultati
        task_name: Nome della task

    Returns:
        Lista di dict con struttura per visualizer_bubble.py
    """
    aggregated = []

    # Trova tutti i file *_results.json
    result_files = list(results_dir.glob("*_results.json"))

    if not result_files:
        return []

    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Estrai model name dal filename
            model_key = result_file.stem.replace('_results', '')

            # Crea entry aggregata
            entry = {
                'task': task_name,
                'model': data['config'].get('model_name', model_key),
                'variant': 'default',
                'config': data['config'],
                'metrics': data['metrics']
            }

            aggregated.append(entry)

        except Exception:
            continue

    return aggregated


def save_aggregated_results(aggregated: List[Dict[str, Any]], output_path: Path):
    """Salva i risultati aggregati in un file JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
