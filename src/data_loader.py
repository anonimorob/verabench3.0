"""
Modulo per caricare dataset e prompt.
"""
import json
from typing import List, Dict, Any


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Carica i test cases dal dataset JSON in ordine deterministico."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Ordina per ID per garantire ordine deterministico
    test_cases = sorted(data['test_cases'], key=lambda x: x['id'])
    return test_cases


def load_prompt(prompt_path: str) -> str:
    """Carica il prompt di sistema dal file JSON."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['system_prompt']
