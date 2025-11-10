"""
Client per l'inferenza dei modelli tramite Cerebras Cloud API.
"""
import os
import time
from typing import Dict, Tuple
from openai import OpenAI


class ModelInferenceClient:
    """Gestisce le chiamate di inferenza ai modelli su Cerebras."""
    
    def __init__(self, model_id: str, api_key: str = None):
        """
        Inizializza il client di inferenza per Cerebras.
        
        Args:
            model_id: ID del modello su Cerebras
            api_key: API key di Cerebras (opzionale se settato in ENV)
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY non trovato. Settalo nel file .env o passa come parametro.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.cerebras.ai/v1"
        )
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        Genera una risposta dal modello.
        
        Args:
            system_prompt: Prompt di sistema
            user_prompt: Prompt dell'utente
            max_new_tokens: Numero massimo di token da generare
            temperature: Temperatura per il sampling (0.0 per deterministico)
            top_p: Parametro top-p per nucleus sampling
        
        Returns:
            Tupla (risposta, latenza_in_secondi, token_usage)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            latency = time.time() - start_time
            
            # Estrai la risposta e i token usage
            answer = response.choices[0].message.content.strip()
            
            # Token usage da Cerebras
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            
            return answer, latency, token_usage
            
        except Exception as e:
            raise RuntimeError(f"Errore durante l'inferenza con {self.model_id}: {str(e)}")
