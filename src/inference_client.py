"""
Client per l'inferenza dei modelli (Cerebras, OpenAI, OpenRouter, Google AI Studio, NVIDIA NIM).
"""
import os
import time
from typing import Dict, Tuple
from openai import OpenAI
import google.generativeai as genai


class ModelInferenceClient:
    
    def __init__(self, model_id: str, provider: str = "cerebras"):
        """        
        Args:
            model_id: ID del modello
            provider: Provider del modello ("cerebras", "openai", "openrouter", "google", o "nvidia")
        """
        self.model_id = model_id
        self.provider = provider
        
        if provider == "cerebras":
            api_key = os.getenv('CEREBRAS_API_KEY')
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY non trovato nel file .env")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.cerebras.ai/v1"
            )
        elif provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY non trovato nel file .env")
            self.client = OpenAI(api_key=api_key)
        elif provider == "openrouter":
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY non trovato nel file .env")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        elif provider == "nvidia":
            api_key = os.getenv('NVIDIA_API_KEY')
            if not api_key:
                raise ValueError("NVIDIA_API_KEY non trovato nel file .env")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1"
            )
        elif provider == "google":
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY non trovato nel file .env")
            genai.configure(api_key=api_key)
            self.client = None  # Google usa API diversa
        else:
            raise ValueError(f"Provider '{provider}' non supportato. Usa 'cerebras', 'openai', 'openrouter', 'nvidia', o 'google'.")
    
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
        start_time = time.time()
        
        # Google AI Studio usa API diversa
        if self.provider == "google":
            try:
                # Combina system e user prompt per Google
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Crea modello Gemini
                model = genai.GenerativeModel(self.model_id)
                
                # Genera risposta
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
                
                latency = time.time() - start_time
                
                # Estrai risposta
                answer = response.text.strip()
                
                # Token usage (Google fornisce metadata)
                token_usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                }
                
                return answer, latency, token_usage
                
            except Exception as e:
                raise RuntimeError(f"Errore durante la generazione con Google AI Studio: {str(e)}")
        
        # OpenAI-compatible providers (Cerebras, OpenAI, OpenRouter, NVIDIA)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
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
            
            # Token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            
            return answer, latency, token_usage
            
        except Exception as e:
            raise RuntimeError(f"Errore durante l'inferenza con {self.model_id}: {str(e)}")
