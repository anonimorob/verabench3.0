"""
Rate limiter per rispettare i limiti di Cerebras.
"""
import time


class RateLimiter:
    """Rate limiter semplice basato su delay fisso tra richieste."""
    
    def __init__(self, requests_per_minute: int = 30):
        """Inizializza il rate limiter."""
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Aspetta se necessario per rispettare il rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
