import time
from typing import Any, Dict

class QuantraCache:
    """In-memory cache with TTL (Time To Live)."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set a value in the cache with a TTL."""
        self.cache[key] = {
            "value": value,
            "expiry": time.time() + ttl_seconds
        }
        
    def get(self, key: str) -> Any:
        """Get a value from the cache. Returns None if expired or not found."""
        if key in self.cache:
            item = self.cache[key]
            if time.time() < item["expiry"]:
                return item["value"]
            else:
                # Expired
                del self.cache[key]
        return None
        
    def invalidate(self, key: str):
        """Remove an item from the cache."""
        if key in self.cache:
            del self.cache[key]
            
    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()

# Global singleton cache instance
cache = QuantraCache()
