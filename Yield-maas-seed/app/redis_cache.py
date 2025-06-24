import redis, pickle
from .config import REDIS_URL

class RedisConnectionError(RuntimeError):
    """Raised when Redis is unreachable or SET/GET fails."""
    pass

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_model(model, key: str):
    """Cache a model using a unique key."""
    try:
        redis_client.set(key, pickle.dumps(model))
    except redis.exceptions.RedisError as e:
        raise RedisConnectionError(str(e)) from e

def get_cached_model(key: str, retries: int = 0):
    """Retrieve a cached model by key with optional retry attempts."""
    for attempt in range(retries + 1):
        try:
            blob = redis_client.get(key)
            return pickle.loads(blob) if blob else None
        except redis.exceptions.RedisError as e:
            if attempt == retries:
                raise RedisConnectionError(str(e)) from e
