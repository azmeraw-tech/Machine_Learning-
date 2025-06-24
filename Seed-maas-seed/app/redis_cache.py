import redis, pickle
from .config import REDIS_URL, MODEL_CACHE_KEY

class RedisConnectionError(RuntimeError):
    """Raised when Redis is unreachable or SET/GET fails."""
    pass

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_model(model):
    try:
        redis_client.set(MODEL_CACHE_KEY, pickle.dumps(model))
    except redis.exceptions.RedisError as e:
        raise RedisConnectionError(str(e)) from e

def get_cached_model(retries: int = 0):  # let api.py pass RETRY_LIMIT
    for attempt in range(retries + 1):
        try:
            blob = redis_client.get(MODEL_CACHE_KEY)
            return pickle.loads(blob) if blob else None
        except redis.exceptions.RedisError as e:
            if attempt == retries:
                raise RedisConnectionError(str(e)) from e
