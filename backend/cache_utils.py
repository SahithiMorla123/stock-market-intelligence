"""Disk-backed result cache so heavy models (FinBERT, BiLSTM) only run
when there's no fresh cached answer yet, instead of on every request."""
import functools
import hashlib
import pickle
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return CACHE_DIR / f"{digest}.pkl"


def cache_get(key: str, ttl_seconds: float):
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            saved_at, value = pickle.load(f)
        if time.time() - saved_at > ttl_seconds:
            return None
        return value
    except Exception:
        return None


def cache_set(key: str, value) -> None:
    path = _cache_path(key)
    with open(path, "wb") as f:
        pickle.dump((time.time(), value), f)


def cached(ttl_seconds: float):
    """Decorator: cache a function's return value on disk for ttl_seconds,
    keyed by its name and arguments."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__module__}.{fn.__qualname__}:{args}:{sorted(kwargs.items())}"
            cached_value = cache_get(key, ttl_seconds)
            if cached_value is not None:
                return cached_value
            result = fn(*args, **kwargs)
            cache_set(key, result)
            return result
        return wrapper
    return decorator
