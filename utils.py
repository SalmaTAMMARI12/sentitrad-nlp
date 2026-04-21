import re
import time
import hashlib
from typing import Optional
 

_result_cache = {}
CACHE_MAX_SIZE = 100
 
def cache_key(text: str, operation: str, lang: str = '') -> str:
    content = f'{operation}:{lang}:{text}'
    return hashlib.md5(content.encode('utf-8')).hexdigest()
 
def get_from_cache(key: str) -> Optional[dict]:
    entry = _result_cache.get(key)
    if entry:
        entry['hits'] = entry.get('hits', 0) + 1
        return entry['result']
    return None
 
def save_to_cache(key: str, result) -> None:
    if len(_result_cache) >= CACHE_MAX_SIZE:
        min_key = min(_result_cache, key=lambda k: _result_cache[k].get('hits', 0))
        del _result_cache[min_key]
    _result_cache[key] = {'result': result, 'hits': 0, 'time': time.time()}
 
def get_cache_stats() -> dict:
    return {
        'size': len(_result_cache),
        'max_size': CACHE_MAX_SIZE,
        'total_hits': sum(e.get('hits', 0) for e in _result_cache.values())
    }
 

def init_history(session_state) -> None:
    if 'history' not in session_state:
        session_state.history = []
    if 'total_analyses' not in session_state:
        session_state.total_analyses = 0
 
def add_to_history(session_state, text, sentiment, translation) -> None:
    session_state.history.append({
        'text': text,
        'text_short': text[:60] + '...' if len(text) > 60 else text,
        'sentiment': sentiment,
        'translation': translation,
        'timestamp': time.strftime('%H:%M:%S')
    })
    session_state.total_analyses += 1
    if len(session_state.history) > 20:
        session_state.history.pop(0)
 
def clean_text(text: str, max_length: int = 512) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_length]
