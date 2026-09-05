import yaml
from pathlib import Path

_CONFIG = None

def load_config(path="config.yaml"):
    global _CONFIG
    if _CONFIG is None:
        with open(path, 'r') as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG

def get(key, default=None):
    cfg = load_config()
    keys = key.split('.')
    val = cfg
    for k in keys:
        val = val.get(k, {})
    return val if val != {} else default