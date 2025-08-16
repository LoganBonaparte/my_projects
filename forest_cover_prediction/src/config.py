import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    config_path = os.path.join(BASE_DIR, 'config', 'params.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_path(key):
    config = load_config()
    return os.path.join(BASE_DIR, config['paths'][key])