"""
IGPO Tool Server - Utilities (Web Search Only)
"""

import os
import json
import time
import uuid
import socket
import datetime
import threading
import traceback
from typing import List, Dict, Any


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def string_to_uuid(input_string: str) -> str:
    """Convert string to deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(input_string)))


def get_network_info() -> str:
    """Get hostname for identification."""
    hostname = socket.gethostname()
    try:
        all_ips = socket.gethostbyname_ex(hostname)[2]
        real_ips = [ip for ip in all_ips if not ip.startswith("127.")]
        return hostname + (real_ips[0] if real_ips else "")
    except:
        return hostname


class FileSystemReader:
    """Simple local file system reader."""
    
    def __init__(self, **kwargs):
        pass
    
    def read_file(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()
    
    def write_file(self, file_path: str, content: Any, append: bool = False) -> bool:
        if isinstance(content, str):
            content = content.encode('utf-8')
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'ab' if append else 'wb') as f:
            f.write(content)
        return True
    
    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path) and not os.path.isdir(file_path)


class MessageClient:
    """Task submission client for web search."""
    
    def __init__(self, path: str = './cache/task_queue', **kwargs):
        self.path = path
        self._handler = None
    
    def submit_tasks(self, task_list: List[Dict]) -> List[Dict]:
        """Submit tasks and get results."""
        if not task_list:
            return task_list
        
        # Lazy init handler
        if self._handler is None:
            from tools_server.handler import Handler
            config = self._load_config()
            self._handler = Handler(config)
        
        try:
            return self._handler.handle_all(task_list)
        except Exception as e:
            print(f"[MessageClient] Error: {e}")
            traceback.print_exc()
            for task in task_list:
                if 'content' not in task:
                    task['content'] = f"Error: {str(e)}"
            return task_list
    
    def _load_config(self) -> Dict:
        """Load config from yaml, then apply per-run environment overrides."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except:
            config = {
                'search_engine': 'google',
                'search_top_k': 10,
                'cache_dir': './cache/tool_cache',
            }

        config['mock_mode'] = _parse_bool_env(
            'IGPO_MOCK_SEARCH',
            config.get('mock_mode', False),
        )
        if os.environ.get('IGPO_SEARCH_ENGINE'):
            config['search_engine'] = os.environ['IGPO_SEARCH_ENGINE']
        if os.environ.get('IGPO_SERPER_API_KEY'):
            config['serper_api_key'] = os.environ['IGPO_SERPER_API_KEY']
        if os.environ.get('IGPO_AZURE_BING_SEARCH_SUBSCRIPTION_KEY'):
            config['azure_bing_search_subscription_key'] = os.environ[
                'IGPO_AZURE_BING_SEARCH_SUBSCRIPTION_KEY'
            ]
        if os.environ.get('IGPO_LOCAL_SEARCH_URL'):
            config['local_search_url'] = os.environ['IGPO_LOCAL_SEARCH_URL']
        if os.environ.get('IGPO_LOCAL_SEARCH_TOPK'):
            config['local_search_topk'] = int(os.environ['IGPO_LOCAL_SEARCH_TOPK'])
        if os.environ.get('IGPO_LOCAL_SEARCH_TIMEOUT_SECONDS'):
            config['local_search_timeout_seconds'] = float(
                os.environ['IGPO_LOCAL_SEARCH_TIMEOUT_SECONDS']
            )
        return config
