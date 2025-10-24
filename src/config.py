"""Init for module level config

Define module level config in a module namespace.
Importing the package will result in the config
being accessable within the audio_frontend
namespace: audio_frontend.app_conf, in addition
processing scripts are generated for the
entrypoint.sh file.

TODO expand the config to include worker config
and other config in the process_audio.py file.

Usage:
    >>> from audio_frontend import app_conf as config
"""

import logging
import sys
from typing import Optional

from pyconfs import Configuration

# config dict (expand as needed)
defaults = {
    "logging": {
        "format": "| %(asctime)s | %(levelname)s - %(name)-10s - %(message)s",
        "level": 10,
        "output_full_path": "./logs/logfile.log",
    },
    "embedding": {
        # Model configuration
        "default_model": "all-mini",  # Default BERT model to use
        "cache_dir": None,            # Model cache directory (None = use transformers default)
        
        # Performance settings
        "batch_size": 1024,           # Default batch size for processing (raised from 32)
        "max_length": 512,            # Maximum sequence length for tokenization
        "normalize_embeddings": True,  # Whether to L2 normalize embeddings
        
        # Device settings
        "device": 'cuda',               # 'cuda', 'cpu', or None (auto-detect)
        "gpu_memory_fraction": 0.8,   # GPU memory usage limit
        
        # Processing settings
        "show_progress": False,       # Progress logging
        "use_segmentation": True,     # Long text segmentation
        "segment_overlap": 50,        # Token overlap between segments
        "fallback_model": "distilbert-base-uncased",  # Fallback model if default fails
    },
    "service": {
        # Request limits (two-level keys only!)
        "max_text_length": 8192,      # Maximum text length in characters
        "max_batch_size": 1024,       # Maximum batch size for embedding requests (raised from 100)
        "max_candidates": 1000,       # Maximum similarity comparison candidates
        "max_request_size": 10485760, # Maximum request size in bytes (10MB)
        "rate_limit_per_minute": 600, # Rate limit per IP per minute

        # Concurrency controls (read by service.core.engine_manager)
        # Tune these to match Ray Serve replica concurrency to avoid proxy-level 500s.
        # Note: keys must remain two-level deep due to config helper constraints.
        # Optimized for 2-worker deployment with RTX A4500 GPUs
        "embedding_concurrency_limit": 64,   # Doubled from 32 for 2-worker setup
        "similarity_concurrency_limit": 16,  # Doubled from 8 for 2-worker setup
        "gpu_ops_concurrency_limit": 4,      # Doubled from 2 for 2-worker setup
        "model_loading_concurrency_limit": 2, # Doubled from 1 for 2-worker setup
        "health_checks_concurrency_limit": 50,
    },
}


# config class for wrapping and accessing the config dictionary
class CustConf(Configuration):
    @staticmethod
    def init_logger(self, name: str) -> logging.Logger:
        """Initialize a logger with the logging configuration defined in the
        configuration dictionary.

        Args:
            name: The name of the logger.
        """

        root_logger = logging.getLogger(name)  # type: ignore[attr]
        root_logger.handlers = []

        root_handler = logging.StreamHandler(stream=sys.stdout)
        root_handler.setFormatter(
            logging.Formatter(fmt=self._app_config.get("logging.format"))
        )
        root_logger.addHandler(root_handler)

        file_handler = logging.FileHandler(
            self._app_config.get("logging.output_full_path")
        )
        file_handler.setFormatter(
            logging.Formatter(fmt=self._app_config.get("logging.format"))
        )
        root_logger.addHandler(file_handler)

        root_logger.setLevel(self._app_config.get("logging.level"))

        return root_logger

    def get(self, key: str) -> Optional[None]:
        """Get the value corresponding to the given key string.

        Args:
            key (str): A string in the form "first_level_key.second_level_key".
        """
        return self.data[key.split(".")[0]][key.split(".")[1]]


# config object for reuse
app_conf = CustConf.from_dict(defaults)
