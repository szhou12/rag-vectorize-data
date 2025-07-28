# rag/embedders/base_embedder.py
import logging
from abc import ABC

class BaseEmbeddingModel(ABC):
    def __init__(self):
        self.model = None  # Placeholder for the actual embedding model
        self.logger = self._init_logger()
    
    def _init_logger(self):
        """
        Initialize a logger specific to the subclass that extends this base class.
        """
        logger = logging.getLogger(self.__class__.__name__)  # Use the subclass name for the logger
        logger.propagate = True  # TODO: Ensure messages propagate to root logger
        return logger