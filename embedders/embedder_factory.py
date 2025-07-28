import logging
from typing import Dict, Type, Optional, Any, Union
from rag.embedders.base_embedder import BaseEmbeddingModel
from rag.embedders.bge_embedder import BgeEmbedding
from rag.embedders.openai_embedder import OpenAIEmbedding

logger = logging.getLogger(__name__)

class EmbedderFactory:
    """
    Factory for creating embedding model instances.
    Provides a centralized way to create different types of embedding models
    with consistent configuration and error handling.
    """

    # Registry of predefined embedder types to their implementing classes
    _embedder_classes: Dict[str, Type[BaseEmbeddingModel]] = {
        "bge": BgeEmbedding,
        "openai": OpenAIEmbedding,
    }

    # Registry of predefined model configurations
    _model_configs = {
        # BGE Embedding Models
        "bge_small_en": {"type": "bge", "model_name": "BAAI/bge-small-en-v1.5"},
        "bge_small_zh": {"type": "bge", "model_name": "BAAI/bge-small-zh-v1.5"},
        "bge_base_en": {"type": "bge", "model_name": "BAAI/bge-base-en-v1.5"},
        "bge_base_zh": {"type": "bge", "model_name": "BAAI/bge-base-zh-v1.5"},
        "bge_large_en": {"type": "bge", "model_name": "BAAI/bge-large-en-v1.5"},
        "bge_large_zh": {"type": "bge", "model_name": "BAAI/bge-large-zh-v1.5"},
        # OpenAI Embedding Models
        "openai_default": {"type": "openai"},
        "openai_3_large": {"type": "openai", "model_name": "text-embedding-3-large"},
    }

    @classmethod
    def register_embedder_class(cls, name: str, embedder_class: Type[BaseEmbeddingModel]) -> None:
        """
        Register a new embedder class.
        
        Args:
            name: Name to register the embedder class under. e.g., "bge"
            embedder_class: The embedder class to register
        """
        cls._embedder_classes[name] = embedder_class
        logger.info(f"Registered embedder class: {name}")

    @classmethod
    def register_model_config(cls, name: str, config: Dict[str, Any]) -> None:
        """
        Register a new model configuration.
        
        Args:
            name: Name to register the model configuration under
            config: The model configuration dictionary
        """
        if "type" not in config:
            raise ValueError("Model configuration must include 'type' key")

        if config["type"] not in cls._embedder_classes:
            raise ValueError(f"Unknown embedder type: {config['type']}")

        cls._model_configs[name] = config
        logger.info(f"Registered model configuration: {name}")

    @classmethod
    def create(cls, model_name: str) -> BaseEmbeddingModel:
        """
        Create an embedder instance based on a predefined model configuration.

        Args:
            model_name: Name of the predefined model configuration
            
        Returns:
            An instance of BaseEmbeddingModel
            
        Raises:
            ValueError: If the model name is not registered
        """
        if model_name not in cls._model_configs:
            raise ValueError(f"Unknown model configuration: {model_name}")

        config = cls._model_configs[model_name]
        embedder_type = config["type"]

        # Create the embedder instance
        try:
            # Remove the type from the config as it's not needed for instantiation
            instance_config = {k: v for k, v in config.items() if k != "type"}
            return cls.create_custom(embedder_type, **instance_config)
        except Exception as e:
            logger.error(f"Failed to create embedder from configuration {model_name}: {e}")
            raise

    @classmethod
    def create_custom(cls, embedder_type: str, **kwargs) -> BaseEmbeddingModel:
        """
        Create an embedder instance with custom parameters.
        
        Args:
            embedder_type: Type of embedder to create
            **kwargs: Additional arguments for the embedder constructor
            
        Returns:
            An instance of BaseEmbeddingModel
            
        Raises:
            ValueError: If the embedder type is not registered
        """
        if embedder_type not in cls._embedder_classes:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

        embedder_class = cls._embedder_classes[embedder_type]

        try:
            embedder = embedder_class(**kwargs)
            logger.info(f"Succeefully created {embedder_type} embedder")
            return embedder
        except Exception as e:
            logger.error(f"Failed to create {embedder_type} embedder: {e}")
            raise

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all available model configurations.
        
        Returns:
            Dictionary of model configurations
        """
        return cls._model_configs.copy()

    @classmethod
    def get_available_types(cls) -> Dict[str, Type[BaseEmbeddingModel]]:
        """
        Get all available embedder types.

        Returns:
            Dictionary of embedder types
        """
        return cls._embedder_classes.copy()

        
## Usage
# # Example 1: Create a predefined model
# bge_en_embedder = EmbedderFactory.create("bge_base_en")
# openai_embedder = EmbedderFactory.create("openai_default")

# # Example 2: Create a custom model
# custom_bge = EmbedderFactory.create_custom("bge", model_name="BAAI/bge-m3")

# # Example 3: Register a new model configuration
# EmbedderFactory.register_model_config(
#     "bge_m3", 
#     {"type": "bge", "model_name": "BAAI/bge-m3"}
# )
# bge_m3_embedder = EmbedderFactory.create("bge_m3")

# # Example 4: Get the model from the embedder
# model = bge_en_embedder.model  # This gives you the actual embedding model

# # Before:
# self.embedders = {
#     "openai": OpenAIEmbedding().model,
#     "bge_en": BgeEmbedding(model_name="BAAI/bge-base-en-v1.5").model,
#     "bge_zh": BgeEmbedding(model_name="BAAI/bge-base-zh-v1.5").model,
# }

# # After:
# self.embedders = {
#     "openai": EmbedderFactory.create("openai_default").model,
#     "bge_en": EmbedderFactory.create("bge_base_en").model,
#     "bge_zh": EmbedderFactory.create("bge_base_zh").model,
# }