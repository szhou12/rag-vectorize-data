from rag.embedders.base_embedder import BaseEmbeddingModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class BgeEmbedding(BaseEmbeddingModel):
    def __init__(self, model_name: str):
        """
        Initialize the BgeEmbedding model with a specified model_name.

        :param model_name: The name of the Hugging Face model to be used for Chines/English embedding.
        """
        super().__init__()

        self.logger.info(f"Initializing BGE Embedding Model: {model_name} ...")

        model_kwargs = {"device": "cpu"} # TODO: may need to change after deploy to cloud
        encode_kwargs = {"normalize_embeddings": True} # set True to compute cosine similarity

        try:
            self.model = HuggingFaceBgeEmbeddings(
                model_name=model_name, 
                model_kwargs=model_kwargs, 
                encode_kwargs=encode_kwargs
            )
            self.logger.info(f"Successfully initialized BGE Embedding Model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize BGE Embedding Model: {model_name} due to {e}")
            raise RuntimeError(f"Error initializing BgeEmbedding: {e}")
    