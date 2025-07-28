# rag/vector_stores/base_vector_store.py

from abc import ABC, abstractmethod

class VectorStore(ABC):
    def __init__(self, embedding_model):
        """
        Initialize the VectorStore with an embedding model.
        
        :param embedding_model: An instance of an embedding model (e.g., OpenAIEmbedding, HuggingFaceBgeEmbedding)
        """
        self.embedding_model = embedding_model
        
    @abstractmethod
    def add_documents(self, documents):
        """
        Add texts and their metadata to the vector store.
        
        :param texts: List of text strings to add
        :param metadatas: List of metadata dictionaries corresponding to each text
        """
        pass

    # @abstractmethod
    # def similarity_search(self, query, k=4):
    #     """
    #     Perform a similarity search for a given query.
        
    #     :param query: The query text
    #     :param k: Number of results to return
    #     :return: List of documents most similar to the query
    #     """
    #     pass

    # @abstractmethod
    # def delete(self, ids):
    #     """
    #     Delete documents from the vector store by their ids.
        
    #     :param ids: List of ids to delete
    #     """
    #     pass

    # @abstractmethod
    # def persist(self):
    #     """
    #     Persist the vector store to disk.
    #     """
    #     pass