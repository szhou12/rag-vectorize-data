# project/src/rag/vector_stores/chroma.py
from uuid import uuid4
from typing import Optional, List, Dict
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from chromadb import HttpClient
from .base_vector_store import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(
            self,
            collection_name: str,
            embedding_model: str,
            host: str = "chroma_container",
            port: int = 8000,
            ssl: bool = False,
            headers: Optional[Dict[str, str]] = None,
            persist_directory: Optional[str] = None # Directory inside the container
    ):
        """
        Initialize the ChromaVectorStore class with HttpClient.

        :param collection_name: Name of the collection.
        :param embedding_model: The embedding model (e.g., OpenAI, BGE).
        :param chroma_host: Hostname or IP address where the Chroma server is running.
        :param chroma_port: Port where Chroma server is listening.
        :param ssl: Boolean to indicate if SSL is used for the connection.
        :param headers: Optional HTTP headers (metadata for HTTP requests) to pass to the Chroma server.
        """
        super().__init__(embedding_model)

        self._persist_directory = persist_directory
        self.collection_name = collection_name

        # TODO: Re-configure the directory after deploy to cloud
        # Set the hardcoded base directory
        # base_dir = "/Users/shuyuzhou/Documents/github/rag-clean-energy"
        # if persist_directory is not None:
        #     # Join the base directory with the persist_directory to create the full path
        #     full_path = os.path.join(base_dir, persist_directory)
        #     # Create the directory if it doesn't exist
        #     if not os.path.exists(full_path):
        #         os.makedirs(full_path, exist_ok=True)
        #     self._persist_directory = full_path

        # Initialize Chroma HttpClient
        self.http_client = HttpClient(
            host=host,
            port=port,
            ssl=ssl,
            headers=headers
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_model,
            persist_directory=self._persist_directory,
            client=self.http_client,
        )

    # TODO: delete after testing
    # def storage_test(self):
    #     collection = self.http_client.get_collection(name=self.collection_name)
    #     result = collection.get()
    #     return result

    def as_retriever(self, **kwargs):
        """
        Wrapper of as_retriever() method of Chroma class.

        :search_type: similarity algorithm - "similarity" (default), "mmr", or "similarity_score_threshold".

        :search_kwargs:
            - k: Amount of documents to return (Default: 4)
            - fetch_k: Amount of documents to pass to "mmr" algorithm (Default: 20)
            - lambda_mult: Diversity of results returned by "mmr"; 1 for minimum diversity and 0 for maximum. (Default: 0.5)
            - score_threshold: Minimum relevance threshold for "similarity_score_threshold"
            - filter: Filter by document metadata
        """
        return self.vector_store.as_retriever(**kwargs)

    def add_documents(self, documents: List[Document], ids: Optional[list[str]] = None, secondary_key: Optional[str] = None):
        """
        Add documents to the vector store.
        Note: if the input documents contains ids and also give .add_documents() ids in the kwargs (ids=uuids), then ids=uuids will take precedence.
        Think of each given uuid as unique identifier of one record stored in Database.

        :param documents: List[Document] - List of Document objects (chunks) to add to the vector store.
        :param ids: List[str] (optional) - Predefined UUIDs for the documents. If None or length mismatch, new UUIDs will be generated.
        :param secondary_key: str (optional) - Secondary key to be extracted from the document metadata. In the case of uploaded file pages, secondary key is 'page'.
        :return: List[dict] [{'id': uuid4, 'source': source}]
        :raises: RuntimeError if the embedding insertion fails or document's source is not found.
        """
        try:
            if ids is not None and len(ids) != len(documents):
                raise ValueError("The length of 'ids' must match the number of 'documents'.")
            # Fallback to generating UUIDs if not provided
            uuids = ids if ids is not None else [str(uuid4()) for _ in range(len(documents))]

            # Generate UUIDs and extract sources
            document_info_list = []
            
            for doc, uuid in zip(documents, uuids):
                source = doc.metadata.get('source', None)  # file parser and scraper class will ensure 'source' NOT None
                if not source:
                    raise ValueError(f"Missing 'source' (None or empty str) in document metadata for document {doc.metadata}")
                
                atom = {'id': uuid, 'source': source}

                # Augment chunk metadata with secondary key if provided
                if secondary_key is not None:
                    secondary_value = doc.metadata.get(secondary_key, None)
                    if secondary_value is None or secondary_value == "":
                        raise ValueError(f"Missing '{secondary_key}' (None or empty str) in document metadata for document {doc.metadata}")
                    atom[secondary_key] = str(secondary_value)

                document_info_list.append(atom)

            # Attempt to add documents to the vector store
            self.vector_store.add_documents(documents=filter_complex_metadata(documents), ids=uuids)

            print(f"Added {len(documents)} document chunks to Chroma in collection {self.collection_name}")

            return document_info_list

        except Exception as e:
            # Catch any errors and raise them as RuntimeError with context information
            raise RuntimeError(f"Failed to add documents to Chroma: {e}")

    
    def delete(self, ids: list[str]):
        """
        Delete documents by assigned ids from the vector store.

        :param ids: list[str] List of uuid4 to identify the documents to be deleted.
        :raises: Exception if deletion fails.
        """
        try:
            self.vector_store.delete(ids=ids)
        except Exception as e:
            raise RuntimeError(f"Error while deleting from Chroma: {e}")
        
    def get_documents_by_ids(self, ids: list[str]):
        """
        Retrieve documents from the vector store by their unique IDs using Chroma's `get`.

        :param ids: List of document IDs to retrieve.
        :return: List of Document objects corresponding to the provided IDs.
        :raises: RuntimeError if document retrieval fails.
        """
        try:
            # documents = self.vector_store.get_by_ids(ids)
            documents = self.vector_store.get(ids)['documents']
            return documents
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve documents by IDs from Chroma: {e}")

    # TODO
    def similarity_search(self, query, k=4):
        # return self.vector_store.similarity_search(query, k=k)
        pass

    