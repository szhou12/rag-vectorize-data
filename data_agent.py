import logging
import os
from contextlib import contextmanager
from collections import defaultdict
from typing import Optional, Literal, List, Tuple, Generator
from langchain.schema import Document
from sqlalchemy.orm import Session
from db_mysql import MySQLManager
from rag.parsers import PDFParser, ExcelParser
from rag.scrapers import WebScraper
from rag.embedders import OpenAIEmbedding, BgeEmbedding
from rag.vector_stores import ChromaVectorStore
from rag.text_processor import TextProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAgent:
    def __init__(
            self,
            mysql_config: dict, 
            vector_db_persist_dir: Optional[str] = None, 
    ) -> None:
        """
        Initialize the DataAgent class.
        Responsibility: Handle data processing tasks such as parsing, embedding, and storing data in MySQL and Chroma.
            1. Intake data: scraping web content, parsing uploaded files
            2. Process data: clean, split, embed
            3. Store data: CRUD embedding data in Chroma, metadata in MySQL

        :param mysql_config: (dict) - Configuration settings for MySQL database connection.
        :param vector_db_persist_dir: (str | None) - Name of Chroma's persistent directory. Used to construct persistent directory. If None, storage is in-memory and emphemeral.
        :return: None
        """
        
        self.mysql_manager = MySQLManager(**mysql_config)

        ## Web scraper utility for scraping contents from URLs
        self.scraper = WebScraper(mysql_manager=self.mysql_manager)

        self.text_processor = TextProcessor()

        ## Embedding models to convert texts to embeddings (vectors)
        self.embedders = {
            "openai": OpenAIEmbedding().model,
            # "bge_en": BgeEmbedding(model_name="BAAI/bge-small-en-v1.5").model,
            # "bge_zh": BgeEmbedding(model_name="BAAI/bge-small-zh-v1.5").model,
            "bge_en": BgeEmbedding(model_name="BAAI/bge-base-en-v1.5").model,
            "bge_zh": BgeEmbedding(model_name="BAAI/bge-base-zh-v1.5").model,
            # "bge_en": BgeEmbedding(model_name="BAAI/bge-large-en-v1.5").model,
            # "bge_zh": BgeEmbedding(model_name="BAAI/bge-large-zh-v1.5").model,
        }

        self.vector_stores = {
            "en": ChromaVectorStore(
                collection_name="docs_en",  # English collection
                embedding_model=self.embedders['bge_en'],
                # embedding_model=self.embedders['openai'],
                persist_directory=vector_db_persist_dir,
            ),
            "zh": ChromaVectorStore(
                collection_name="docs_zh",  # Chinese collection
                embedding_model=self.embedders['bge_zh'],
                # embedding_model=self.embedders['openai'],
                persist_directory=vector_db_persist_dir,
            ),
        }
    
    # TODO: delete after testing
    # def chroma_storage_testing(self):
    #     """
    #     Test the connection to Chroma and storage functionality.
    #     """
    #     # en_result = self.vector_stores['en'].storage_test()
    #     zh_result = self.vector_stores['zh'].storage_test()
    #     res = f"中文数据库: {zh_result}"
    #     return res


    def close(self):
        """
        Close all resources in DataAgent.
        """
        # Close MySQL connections or sessions
        if self.mysql_manager:
            self.mysql_manager.close()

        # TODO: Close any vector stores (if applicable)
        # if self.vector_stores:
            # for store in self.vector_stores.values():
            #     store.close()  # Assuming ChromaVectorStore has a `close` method
            # self.vector_stores.clear()

        print("DataAgent resources cleaned up.")

    @contextmanager
    def transaction(self, commit: bool = True) -> Generator[Session, None, None]:
        """
        Context manager for SQLAlchemy transactions.
        It automatically commits or rolls back the transaction (skipped if read-only operation) and closes the session.
        
        The try-except block here is responsible for managing the session lifecycle, ensuring that the session is managed correctly (opening a session, committing, or rolling back). i.e., This block does not handle the business logic of any method that calls this context manager.

        General pattern of Python's generator annotation: Generator[yield_type, send_type, return_type]

        Usage:
        with self.transaction() as session:
            # Perform database operations

        :param commit: Whether to commit the transaction. Defaults to True. 
                       For read-only operations, set commit=False.
        :yield: SQLAlchemy session
        """
        session = self.mysql_manager.create_session()
        try:
            yield session # Hand control to the caller for this context
            if commit:
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Transaction failed: {e}")
            raise
        finally:
            self.mysql_manager.close_session(session)

    def process_file(self, filepath: str, file_size: float, language: Literal["en", "zh"]):
        """
        Process an uploaded file: parse file content, embed, and save to vector store.
        
        :param filepath: (str) The file path. Currently support: PDF (multiple pages), Excel (multiple sheets)
        :param file_size: (float) The size of the uploaded file in MB.
        :param language: The language of the web page content. Only "en" (English) or "zh" (Chinese) are accepted.
        :return: None
        """
        logger.info(f"Starting process_file with filepath: {filepath}, size: {file_size}, language: {language}")
        try:
            # Step 1: Check if the file already exists in the database
            logger.info(f"Step 1: Checking if file exists at {filepath}")
            if self._file_source_exists(filepath):
                # print(f"File <{filepath}> already exists in the database.")
                logger.info(f"File <{filepath}> already exists in the database.")
                return

            # Step 2: Parse the file based on the file extension
            ## metadata := [{'source': 'example.pdf', 'page': 1, 'language': 'zh', 'file_size': 2.50}, ...]
            logger.info("Step 2: Parse the file")

            docs, metadata = self._parse_file(filepath, file_size, language)

            logger.info(f"File parsed successfully. Got {len(docs)} documents")

            # Step 3: Clean content before splitting
            logger.info("Step 3: Clean content before splitting")

            self.text_processor.clean_page_content(docs)

            logger.info(f"File cleaned successfully. Got {len(docs)} documents, doc0 = {docs[0]}")

            # Step 4: Split content into manageable chunks, prepend source to each chunk
            logger.info("Step 4: Split into chunks")

            new_file_pages_chunks = self.text_processor.split_text(docs)
            self.text_processor.prepend_source_in_content(new_file_pages_chunks, source=os.path.basename(filepath))

            logger.info(f"File split successfully. Got {len(new_file_pages_chunks)} chunks, chunk0 = {new_file_pages_chunks[0]}")

            # Step 5: Embed each chunk (Document) and save to the vector store
            logger.info("Step 5: Embed and save to vector store")
            chunk_metadata_list = self.insert_file_data(docs_metadata=metadata, chunks=new_file_pages_chunks, language=language)
            # print(f"Data successfully inserted into both Chroma and MySQL: {len(chunk_metadata_list)} data chunks")
            logger.info(f"Data successfully inserted into both Chroma and MySQL: {len(chunk_metadata_list)} data chunks")

        except FileNotFoundError as e:
            print(f"File not found: {filepath}")
        except ValueError as e:
            print(f"Invalid file or language: {e}")
        except RuntimeError as e:
            print(f"Runtime error occurred: {e}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")

    def process_url(self, url: str, max_pages: int = 1, autodownload: bool = False, refresh_frequency: Optional[int] = None, language: Literal["en", "zh"] = "en"):
        """
        Process a given URL: scrape content, embed, and save to vector store.
        
        :param url: start URL to scrape content from
        :param max_pages: The maximum number of pages to scrape. If > 1, scrape sub-URLs using BFS. Default is 1.
        :param autodownload: Whether to automatically download files linked in the URL. Default is False.
        :param refresh_frequency: The frequency in days to re-scrape and update the page content.
        :param language: The language of the web page content. Only "en" (English) or "zh" (Chinese) are accepted.
        :return: None
        """
        # Step 1: Scrape content from the URL
        web_pages, newly_downloaded_files = self.scraper.scrape(url, max_pages, autodownload)

        # Step 2: Categorize the web_pages into new, expired, and up-to-date
        # TODO: handle expired_docs, up_to_date_docs later
        new_web_pages, expired_web_pages, up_to_date_web_pages = self._categorize_web_documents(web_pages)

        if not new_web_pages:
            print("No new web pages scraped")
            return 0, 0

        # Step 3: Extract metadata for the new documents
        # new_web_pages_metadata := [{'source': source, 'refresh_frequency': freq, 'language': lang}]
        new_web_pages_metadata = self.extract_metadata(new_web_pages, refresh_frequency, language)

        # Step 4: Clean content before splitting
        self.text_processor.clean_page_content(new_web_pages)
        
        # Step 5: Split content into manageable chunks
        new_web_pages_chunks = self.text_processor.split_text(new_web_pages)
        self.text_processor.prepend_source_in_content(new_web_pages_chunks)


        # Step 6: Insert data: insert content into Chroma, insert metadata into MySQL
        # chunk_metadata_list := [{'source': source, 'id': chunk_id}, ...]
        try:
            chunk_metadata_list = self.insert_web_data(docs_metadata=new_web_pages_metadata, chunks=new_web_pages_chunks, language=language)
            print(f"Data successfully inserted into both Chroma and MySQL: {len(chunk_metadata_list)} data chunks")
        except RuntimeError as e:
            print(f"Failed to insert data into Chroma and MySQL due to an error: {e}")

        # Reset self.scraped_urls in WebScraper instance
        self.scraper.fetch_active_urls_from_db()

        return len(web_pages), len(newly_downloaded_files)
    
    def update_single_url(self, url: str):
        """
        Update the content of a given URL by re-scraping and re-embedding the content.

        :param url: The URL to update content for.
        """
        update_web_page = self.scraper.load_url(url)
        
        if update_web_page is None:
            print(f"Failed to load URL: {url}")
            return

        self.text_processor.clean_page_content(update_web_page)

        update_web_page_chunks = self.text_processor.split_text(update_web_page)

        try:
            chunk_metadata_list = self.update_web_data(source=url, chunks=update_web_page_chunks)
            print(f"Data successfully updated in both Chroma and MySQL: {chunk_metadata_list}")
        except RuntimeError as e:
            print(f"Failed to update data in Chroma and MySQL due to an error: {e}")

        # Reset self.scraped_urls in WebScraper instance
        self.scraper.fetch_active_urls_from_db()

    def _parse_file(self, filepath: str, file_size: float, language: Literal["en", "zh"] = "en") -> Tuple[List[Document], List[dict]]:
        """
        Process an uploaded file: parse file content, embed, and save to vector store.
        
        :param filepath: (str) The file path. Currently supports: PDF (multiple pages), Excel (multiple sheets)
        :param file_size: (float) The size of the entire uploaded file in MB.
        :param language: The language of the web page content. Only "en" (English) or "zh" (Chinese) are accepted.
        :return: A tuple containing a list of Langchain Document objects and metadata.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        
        file_ext = os.path.splitext(filepath)[1].lower()

        # Select the appropriate parser based on file extension
        if file_ext == '.pdf':
            parser_class = PDFParser
        elif file_ext in ['.xls', '.xlsx']:
            parser_class = ExcelParser
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        # Init parser
        parser = parser_class(filepath)
        logger.info(f"Parser initialized: {parser.__class__.__name__}")

        try:
            logger.info("Starting document parsing...")
            
            docs, metadata = parser.load_and_parse() # metadata := [{'source': src, 'page': page}]

            # Further processing of docs and metadata
            # Step 1: Augment metadata with language and ensure 'page' is a string
            # each item is a dictionary
            for item in metadata:
                item["language"] = language
                item["page"] = str(item["page"])  # Convert page number to string
                item["file_size"] = file_size
            
            # Step 2: Additional processing for Excel files
            if file_ext in ['.xls', '.xlsx']:
                for doc, meta in zip(docs, metadata):
                    # Replace doc.page_content with doc.metadata['text_as_html']
                    if 'text_as_html' in doc.metadata:
                        doc.page_content = doc.metadata['text_as_html']
                    # Add doc.metadata['page'] = metadata['page']
                    doc.metadata['page'] = meta['page']
                    # Replace doc.metadata['source'] with metadata['source'] as doc.metadata['source'] is .md
                    doc.metadata['source'] = meta['source']
            
            logger.info(f"Parsing complete. Returning {len(docs)} documents with metadata")

            return docs, metadata
        
        except Exception as e:
            logger.error(f"Error during parsing: {e}")
            raise RuntimeError(f"Error parsing file {filepath}: {e}")

    def _group_sources_by_key(self, data: List[dict], key: str) -> dict:
        """
        Group a list of dictionaries by a given key.

        :param data: List of dictionaries to group. e.g., [{'source': 'example.com', 'language': 'en'}, {'source': 'example1.com', 'language': 'zh'}]
        :param key: The key to group the sources by. e.g., key='language'
        """
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item[key]].append(item['source'])
        return dict(grouped_data)
    
    def _init_embedder(self, embedder_type: str):
        """
        Initialize the embedding model based on the provided type.
        
        :param embedder_type: Type of embedding model to use ("openai" or "bge")
        :return: The model instance from the embedding model
        :raises ValueError: If the embedder type is not supported or if the API key is missing.
        """
        embedder_type = embedder_type.lower()

        if embedder_type == "openai":
            try:
                openai_embedding = OpenAIEmbedding()
                return openai_embedding.model 
            except ValueError as e:
                raise ValueError(f"Failed to initialize OpenAI Embeddings: {e}")
        elif embedder_type == "bge":
            try:
                huggingface_embedding = BgeEmbedding()
                return huggingface_embedding.model
            except Exception as e:
                raise ValueError(f"Failed to initialize Hugging Face BGE Embeddings: {e}")
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
        
    def _categorize_web_documents(self, docs: List[Document]) -> Tuple[List[Document], List[Document], List[Document]]:
        """
        Categorize documents (scraped web page) into new, expired, and up-to-date based on their status in the MySQL database.
        
        :param docs: List[Document] - Documents returned from the scraper.
        :return: Tuple[List[Document], List[Document], List[Document]] - (new_docs, expired_docs, up_to_date_docs)
        """
        new_docs, expired_docs, up_to_date_docs = [], [], []
        
        with self.transaction(commit=False) as session:
            # try-except block is responsible for this method's business logic
            try:
                for document in docs:
                    existing_page = self.mysql_manager.check_web_page_exists(session, document.metadata['source'])

                    if existing_page:
                        if existing_page.is_refresh_needed():
                            expired_docs.append(document)
                        else:
                            up_to_date_docs.append(document)
                    else:
                        new_docs.append(document)

            except Exception as e:
                # Handle any exceptions that occur in the business logic
                print(f"An error occurred while categorizing documents: {e}")
                raise  # Re-raise the exception after logging

        return new_docs, expired_docs, up_to_date_docs
    

    def extract_metadata(self, docs: List[Document], refresh_frequency: Optional[int] = None, language: Literal["en", "zh"] = "en", extra_metadata: Optional[dict] = None):
        """
        Extract metadata from the web page documents and optionally augment it with additional metadata.

        :param docs: List[Document]
        :param refresh_frequency: The re-scraping frequency in days for web contents. Keep None for uploaded files.
        :param language: The language of the web page/uploaded file content, either "en" (English) or "zh" (Chinese).
        :param extra_metadata: Optional dictionary to augment each dict with additional metadata.
        :return: List[dict] - [{'source': src, 'refresh_frequency': refresh_freq, 'language': lang}]
        """
        document_info_list = []
        
        for doc in docs:
            source = doc.metadata.get('source', None)
            if source:
                # Step 1: Create the base atom dictionary
                atom = {
                    'source': source, 
                    'refresh_frequency': refresh_frequency, 
                    'language': language
                }

                # Step 2: Merge additional metadata from the dictionary (if provided) into atom
                if extra_metadata:
                    atom.update(extra_metadata)
                
                document_info_list.append(atom)
            else:
                print(f"Source not found in metadata: {doc.metadata}")

        return document_info_list
    
    def insert_web_data(self, docs_metadata: List[dict], chunks: List[Document], language: Literal["en", "zh"]) -> List[dict]:
        """
        Wrapper function to handle atomic insertion of scraped web content into Chroma (for embeddings) and MySQL (for metadata).
        Implements the manual two-phase commit (2PC) pattern.
        
        :param docs_metadata: List[dict] - Metadata of documents to be inserted into MySQL.
        :param chunks: List[Document] - Chunks of document text to be inserted into Chroma.
        :param language: The language of the inserted data content. Only "en" (English) or "zh" (Chinese) are accepted.
        :raises: Exception if any part of the insertion process fails.
        :return: List[dict] chunks_metadata - Metadata of chunks inserted into Chroma.
        """
        # The outer try-except focuses solely on handling the Chroma rollback and logging errors
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Insert embeddings into Chroma (vector store)
                chunks_metadata = self.vector_stores[language].add_documents(documents=chunks)

                # Step 2: Insert metadata into MySQL
                self.mysql_manager.insert_web_pages(session, docs_metadata)
                self.mysql_manager.insert_web_page_chunks(session, chunks_metadata)

                # Step 3: Commit is handled automatically by the context manager on success

                # If both steps succeed, return the chunk metadata
                return chunks_metadata

        except Exception as e:
            print(f"Error during data insertion into Chroma and MySQL: {e}")

            # Rollback Chroma changes if MySQL fails
            if 'chunks_metadata' in locals():
                try:
                    chunk_ids = [item['id'] for item in chunks_metadata]
                    self.vector_stores[language].delete(ids=chunk_ids)  # Delete embeddings by ids in Chroma
                except Exception as chroma_rollback_error:
                    print(f"Failed to rollback Chroma insertions: {chroma_rollback_error}")

            # Re-raise the exception to notify the caller
            raise RuntimeError(f"Data insertion failed: {e}")


    def update_web_data(self, source: str, chunks: List[Document]) -> List[dict]:
        """
        Update data for a SINGLE source URL and its chunks.
        Implements atomic behavior using manual two-phase commit (2PC) pattern.
        
        :param source: Single URL of the web page being updated.
        :param chunks: List[Document] - New chunks of document text to be inserted into Chroma.
        :raises: RuntimeError if any part of the update process fails.
        :return: List[dict] new_chunks_metadata - Metadata of new chunks inserted into Chroma.
        """
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Get
                # 1-1: MySQL: Get old chunk ids by source
                old_chunk_ids = self.mysql_manager.get_web_page_chunk_ids_by_single_source(session, source)
                # 1-2: MySQL: Get language by source
                language = self.mysql_manager.get_web_page_language_by_single_source(session, source)
                # 1-3: Chroma: Get old documents from Chroma before deletion (for potential rollback)
                old_documents = self.vector_stores[language].get_documents_by_ids(ids=old_chunk_ids)

                # Step 2: Delete
                # 2-1: MySQL: Delete WebPageChunk by old ids
                self.mysql_manager.delete_web_page_chunks_by_ids(session, old_chunk_ids)
                # 2-2: Chroma: Delete old chunks by old ids
                self.vector_stores[language].delete(ids=old_chunk_ids)

                # Step 3: Upsert
                # 3-1: MySQL: Update the 'date' field for WebPage
                self.mysql_manager.update_web_pages_date(session, [source])
                # 3-2: Chroma: Insert new chunks into Chroma, get new chunk ids
                new_chunks_metadata = self.vector_stores[language].add_documents(chunks)
                # 3-3: MySQL: Insert new WebPageChunk into MySQL
                self.mysql_manager.insert_web_page_chunks(session, new_chunks_metadata)

                # If all steps succeed, return the new chunk metadata
                return new_chunks_metadata

        except Exception as e:
            print(f"Error updating data for source {source}: {e}")
            
            # Rollback Chroma changes if MySQL fails
            try:
                # If new chunks were already inserted into Chroma, delete them to maintain consistency
                if 'new_chunks_metadata' in locals():
                    new_chunk_ids = [item['id'] for item in new_chunks_metadata]
                    self.vector_stores[language].delete(new_chunk_ids)

                # Restore old chunks to Chroma if they were deleted
                if 'old_documents' in locals() and old_documents:
                    self.vector_stores[language].add_documents(documents=old_documents, ids=old_chunk_ids)
            except Exception as chroma_rollback_error:
                print(f"Failed to rollback Chroma insertions: {chroma_rollback_error}")
            
            # Re-raise the exception to notify the caller
            raise RuntimeError(f"Data update failed for source {source}: {e}")
        
    def update_web_data_refresh_frequency(self, metadata: List[dict]):
        """
        Update the refresh frequency for specified scraped web pages.

        :param metadata: List of dictionaries received from Front-End, each containing 'source' and 'refresh_frequency'.
                                    Example: [{'source': 'https://rmi.org', 'refresh_frequency': 7},
                                              {'source': 'https://iea.org', 'refresh_frequency': 30}]
        :return: None
        """
        try:
            with self.transaction(commit=True) as session:
                self.mysql_manager.update_web_pages_refresh_frequency(session, sources_and_freqs=metadata)
                print(f"Successfully updated refresh frequency for web pages: {metadata}")
        except Exception as e:
            print(f"Error updating refresh frequency for web pages: {e.__class__.__name__}: {e}")



    def get_web_page_metadata(self, sources: Optional[List[str]] = None) -> List[dict]:
        """
        Get WebPage content (metadata) for web pages by their sources if provided; otherwise, return all.
        
        :param sources: Optional list of sources (e.g. URLs) of the web pages to be fetched. If None, return all.
        :return: List[dict] - Metadata of the web pages stored in WebPage table. Example: [{'id': 1, 'source': 'https://example.com', 'date': '2024-10-08', 'language': 'en', 'refresh_frequency': 30}, ...]
        """
        # read-only transaction (no commit required) outside try-except block
        with self.transaction(commit=False) as session:
            try:
                web_pages = self.mysql_manager.get_web_pages(session, sources)
                return web_pages
            except Exception as e:
                print(f"Error getting web metadata: {e}")
                return []

    def delete_web_data(self, metadata: List[dict]):
        """
        Delete data for web pages by grouping sources based on their language.

        :param metadata: List of dictionaries received from Front-End, each containing 'source' and 'language'.
                                    Example: [{'source': 'example.com', 'language': 'en'}, 
                                              {'source': 'example1.com', 'language': 'zh'}]
        :return: None
        """
        # Step 1: Transform input to {'en': [source1, source2], 'zh': [source3, source4]}
        sources_by_language = self._group_sources_by_key(data=metadata, key='language')

        # Step 2: Delete data for each language group using existing logic
        for language, sources in sources_by_language.items():
            if sources:  # Proceed only if there are sources to delete
                self.delete_web_content_and_metadata(sources=sources, language=language)


    def delete_web_data_by_sources(self, sources: List[str]):
        """
        Delete data for multiple sources by their language.
        
        :param sources: List of sources (e.g. URLs) of the web pages to be deleted.
        :return: None
        """
        # Get and categorize sources by language: {'en': [source1, source2], 'zh': [source3, source4]}
        sources_by_language = self.mysql_manager.get_web_page_languages_by_sources(sources)

        # Process deletion for English sources
        if sources_by_language['en']:
            self.delete_web_content_and_metadata(sources_by_language['en'], language="en")

        # Process deletion for Chinese sources
        if sources_by_language['zh']:
            self.delete_web_content_and_metadata(sources_by_language['zh'], language="zh")


    def delete_web_content_and_metadata(self, sources: List[str], language: Literal["en", "zh"]) -> None:
        """
        Delete content data from Chroma and metadata from MySQL for a list of web sources.
        Implements atomic behavior using manual two-phase commit (2PC) pattern.
        
        :param sources: List of sources (e.g. URLs) of the web pages to be deleted.
        :param language: The language of the web page content. Only "en" (English) or "zh" (Chinese) are accepted.
        :return: None
        :raises: RuntimeError if any part of the deletion process fails.
        """
        try:
            # 领域展开
            with self.transaction(commit=True) as session:
                # Step 1: Get chunk IDs and documents
                # 1-1: MySQL: Get all chunk ids for the given sources
                old_chunk_ids = self.mysql_manager.get_web_page_chunk_ids_by_sources(session, sources)
                # 1-2: Chroma: Get old documents from Chroma before deletion (for potential rollback)
                old_documents = self.vector_stores[language].get_documents_by_ids(ids=old_chunk_ids)

                # Step 2: Delete from MySQL and Chroma
                # 2-1: Delete WebPageChunk from MySQL by old chunk IDs
                self.mysql_manager.delete_web_page_chunks_by_ids(session, old_chunk_ids)
                # 2-2: Delete WebPages from MySQL by sources
                self.mysql_manager.delete_web_pages_by_sources(session, sources)
                # 2-3: Delete chunks from Chroma by old chunk IDs
                self.vector_stores[language].delete(ids=old_chunk_ids)

                # If everything succeeds, commit is handled automatically by the context manager
                print(f"Successfully deleted data for sources in {language}: {sources}")

        except Exception as e:
            print(f"Error deleting data for sources {sources}: {e}")
            
            # Rollback Chroma changes if MySQL fails
            try:
                if 'old_documents' in locals() and old_documents:
                    self.vector_stores[language].add_documents(documents=old_documents, ids=old_chunk_ids)

            except Exception as chroma_rollback_error:
                print(f"Failed to rollback Chroma insertions: {chroma_rollback_error}")
            
            # Raise the error to notify the caller
            raise RuntimeError(f"Data deletion failed for sources {sources}: {e}")

    def _file_source_exists(self, filepath: str) -> bool:
        """
        Check if the file already exists in the FilePage database based on the source filepath.

        :param filepath: The file path to check.
        :return: True if the file exists in the database, otherwise False.
        """
        # Use the context manager for read-only transaction (no commit required)
        with self.transaction(commit=False) as session:
            try:
                # Check if the file exists in the database
                existing_file = self.mysql_manager.check_file_exists_by_source(session, filepath)

                # Return True if the file exists, False otherwise
                return existing_file is not None
            except Exception as e:
                print(f"Error checking if file exists in the database: {e}")
                return False
    

    def insert_file_data(self, docs_metadata: List[dict], chunks: List[Document], language: Literal["en", "zh"]) -> List[dict]:
        """
        Wrapper function to handle atomic insertion of uploaded file content into Chroma (for embeddings) and MySQL (for metadata).
        Implements the manual two-phase commit (2PC) pattern.
        
        :param docs_metadata: List[dict] - Metadata of documents to be inserted into MySQL. [{"source": filename, "page": page num/sheet name, "language": en/zh, "file_size": 7.1}, {...}]
        :param chunks: List[Document] - Chunks of document text to be inserted into Chroma.
        :param language: The language of the inserted data content. Only "en" (English) or "zh" (Chinese) are accepted.
        :raises: Exception if any part of the insertion process fails.
        :return: List[dict] chunks_metadata - Metadata of chunks inserted into Chroma.
        """
        try:
            # Use the context manager for transactional database operations
            with self.transaction(commit=True) as session:
                # Step 1: Insert embeddings into Chroma (vector store)
                chunks_metadata = self.vector_stores[language].add_documents(documents=chunks, secondary_key='page')

                # Step 2: Insert metadata into MySQL
                self.mysql_manager.insert_file_pages(session, docs_metadata)
                self.mysql_manager.insert_file_page_chunks(session, chunks_metadata)

                # If both steps succeed, return the chunk metadata
                return chunks_metadata

        except Exception as e:
            print(f"Error during data insertion into Chroma and MySQL: {e}")

            # Rollback Chroma changes if MySQL fails
            try:
                if 'chunks_metadata' in locals():
                    chunk_ids = [item['id'] for item in chunks_metadata]
                    self.vector_stores[language].delete(ids=chunk_ids)  # Delete embeddings by ids in Chroma
            except Exception as chroma_rollback_error:
                print(f"Failed to rollback Chroma insertions: {chroma_rollback_error}")
            
            # Re-raise the exception to notify the caller
            raise RuntimeError(f"Data insertion failed: {e}")
    

    def get_file_metadata(self, sources: Optional[List[str]] = None) -> List[dict]:
        """
        Get FilePage content (metadata) for uploaded files by their sources if provided; otherwise, return all on source level.
        
        :param sources: Optional list of sources of the uploaded file pages to be fetched. [{'source': str}]. If None, return all.
        :return: List[dict] - Metadata of the uploaded file pages stored in FilePage table. Example: [{'source': 'example.pdf', 'date': '2024-10-08', 'language': 'en', 'file_size': 7.10, 'total_records': 3}, {...}]
        """
        # Use the context manager for read-only transaction (no commit required)
        with self.transaction(commit=False) as session:
            try:
                file_metadata = self.mysql_manager.get_files(session, sources)
                return file_metadata
            except Exception as e:
                print(f"Error getting file metadata: {e}")
                return []


    def get_file_page_metadata(self, sources_and_pages: Optional[List[dict]] = None) -> List[dict]:
        """
        Get FilePage content (metadata) for uploaded files by their sources and pages if provided; otherwise, return all on (source, page) level.
        
        :param sources_and_pages: Optional list of sources and pages of the uploaded file pages to be fetched. [{'source': str, 'page': str}]. If None, return all.
        :return: List[dict] - Metadata of the uploaded file pages stored in FilePage table. Example: [{'id': 1, 'source': 'path/to/file.pdf', 'page': '1', 'date': '2024-10-08', 'language': 'en'}, ...]
        """
        # Use the context manager for read-only transaction (no commit required)
        with self.transaction(commit=False) as session:
            try:
                # Get metadata for uploaded files by sources and pages if provided; otherwise, return all
                file_metadata = self.mysql_manager.get_file_pages(session, sources_and_pages)
                return file_metadata
            except Exception as e:
                print(f"Error getting file page metadata: {e}")
                return []
    
    def delete_file_data(self, files_by_language: dict):
        """
        Delete file data for multiple sources grouped by language.
        
        :param files_by_language: Dictionary with language keys ('en', 'zh') and list of file data dictionaries as values.
                              Example: {'en': [], 'zh': [{'id': 3, 'source': 'path/to/clean_energy.xlsx', 'page': 'intro', ...}]}
        :return: None
        """
        # Process deletion for English sources
        if files_by_language['en']:
            self.delete_file_content_and_metadata(files_by_language['en'], language="en")

        # Process deletion for Chinese sources
        if files_by_language['zh']:
            self.delete_file_content_and_metadata(files_by_language['zh'], language="zh")
    
    def delete_file_content_and_metadata(self, sources_and_pages: List[dict[str, str]], language: Literal["en", "zh"]) -> None:
        """
        Delete content data from Chroma and metadata from MySQL for a list of uploaded files.
        Implements atomic behavior using manual two-phase commit (2PC) pattern.
        
        :param sources_and_pages: List of sources and pages of the uploaded file pages to be deleted. [{'source': str, 'page': str}]
        :param language: The language of the web page content. Only "en" (English) or "zh" (Chinese) are accepted.
        :return: None
        :raises: RuntimeError if any part of the deletion process fails.
        """
        try:
            # Use the context manager for transactional database operations
            with self.transaction(commit=True) as session:
                # Step 1: Get chunk IDs and documents
                old_chunk_ids = self.mysql_manager.get_file_page_chunk_ids(session, sources_and_pages)
                old_documents = self.vector_stores[language].get_documents_by_ids(ids=old_chunk_ids)

                # Step 2: Delete from MySQL and Chroma
                # 2-1: Delete FilePageChunk from MySQL by old chunk IDs
                self.mysql_manager.delete_file_page_chunks_by_ids(session, old_chunk_ids)
                # 2-2: Delete FilePage from MySQL by sources and pages
                self.mysql_manager.delete_file_pages_by_sources_and_pages(session, sources_and_pages)
                # 2-3: Delete chunks from Chroma by old chunk IDs
                self.vector_stores[language].delete(ids=old_chunk_ids)

                # Commit handled by context manager if everything succeeds
                print(f"Successfully deleted data for sources: {sources_and_pages}")

        except Exception as e:
            print(f"Error deleting data for sources {sources_and_pages}: {e}")

            # Rollback Chroma changes if MySQL fails
            try:
                if 'old_documents' in locals() and old_documents:
                    self.vector_stores[language].add_documents(documents=old_documents, ids=old_chunk_ids, secondary_key='page')
            except Exception as chroma_rollback_error:
                print(f"Failed to rollback Chroma insertions: {chroma_rollback_error}")

            # Re-raise the exception to notify the caller
            raise RuntimeError(f"Data deletion failed for sources {sources_and_pages}: {e}")

    