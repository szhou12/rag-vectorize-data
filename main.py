import os
from typing import List, Dict, Optional, Tuple
from langchain.schema import Document
from rag.parsers import PDFParser, ExcelParser
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Literal, Tuple
from langchain.schema import Document
import logging

class ContentSource(ABC):
    """Abstract base class for different content sources (files, web pages)"""
    
    def __init__(self, mysql_manager, vector_stores, text_processor, transaction_context):
        self.mysql_manager = mysql_manager
        self.vector_stores = vector_stores
        self.text_processor = text_processor
        self.transaction = transaction_context  # Reference to DataAgent's transaction method
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Main processing pipeline for this content source"""
        pass
    
    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Check if content already exists in storage"""
        pass
    
    @abstractmethod
    def get_metadata(self, filters: Optional[Any] = None) -> List[Dict]:
        """Retrieve metadata for this content source"""
        pass
    
    @abstractmethod
    def delete(self, targets: Any) -> None:
        """Delete content and metadata for this content source"""
        pass
    
    @abstractmethod
    def _parse_content(self, source: str, **kwargs) -> Tuple[List[Document], List[Dict]]:
        """Parse/extract content from the source"""
        pass
    
    @abstractmethod
    def _insert_data(self, docs_metadata: List[Dict], chunks: List[Document], language: str) -> List[Dict]:
        """Insert data into storage systems (Chroma + MySQL)"""
        pass
class FileContentSource(ContentSource):
    """Handles processing of uploaded files (PDF, Excel)"""
    
    def process(self, filepath: str, file_size: float, language: Literal["en", "zh"]) -> Optional[List[Dict]]:
        """
        Process an uploaded file: parse content, embed, and save to vector store.
        
        :param filepath: Path to the file to process
        :param file_size: Size of the uploaded file in MB
        :param language: Language of the content ("en" or "zh")
        :return: List of chunk metadata if successful, None if file already exists
        """
        self.logger.info(f"Starting file processing: {filepath}, size: {file_size}, language: {language}")
        
        try:
            # Step 1: Check if file already exists
            if self.exists(filepath):
                self.logger.info(f"File {filepath} already exists in database")
                return None
            
            # Step 2: Parse the file
            self.logger.info("Parsing file content")
            docs, metadata = self._parse_content(filepath, file_size=file_size, language=language)
            self.logger.info(f"Parsed {len(docs)} documents")
            
            # Step 3: Clean content
            self.logger.info("Cleaning content")
            self.text_processor.clean_page_content(docs)
            
            # Step 4: Split into chunks
            self.logger.info("Splitting into chunks")
            chunks = self.text_processor.split_text(docs)
            self.text_processor.prepend_source_in_content(chunks, source=os.path.basename(filepath))
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # Step 5: Insert data
            self.logger.info("Inserting data into storage")
            chunk_metadata = self._insert_data(metadata, chunks, language)
            self.logger.info(f"Successfully inserted {len(chunk_metadata)} chunks")
            
            return chunk_metadata
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid file or language: {e}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Runtime error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
    
    def exists(self, filepath: str) -> bool:
        """Check if file already exists in the database"""
        with self.transaction(commit=False) as session:
            try:
                existing_file = self.mysql_manager.check_file_exists_by_source(session, filepath)
                return existing_file is not None
            except Exception as e:
                self.logger.error(f"Error checking file existence: {e}")
                return False
    
    def get_metadata(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Get file metadata with optional filtering
        
        :param filters: Dict that can contain:
                       - 'sources': List[str] - specific file sources
                       - 'level': 'file' or 'page' - aggregation level  
                       - 'sources_and_pages': List[Dict] - specific source+page combinations
        """
        if filters is None:
            filters = {}
        
        level = filters.get('level', 'file')
        sources = filters.get('sources')
        sources_and_pages = filters.get('sources_and_pages')
        
        with self.transaction(commit=False) as session:
            try:
                if level == 'page' or sources_and_pages:
                    # Get page-level metadata
                    return self.mysql_manager.get_file_pages(session, sources_and_pages)
                else:
                    # Get file-level metadata (default)
                    return self.mysql_manager.get_files(session, sources)
            except Exception as e:
                self.logger.error(f"Error getting file metadata: {e}")
                return []
    
    def delete(self, targets: Dict[str, List[Dict]]) -> None:
        """
        Delete file data grouped by language
        
        :param targets: Dict with language keys and file data as values
                       Example: {'en': [], 'zh': [{'source': 'file.pdf', 'page': '1'}, ...]}
        """
        for language, file_data in targets.items():
            if file_data:
                self._delete_content_and_metadata(file_data, language)
    
    def _parse_content(self, filepath: str, file_size: float, language: Literal["en", "zh"]) -> Tuple[List[Document], List[Dict]]:
        """Parse file content based on file extension"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist")
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Select parser based on file extension
        if file_ext == '.pdf':
            parser_class = PDFParser
        elif file_ext in ['.xls', '.xlsx']:
            parser_class = ExcelParser
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Initialize and use parser
        parser = parser_class(filepath)
        self.logger.info(f"Using parser: {parser.__class__.__name__}")
        
        try:
            docs, metadata = parser.load_and_parse()
            
            # Augment metadata with additional info
            for item in metadata:
                item["language"] = language
                item["page"] = str(item["page"])
                item["file_size"] = file_size
            
            # Special processing for Excel files
            if file_ext in ['.xls', '.xlsx']:
                for doc, meta in zip(docs, metadata):
                    if 'text_as_html' in doc.metadata:
                        doc.page_content = doc.metadata['text_as_html']
                    doc.metadata['page'] = meta['page']
                    doc.metadata['source'] = meta['source']
            
            self.logger.info(f"Parsing complete: {len(docs)} documents")
            return docs, metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing file {filepath}: {e}")
            raise RuntimeError(f"Error parsing file {filepath}: {e}")
    
    def _insert_data(self, docs_metadata: List[Dict], chunks: List[Document], language: Literal["en", "zh"]) -> List[Dict]:
        """Insert file data into Chroma and MySQL with 2PC pattern"""
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Insert embeddings into Chroma
                chunks_metadata = self.vector_stores[language].add_documents(
                    documents=chunks, 
                    secondary_key='page'
                )
                
                # Step 2: Insert metadata into MySQL
                self.mysql_manager.insert_file_pages(session, docs_metadata)
                self.mysql_manager.insert_file_page_chunks(session, chunks_metadata)
                
                return chunks_metadata
                
        except Exception as e:
            self.logger.error(f"Error inserting file data: {e}")
            
            # Rollback Chroma changes if MySQL fails
            if 'chunks_metadata' in locals():
                try:
                    chunk_ids = [item['id'] for item in chunks_metadata]
                    self.vector_stores[language].delete(ids=chunk_ids)
                except Exception as rollback_error:
                    self.logger.error(f"Failed to rollback Chroma: {rollback_error}")
            
            raise RuntimeError(f"File data insertion failed: {e}")
    
    def _delete_content_and_metadata(self, sources_and_pages: List[Dict], language: Literal["en", "zh"]) -> None:
        """Delete file content from Chroma and metadata from MySQL"""
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Get data for potential rollback
                old_chunk_ids = self.mysql_manager.get_file_page_chunk_ids(session, sources_and_pages)
                old_documents = self.vector_stores[language].get_documents_by_ids(ids=old_chunk_ids)
                
                # Step 2: Delete from MySQL and Chroma
                self.mysql_manager.delete_file_page_chunks_by_ids(session, old_chunk_ids)
                self.mysql_manager.delete_file_pages_by_sources_and_pages(session, sources_and_pages)
                self.vector_stores[language].delete(ids=old_chunk_ids)
                
                self.logger.info(f"Successfully deleted data for {len(sources_and_pages)} file pages")
                
        except Exception as e:
            self.logger.error(f"Error deleting file data: {e}")
            
            # Rollback Chroma changes
            try:
                if 'old_documents' in locals() and old_documents:
                    self.vector_stores[language].add_documents(
                        documents=old_documents, 
                        ids=old_chunk_ids, 
                        secondary_key='page'
                    )
            except Exception as rollback_error:
                self.logger.error(f"Failed to rollback Chroma: {rollback_error}")
            
            raise RuntimeError(f"File data deletion failed: {e}")
        

from typing import List, Dict, Optional, Tuple, Literal, Union
from langchain.schema import Document
from rag.scrapers import WebScraper

class WebContentSource(ContentSource):
    """Handles processing of scraped web content"""
    
    def __init__(self, mysql_manager, vector_stores, text_processor, transaction_context):
        super().__init__(mysql_manager, vector_stores, text_processor, transaction_context)
        # Web scraper utility for scraping content from URLs
        self.scraper = WebScraper(mysql_manager=mysql_manager)
    
    def process(self, url: str, max_pages: int = 1, autodownload: bool = False, 
                refresh_frequency: Optional[int] = None, language: Literal["en", "zh"] = "en") -> Tuple[int, int]:
        """
        Process a given URL: scrape content, embed, and save to vector store.
        
        :param url: Start URL to scrape content from
        :param max_pages: Maximum number of pages to scrape (BFS if > 1)
        :param autodownload: Whether to automatically download linked files
        :param refresh_frequency: Frequency in days to re-scrape content
        :param language: Language of the content ("en" or "zh")
        :return: Tuple of (total_pages_scraped, newly_downloaded_files)
        """
        self.logger.info(f"Starting web processing: {url}, max_pages: {max_pages}, language: {language}")
        
        try:
            # Step 1: Scrape content from the URL
            web_pages, newly_downloaded_files = self.scraper.scrape(url, max_pages, autodownload)
            
            # Step 2: Categorize web pages into new, expired, and up-to-date
            new_web_pages, expired_web_pages, up_to_date_web_pages = self._categorize_documents(web_pages)
            
            if not new_web_pages:
                self.logger.info("No new web pages scraped")
                return len(web_pages), len(newly_downloaded_files)
            
            # Step 3: Extract metadata for new documents
            new_web_pages_metadata = self._extract_metadata(new_web_pages, refresh_frequency, language)
            
            # Step 4: Clean and process content
            self.text_processor.clean_page_content(new_web_pages)
            
            # Step 5: Split content into chunks
            new_web_pages_chunks = self.text_processor.split_text(new_web_pages)
            self.text_processor.prepend_source_in_content(new_web_pages_chunks)
            
            # Step 6: Insert data
            chunk_metadata_list = self._insert_data(new_web_pages_metadata, new_web_pages_chunks, language)
            self.logger.info(f"Successfully inserted {len(chunk_metadata_list)} data chunks")
            
            # Reset scraped URLs in WebScraper instance
            self.scraper.fetch_active_urls_from_db()
            
            return len(web_pages), len(newly_downloaded_files)
            
        except RuntimeError as e:
            self.logger.error(f"Failed to process URL {url}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing URL {url}: {e}")
            raise RuntimeError(f"Web processing failed: {e}")
    
    def update(self, url: str) -> List[Dict]:
        """
        Update content for a single URL by re-scraping and re-embedding.
        
        :param url: The URL to update
        :return: List of new chunk metadata
        """
        self.logger.info(f"Updating content for URL: {url}")
        
        # Load the updated web page content
        update_web_page = self.scraper.load_url(url)
        
        if update_web_page is None:
            self.logger.error(f"Failed to load URL: {url}")
            raise RuntimeError(f"Failed to load URL: {url}")
        
        # Clean and process content
        self.text_processor.clean_page_content(update_web_page)
        update_web_page_chunks = self.text_processor.split_text(update_web_page)
        
        try:
            chunk_metadata_list = self._update_data(url, update_web_page_chunks)
            self.logger.info(f"Successfully updated data for {url}: {len(chunk_metadata_list)} chunks")
            
            # Reset scraped URLs in WebScraper instance
            self.scraper.fetch_active_urls_from_db()
            
            return chunk_metadata_list
            
        except RuntimeError as e:
            self.logger.error(f"Failed to update data for {url}: {e}")
            raise
    
    def update_refresh_frequency(self, metadata: List[Dict]) -> None:
        """
        Update refresh frequency for specified web pages.
        
        :param metadata: List of dicts with 'source' and 'refresh_frequency'
                        Example: [{'source': 'https://example.com', 'refresh_frequency': 7}]
        """
        try:
            with self.transaction(commit=True) as session:
                self.mysql_manager.update_web_pages_refresh_frequency(session, sources_and_freqs=metadata)
                self.logger.info(f"Successfully updated refresh frequency for {len(metadata)} web pages")
        except Exception as e:
            self.logger.error(f"Error updating refresh frequency: {e}")
            raise RuntimeError(f"Failed to update refresh frequency: {e}")
    
    def exists(self, url: str) -> bool:
        """Check if web page exists and its status"""
        # For web pages, we typically allow re-processing even if they exist
        # since they might need updating. This method could be used to check
        # if a page exists without processing it.
        with self.transaction(commit=False) as session:
            try:
                existing_page = self.mysql_manager.check_web_page_exists(session, url)
                return existing_page is not None
            except Exception as e:
                self.logger.error(f"Error checking web page existence: {e}")
                return False
    
    def get_metadata(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Get web page metadata with optional filtering.
        
        :param filters: Dict that can contain:
                       - 'sources': List[str] - specific URLs to filter
        :return: List of web page metadata
        """
        sources = filters.get('sources') if filters else None
        
        with self.transaction(commit=False) as session:
            try:
                web_pages = self.mysql_manager.get_web_pages(session, sources)
                return web_pages
            except Exception as e:
                self.logger.error(f"Error getting web metadata: {e}")
                return []
    
    def delete(self, targets: Union[List[Dict], List[str]]) -> None:
        """
        Delete web data. Supports multiple input formats:
        1. List of dicts with 'source' and 'language' keys
        2. List of source URLs (will auto-detect language)
        
        :param targets: Sources to delete in various formats
        """
        if not targets:
            return
        
        # Handle different input formats
        if isinstance(targets[0], dict):
            # Format: [{'source': 'url', 'language': 'en'}, ...]
            self._delete_by_source_and_language(targets)
        else:
            # Format: ['url1', 'url2', ...]
            self._delete_by_sources_only(targets)
    
    def _parse_content(self, url: str, **kwargs) -> Tuple[List[Document], List[Dict]]:
        """
        For web content, this is handled by the scraper in the process method.
        This method is primarily used for single URL updates.
        """
        web_page = self.scraper.load_url(url)
        if web_page is None:
            raise RuntimeError(f"Failed to scrape content from {url}")
        
        # Extract metadata
        refresh_frequency = kwargs.get('refresh_frequency')
        language = kwargs.get('language', 'en')
        metadata = self._extract_metadata(web_page, refresh_frequency, language)
        
        return web_page, metadata
    
    def _insert_data(self, docs_metadata: List[Dict], chunks: List[Document], language: Literal["en", "zh"]) -> List[Dict]:
        """Insert web data into Chroma and MySQL with 2PC pattern"""
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Insert embeddings into Chroma
                chunks_metadata = self.vector_stores[language].add_documents(documents=chunks)
                
                # Step 2: Insert metadata into MySQL
                self.mysql_manager.insert_web_pages(session, docs_metadata)
                self.mysql_manager.insert_web_page_chunks(session, chunks_metadata)
                
                return chunks_metadata
                
        except Exception as e:
            self.logger.error(f"Error inserting web data: {e}")
            
            # Rollback Chroma changes if MySQL fails
            if 'chunks_metadata' in locals():
                try:
                    chunk_ids = [item['id'] for item in chunks_metadata]
                    self.vector_stores[language].delete(ids=chunk_ids)
                except Exception as rollback_error:
                    self.logger.error(f"Failed to rollback Chroma: {rollback_error}")
            
            raise RuntimeError(f"Web data insertion failed: {e}")
    
    def _update_data(self, source: str, chunks: List[Document]) -> List[Dict]:
        """Update data for a single source URL using 2PC pattern"""
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Get existing data
                old_chunk_ids = self.mysql_manager.get_web_page_chunk_ids_by_single_source(session, source)
                language = self.mysql_manager.get_web_page_language_by_single_source(session, source)
                old_documents = self.vector_stores[language].get_documents_by_ids(ids=old_chunk_ids)
                
                # Step 2: Delete old data
                self.mysql_manager.delete_web_page_chunks_by_ids(session, old_chunk_ids)
                self.vector_stores[language].delete(ids=old_chunk_ids)
                
                # Step 3: Insert new data
                self.mysql_manager.update_web_pages_date(session, [source])
                new_chunks_metadata = self.vector_stores[language].add_documents(chunks)
                self.mysql_manager.insert_web_page_chunks(session, new_chunks_metadata)
                
                return new_chunks_metadata
                
        except Exception as e:
            self.logger.error(f"Error updating data for {source}: {e}")
            
            # Rollback Chroma changes
            try:
                if 'new_chunks_metadata' in locals():
                    new_chunk_ids = [item['id'] for item in new_chunks_metadata]
                    self.vector_stores[language].delete(new_chunk_ids)
                
                if 'old_documents' in locals() and old_documents:
                    self.vector_stores[language].add_documents(documents=old_documents, ids=old_chunk_ids)
            except Exception as rollback_error:
                self.logger.error(f"Failed to rollback Chroma: {rollback_error}")
            
            raise RuntimeError(f"Data update failed for {source}: {e}")
    
    def _categorize_documents(self, docs: List[Document]) -> Tuple[List[Document], List[Document], List[Document]]:
        """Categorize web documents into new, expired, and up-to-date"""
        new_docs, expired_docs, up_to_date_docs = [], [], []
        
        with self.transaction(commit=False) as session:
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
                self.logger.error(f"Error categorizing documents: {e}")
                raise
        
        return new_docs, expired_docs, up_to_date_docs
    
    def _extract_metadata(self, docs: List[Document], refresh_frequency: Optional[int], 
                         language: Literal["en", "zh"], extra_metadata: Optional[Dict] = None) -> List[Dict]:
        """Extract metadata from web page documents"""
        document_info_list = []
        
        for doc in docs:
            source = doc.metadata.get('source')
            if source:
                atom = {
                    'source': source,
                    'refresh_frequency': refresh_frequency,
                    'language': language
                }
                
                if extra_metadata:
                    atom.update(extra_metadata)
                
                document_info_list.append(atom)
            else:
                self.logger.warning(f"Source not found in metadata: {doc.metadata}")
        
        return document_info_list
    
    def _delete_by_source_and_language(self, metadata: List[Dict]) -> None:
        """Delete web data grouped by language from metadata"""
        # Group sources by language
        sources_by_language = self._group_sources_by_key(metadata, 'language')
        
        for language, sources in sources_by_language.items():
            if sources:
                self._delete_content_and_metadata(sources, language)
    
    def _delete_by_sources_only(self, sources: List[str]) -> None:
        """Delete web data by sources, auto-detecting language"""
        # Get language information for each source
        sources_by_language = self.mysql_manager.get_web_page_languages_by_sources(sources)
        
        if sources_by_language['en']:
            self._delete_content_and_metadata(sources_by_language['en'], "en")
        
        if sources_by_language['zh']:
            self._delete_content_and_metadata(sources_by_language['zh'], "zh")
    
    def _delete_content_and_metadata(self, sources: List[str], language: Literal["en", "zh"]) -> None:
        """Delete web content from Chroma and metadata from MySQL"""
        try:
            with self.transaction(commit=True) as session:
                # Step 1: Get data for potential rollback
                old_chunk_ids = self.mysql_manager.get_web_page_chunk_ids_by_sources(session, sources)
                old_documents = self.vector_stores[language].get_documents_by_ids(ids=old_chunk_ids)
                
                # Step 2: Delete from MySQL and Chroma
                self.mysql_manager.delete_web_page_chunks_by_ids(session, old_chunk_ids)
                self.mysql_manager.delete_web_pages_by_sources(session, sources)
                self.vector_stores[language].delete(ids=old_chunk_ids)
                
                self.logger.info(f"Successfully deleted data for {len(sources)} sources in {language}")
                
        except Exception as e:
            self.logger.error(f"Error deleting web data for sources {sources}: {e}")
            
            # Rollback Chroma changes
            try:
                if 'old_documents' in locals() and old_documents:
                    self.vector_stores[language].add_documents(documents=old_documents, ids=old_chunk_ids)
            except Exception as rollback_error:
                self.logger.error(f"Failed to rollback Chroma: {rollback_error}")
            
            raise RuntimeError(f"Web data deletion failed for sources {sources}: {e}")
    
    def _group_sources_by_key(self, data: List[Dict], key: str) -> Dict:
        """Group sources by a given key (e.g., 'language')"""
        from collections import defaultdict
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item[key]].append(item['source'])
        return dict(grouped_data)



def main():
    print("Hello from rag-vectorize-data!")


if __name__ == "__main__":
    main()
