# rag/text_processor/text_processor.py
import re
from typing import Optional, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TextProcessor:
    def __init__(self):
        pass

    def split_text(self, docs, chunk_size=1000, chunk_overlap=200):
        """
        Split the docs (List[Document]) into smaller chunks suitable for embedding.
        
        :param docs: List[Document] - The list of documents to split.
        :param chunk_size: int - The size of each chunk. Default is 1000 characters.
        :param chunk_overlap: int - The overlap between chunks. Default is 200 characters overlapped.
        :return: List[Document] - The list of documents split into smaller chunks.
        """
        # Add additional separators customizing for Chinese texts
        # Ref: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
            # Existing args
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        doc_chunks = text_splitter.split_documents(docs)
        return doc_chunks

    def clean_page_content(self, docs: List[Document]):
        """
        Clean up the page content of each document in the list.
        Clean up newline characters and whitespaces to obtain compact text.
        Change happens in-place.

        :param docs: List[Document]
        """
        for document in docs:
            cleaned_content = self.clean_text(document.page_content)
            document.page_content = cleaned_content
    
    def clean_text(self, text: str) -> str:
        """
        Clean up text:
        1. handle newline characters '\n'
        2. handle whitespaces
        3. other situations

        :param text: The input text to clean.
        :return: The cleaned text with repeated newlines removed.
        """
        # Remove UTF-8 BOM if present. its presence can cause issues in text processing
        text = text.replace('\ufeff', '')

        # Replace multiple newlines with a single newline, preserving paragraph structure
        text = re.sub(r'\n{2,}', '\n\n', text)

        # Replace all sequences of whitespace characters (spaces, tabs, etc.) excluding newline with a single space
        text = re.sub(r'[^\S\n]+', ' ', text)

        # Finally, strip leading and trailing whitespace (including newlines)
        return text.strip()
    
    def prepend_source_in_content(self, docs: List[Document], source: Optional[str] = None):
        """
        Prepend the source information to the page content of each document in the list.
        If a source is provided, use that. Otherwise, use the 'source' key from each document's metadata.

        :param docs: List[Document] - The list of documents to prepend the source to.
        :param source: str (optional) - The source to prepend to the content. If not provided, use the 'source' key from the document's metadata.
        """
        
        for doc in docs:
            current_source = source or doc.metadata.get('source')
            if current_source:
                prefix = f"<source>{current_source}<\source>"
                doc.page_content = " ".join([prefix, doc.page_content])
