# rag/parsers/pdf_parser.py
import os
from rag.parsers.base_parser import BaseParser
from langchain_community.document_loaders import PyMuPDFLoader

class PDFParser(BaseParser):
    
    def save_file(self):
        """
        Ensure the PDF file exists in the specified directory = self.dir.
        Since we're only dealing with existing PDFs, no new file writing is required.
        :return: The file path where the file is saved.
        """
        # Create the directory if it does not exist yet
        if not os.path.exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)

        # In this case, assume the file is already at self.filepath and just return the path
        if os.path.exists(self.filepath):
            print(f'File already exists at {self.filepath}')
        else:
            raise FileNotFoundError(f"The file {self.filepath} does not exist to save.")
        
        return self.filepath


    def load_and_parse(self):
        """
        Load and parse the PDF file from a file path.
        Note: 1 document = 1 page. e.g. if a file has 36 pages, then return a list of 36 documents

        :return: Tuple[List[Document], List[Dict]] - A list of Langchain Document objects and their corresponding metadata.
        """
        loader = PyMuPDFLoader(self.filepath)
        docs = loader.load()

        metadata = [{"source": self.filepath, "page": doc.metadata.get('page', None)} for doc in docs]

        return docs, metadata