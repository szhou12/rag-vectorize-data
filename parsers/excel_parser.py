# rag/parsers/excel_parser.py
from rag.parsers.base_parser import BaseParser
import os
import pandas as pd
from langchain_community.document_loaders import UnstructuredMarkdownLoader

class ExcelParser(BaseParser):

    def save_file(self, sheet_name, markdown_text):
        """
        Save the Excel file (per sheet) in Markdown format to the directory = self.dir
        :return: The file path where the file is saved.
        """
        # Create the directory if it does not exist yet
        if not os.path.exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            
        md_file_path = os.path.join(self.dir, f"{self.file_basename}_{sheet_name}.md")
        if not os.path.exists(md_file_path):
            print(f'Saving <{sheet_name}> sheet as md file to temp directory')
            with open(md_file_path, 'w') as f:
                f.write(markdown_text)
        
        return md_file_path


    def load_and_parse(self):
        """
        Load and parse the Excel file of multiple sheets.

        :return: Tuple[List[Document], List[Dict]] - A list of Langchain Document objects and their corresponding metadata.
        """
        docs = []
        metadata = []

        excel_data = pd.read_excel(self.filepath, sheet_name=None)  # Read all sheets
        
        # iterate over each sheet
        for sheet_name, df in excel_data.items():
            if df.empty: # Skip empty sheets
                continue

            df = self.clean_df(df)
            markdown_text = df.to_markdown(index=False)
            
            file_path = self.save_file(sheet_name, markdown_text)

            loader = UnstructuredMarkdownLoader(file_path, mode="elements")
            docs.extend(loader.load())

            metadata.append({"source": self.filepath, "page": sheet_name})

            self.delete_markdown_sheet(sheet_name)
        
        return docs, metadata
    
    def clean_df(self, df):
        """
        Clean the DataFrame before converting to markdown.
        """
        df.dropna(how='all', inplace=True)  # Drop rows where all cells are empty
        df.dropna(axis=1, how='all', inplace=True)  # Drop columns where all cells are empty

        # df.fillna('', inplace=True)  # Replace NaN cells with empty strings
        
        # Separate numeric and non-numeric columns
        non_numeric_columns = df.select_dtypes(include=['object']).columns
        
        # Replace NaN values in non-numeric (object/string) columns with empty strings
        df[non_numeric_columns] = df[non_numeric_columns].fillna('')

        return df
    
    def delete_markdown_sheet(self, sheet_name):
        """
        Delete the corresponding markdown file for the given sheet.
        
        :param sheet_name: The name of the sheet whose markdown file is to be deleted.
        :return: True if the file was deleted successfully, False if the file was not found.
        """
        md_file_path = os.path.join(self.dir, f"{self.file_basename}_{sheet_name}.md")
        
        if os.path.exists(md_file_path):
            os.remove(md_file_path)
            print(f'Deleted markdown file for sheet <{sheet_name}>')
            return True
        else:
            print(f'Markdown file for sheet <{sheet_name}> not found')
            return False