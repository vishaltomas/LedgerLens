import pymupdf

from pathlib import Path
import os
import re 



__PATH__ = Path(__file__).resolve()
DATA_PATH = Path(Path(__PATH__).parent.parent, "data")

class Parser:
    def __init__(self):
        self.data_path = DATA_PATH
        self.pdf_files_path = Path(DATA_PATH, "pdf")

    def extract(self):
        pdf_files = os.listdir(self.pdf_files_path)
        file_chunks = {}
        for file in pdf_files:
            start_index = 0
            chunks = []
            for page, index in self.extract_pdf(file):
                sections = self.extract_section(page)
                for sec in sections:
                    chunks.append((start_index, sec["start"], index))
                    start_index = sec["end"] + 1
            file_chunks[file] = chunks
        
        

    
    def extract_pdf(self, file_name:str):
        """
        Extracting pdf data
        """
        file_path = Path(self.pdf_files_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exists")
        pdf = pymupdf.open(file_path)
        for page_index in range(pdf.page_count):
            yield pdf[page_index].get_text(), page_index

    def extract_section(self, page:str):
        """
        Extract each section using pattern: "Item N"
        """
        sections = []
        ITEM_PATTERN = re.compile(r"^[ \t]*item\s+(\d{1,2}[A-C]*)", re.I | re.M)
        for m in ITEM_PATTERN.finditer(page):
            sections.append({
                "title": page[m.start(), m.end()],
                "start": m.start(),
                "end": m.end()
            })
        return sections