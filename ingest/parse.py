import pymupdf

from pathlib import Path
import os
import re
from bisect import bisect_right
from math import ceil

from tqdm import tqdm

from ingest.embed import Embed


__PATH__ = Path(__file__).resolve()
DATA_PATH = Path(Path(__PATH__).parent.parent, "data")
SAFETY_TOKEN_GAP = 50

class Parser:
    def __init__(self, embed_model: str):
        self.data_path = DATA_PATH
        self.pdf_files_path = Path(DATA_PATH, "pdf")
        self.embedding = Embed(embed_model)
        self.chunks = {}

    def extract(self, files: list[str] | None = None):
        """
        Fill self.chunks as {doc_name: [{page_index, chunk_index, content}, ...]}
        files: PDF file names to parse; all PDFs in data/pdf when None
        """
        for file in tqdm(files or os.listdir(self.pdf_files_path), desc="Parsing", unit="pdf"):
            sections = self.build_sections(self.extract_pdf(file))
            self.chunks[Path(file).stem] = self.chunk_sections(sections)

    def build_sections(self, pages: list[str]):
        """
        Split pages into sections at each "Item N" heading
        Returns (text, page_offsets, page_indexes) per section
        """
        sections = []
        text, page_offsets, page_indexes = "", [], []
        for page_index, page in enumerate(pages):
            start_index = 0
            # Each heading closes the current section and starts a new one with that heading
            for sec in self.extract_section(page):
                page_offsets.append(len(text))
                page_indexes.append(page_index)
                text += page[start_index : sec["start"]]
                sections.append((text, page_offsets, page_indexes))
                text, page_offsets, page_indexes = "", [], []
                start_index = sec["start"]
            # The rest of the page carries over to the next page
            page_offsets.append(len(text))
            page_indexes.append(page_index)
            text += page[start_index:]
        sections.append((text, page_offsets, page_indexes))
        return sections

    def chunk_sections(self, sections):
        """
        Split each section into token-limited chunks; a chunk belongs to the page it starts on.
        """
        chunks = []
        for text, page_offsets, page_indexes in sections:
            for chunk_start, chunk in self.chunk_restructure(text):
                page_index = page_indexes[bisect_right(page_offsets, chunk_start) - 1]
                chunks.append(
                    {"page_index": page_index, "chunk_index": len(chunks), "content": chunk}
                )
        return chunks

    def extract_pdf(self, file_name: str):
        """
        Extracting pdf data
        """
        def filter_text(content: str):
            return (
                content\
                .replace("\x00", "")
            )
        file_path = Path(self.pdf_files_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exists")
        pdf = pymupdf.open(file_path)
        pages = []
        for page_index in range(pdf.page_count):
            # Some PDFs contain NUL bytes, which Postgres text columns reject
            pages.append(filter_text(pdf[page_index].get_text()))
        return pages

    def extract_section(self, page: str):
        """
        Extract each section using pattern: "Item N"
        """
        sections = []
        ITEM_PATTERN = re.compile(r"^[ \t]*item\s+(\d{1,2}[A-C]*)", re.I | re.M)
        for m in ITEM_PATTERN.finditer(page):
            sections.append({"start": m.start(), "end": m.end()})
        return sections

    def chunk_restructure(self, chunk: str):
        """
        Restructuring chunks based on max tokens.
        Yields (start offset in chunk, chunk text) pairs.
        """
        # returns the index of characters extacted as part of tokenization
        offsets = self.embedding.tokenizer(
            chunk, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        max_tokens = self.embedding.max_chunk_tokens - SAFETY_TOKEN_GAP
        total_tokens = len(offsets)
        possible_num_chunks = ceil(total_tokens / max_tokens)
        for chunk_index in range(possible_num_chunks):
            window = offsets[
                chunk_index * max_tokens : (chunk_index + 1)
                * max_tokens
            ]
            yield window[0][0], chunk[window[0][0] : window[-1][1]]
