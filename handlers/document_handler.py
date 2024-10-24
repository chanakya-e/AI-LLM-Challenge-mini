from typing import List
import pdfplumber
from io import BytesIO

class DocumentHandler:
    """Handles extraction of text from PDF documents."""
    
    def __init__(self, pdf_file: bytes, max_tokens: int = 512, overlap: int = 50):
        """Initializes the DocumentHandler with a file-like object."""
        self.pdf_file = pdf_file
        self.max_tokens = max_tokens
        self.overlap = overlap  # Overlap between chunks to maintain context

    def extract_text(self) -> str:
        """Extracts all text from the PDF."""
        with pdfplumber.open(BytesIO(self.pdf_file)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into chunks based on the maximum token length."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.overlap:] if self.overlap < len(current_chunk) else current_chunk
                current_length = sum(len(w) for w in current_chunk) + len(current_chunk) - 1

            current_chunk.append(word)
            current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
