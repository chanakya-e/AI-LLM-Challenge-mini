from typing import List
import pdfplumber
from io import BytesIO
import tiktoken

class DocumentHandler:
    """Handles extraction of text from PDF documents."""
    
    def __init__(self, pdf_file: bytes, model_name: str, max_tokens: int = 12800, overlap_tokens: int = 500):
        """Initializes the DocumentHandler with a file-like object."""
        self.pdf_file = pdf_file
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.encoding_for_model(self.model_name)

    def extract_text(self) -> str:
        """Extracts all text from the PDF."""
        with pdfplumber.open(BytesIO(self.pdf_file)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into chunks with overlapping tokens."""
        tokens = self.encoding.encode(text)  # Convert text into tokens
        chunks = []

        # Initialize start index for token chunks
        start_index = 0
        while start_index < len(tokens):
            # Determine the end index for the current chunk
            end_index = min(start_index + self.max_tokens, len(tokens))

            # Create the chunk of tokens
            chunk_tokens = tokens[start_index:end_index]
            chunk_text = self.encoding.decode(chunk_tokens)  # Decode tokens back to text
            
            # Add the chunk to the list
            chunks.append(chunk_text.strip())
            
            # Move the start index forward for the next chunk, retaining overlap
            start_index = end_index - self.overlap_tokens

        return chunks
