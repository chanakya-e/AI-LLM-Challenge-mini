from openai import OpenAI

client = OpenAI()
from typing import List

class QueryHandler:
    """Handles interaction with the OpenAI GPT model for querying text."""

    def __init__(self, model_name: str = "gpt-4-0-mini", confidence_threshold: float = 0.3):
        """Initialize the query handler with an OpenAI model."""
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

    def query_model(self, text_chunks: List[str], question: str) -> str:
        """Queries the OpenAI model with a given question and text chunks.
        Returns the responses sorted by relevance. If none meet expectations, return 'Data Not Available'.
        """
        prompt = "\n".join(text_chunks) + f"\nQuestion: {question}\nAnswer:"
        try:
            response = client.chat.completions.create(model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150)
            answer = response.choices[0].message.content.strip()
            return [answer] if answer else ["Data Not Available"]
        except Exception as e:
            return [f"Error: {str(e)}"]

