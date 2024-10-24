from openai import OpenAI

class QueryHandler:
    def __init__(self, model_name: str = "gpt-4", openai_api_key: str = None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=self.openai_api_key)

    def handle_query(self, text_chunks: list[str], question: str) -> str:
        prompt = "\n".join(text_chunks) + f"\nQuestion: {question}\nAnswer:"
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "Data Not Available"
        except Exception as e:
            return f"Error: {str(e)}"
