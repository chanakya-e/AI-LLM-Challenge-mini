from handlers.query_handler import QueryHandler
from handlers.document_handler import DocumentHandler
from handlers.slack_notifier import SlackNotifier
from config.config import Config
from typing import List
import json

class AIAgent:
    def __init__(self, pdf_files: List[bytes], model_name: str = None, slack_token: str = None, slack_channel: str = None, openai_api_key: str = None):
        # Use the provided model, token, and channel if given, otherwise use default from config
        self.model_name = model_name or "gpt-4"  # Assume this is the default model to use
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        self.slack_notifier = SlackNotifier(slack_token or Config.SLACK_API_TOKEN)
        self.slack_channel = slack_channel or Config.SLACK_CHANNEL
        self.document_handlers = [DocumentHandler(pdf_file) for pdf_file in pdf_files]
        self.query_handler = QueryHandler(model_name=self.model_name, openai_api_key=self.openai_api_key)
    
    def process_and_notify(self, questions: List[str], update_status_func=None):
        all_text_chunks = []

        for i, handler in enumerate(self.document_handlers):
            # Display which document is being processed
            if update_status_func:
                update_status_func(f"Processing document {i + 1}/{len(self.document_handlers)}...")

            text = handler.extract_text()
            text_chunks = handler.chunk_text(text)
            all_text_chunks.extend(text_chunks)

        all_responses = []
        for i, question in enumerate(questions):
            # Display which question is being processed
            if update_status_func:
                update_status_func(f"Processing question {i + 1}/{len(questions)}: '{question}'...")

            answer = self.query_handler.handle_query(all_text_chunks, question)
            all_responses.append({"question": question, "answer": answer})

        # Format output as structured JSON
        output_json = {"questions and their answers": all_responses}
        self.slack_notifier.post_message(self.slack_channel, json.dumps(output_json, indent=4))
