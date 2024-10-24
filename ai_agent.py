from handlers.query_handler import QueryHandler
from handlers.document_handler import DocumentHandler
from handlers.slack_notifier import SlackNotifier
from config.config import Config
from typing import List
import json
import openai
from openai import OpenAI


class AIAgent:
    def __init__(self, pdf_files: List[bytes], model_name: str = None, slack_token: str = None, slack_channel: str = None, openai_api_key: str = None):
        
        # Set model name and key
        self.model_name = model_name or Config.OPENAI_MODEL
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        
        # Get maximum tokens for the model
        self.max_tokens = self.get_max_tokens(self.model_name)

        # Initialize SlackNotifier and set Slack channel
        self.slack_notifier = SlackNotifier(slack_token or Config.SLACK_API_TOKEN)
        self.slack_channel = slack_channel or Config.SLACK_CHANNEL

        # Initialize DocumentHandlers for each provided PDF file
        self.document_handlers = [DocumentHandler(pdf_file, self.model_name, self.max_tokens) for pdf_file in pdf_files]

        # Initialize QueryHandler
        self.query_handler = QueryHandler(model_name=self.model_name, openai_api_key=self.openai_api_key)


    def get_max_tokens(self, model_name: str) -> int:
        """Retrieve the maximum tokens for the given model."""
        # Known max tokens for specific models
        max_tokens_dict = {
            "gpt-4": 8192,
            "gpt-4o-mini": 128000,  # Total context window
            "gpt-3.5-turbo": 4096,
        }
    
        # Check if the model is in the known dictionary
        if model_name in max_tokens_dict:
            return max_tokens_dict[model_name]
    
        # If the model is not known, raise an error
        raise ValueError(f"Max tokens information is not available for model: {model_name}")


    def process_and_notify(self, questions: List[str], update_status_func=None):
        all_text_chunks = []

        # Extract and chunk text from each document
        for i, handler in enumerate(self.document_handlers):
            if update_status_func:
                update_status_func(f"Processing document {i + 1}/{len(self.document_handlers)}...")

            text = handler.extract_text()
            text_chunks = handler.chunk_text(text)
            all_text_chunks.extend(text_chunks)

        all_responses = []
        # Generate answers for each question
        for i, question in enumerate(questions):
            if update_status_func:
                update_status_func(f"Processing question {i + 1}/{len(questions)}: '{question}'...")

            answer = self.query_handler.handle_query(all_text_chunks, question)
            all_responses.append({"question": question, "answer": answer})

        # Post answers to Slack
        output_json = {"questions and their answers": all_responses}
        self.slack_notifier.post_message(self.slack_channel, json.dumps(output_json, indent=4))
