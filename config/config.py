import os

class Config:
    """Configuration for the AI agent."""
    
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
    SLACK_API_TOKEN = os.getenv("SLACK_API_TOKEN", "dummy")
    SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "C07T3RT54MQ")

