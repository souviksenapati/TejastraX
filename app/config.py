import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-2.0-flash"

EMBED_MODEL = "Gemini"
EMBED_DIM = 768  # As returned by Gemini
