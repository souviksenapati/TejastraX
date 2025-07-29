import sys
import os
import json

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from app.main import app
from mangum import Mangum

# Create Netlify-compatible handler
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """Netlify Functions handler"""
    return handler(event, context)