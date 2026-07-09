"""
Vercel serverless entry point for Mail Curator.ai.
This file wraps the FastAPI app for deployment on Vercel.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the FastAPI app
from dashboard_server import app

# Vercel requires the app to be exposed as 'app' or 'handler'
app = app

# For Vercel's Python runtime
handler = app
