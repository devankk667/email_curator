#!/usr/bin/env python3
"""
Startup script for Mail Curator.ai web application.
This script handles environment setup and launches the FastAPI server.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements.txt exists and prompt to install dependencies."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if not requirements_path.exists():
        print("Error: requirements.txt not found!")
        sys.exit(1)
    
    # Check if virtual environment is active
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("Warning: No virtual environment detected.")
        print("It's recommended to use a virtual environment for this project.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Please create a virtual environment first:")
            print("  python -m venv venv")
            print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            sys.exit(1)

def check_env_file():
    """Check if .env file exists, create from .env.example if not."""
    env_path = Path(__file__).parent / ".env"
    env_example = Path(__file__).parent / ".env.example"
    
    if not env_path.exists():
        if env_example.exists():
            print("Creating .env file from .env.example...")
            import shutil
            shutil.copy(env_example, env_path)
            print("✓ Created .env file")
            print("⚠ Please edit .env file with your configuration before running the app.")
            print("  Especially set GROQ_API_KEY if you want advanced RAG features.")
            response = input("Continue with default settings? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        else:
            print("Warning: Neither .env nor .env.example found.")
            print("The app will use default configuration.")

def download_spacy_model():
    """Download spaCy model if not already installed."""
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("✓ spaCy model 'en_core_web_sm' already installed")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("✓ spaCy model downloaded successfully")
    except ImportError:
        print("Warning: spaCy not installed. Run: pip install -r requirements.txt")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "generated_emails",
        "gmail_emails",
        "templates"
    ]
    
    for dir_name in directories:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
    
    print("✓ Required directories ready")

def main():
    """Main entry point."""
    print("=" * 60)
    print("Mail Curator.ai - Startup")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run checks
    check_requirements()
    check_env_file()
    download_spacy_model()
    ensure_directories()
    
    print("\n" + "=" * 60)
    print("Starting FastAPI server...")
    print("=" * 60)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Environment variables loaded from .env")
    except ImportError:
        print("⚠ python-dotenv not installed, using system environment variables")
    
    # Get configuration from environment
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "true").lower() == "true"
    
    print(f"\nServer will run at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "dashboard_server:app",
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
