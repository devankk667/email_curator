#!/bin/bash
# Startup script for Render and other cloud platforms

# Install dependencies if needed
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p generated_emails gmail_emails templates

# Run the application
python run.py
