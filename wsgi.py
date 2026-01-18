"""
WSGI entry point for the breast cancer classification application
"""
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the application directly from app.py
from app import app as application

# Explicitly make this available for gunicorn as 'application'
application = application

if __name__ == "__main__":
    application.run()