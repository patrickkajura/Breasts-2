#!/usr/bin/env python3
"""
Start script for the breast cancer classification application
Used as an alternative entry point for deployment platforms
"""
import os
import sys
from app import create_app

# Create the Flask app
try:
    from app import BreastCancerClassifier
    # Initialize the classifier globally
    classifier = BreastCancerClassifier()
    app = create_app()
except Exception as e:
    print(f"Error initializing application: {e}")
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def error_page():
        return f"Application failed to start: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)