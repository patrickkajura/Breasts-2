"""
Standard application entry point for Render deployment
This creates an alias for our actual application to satisfy Render's default expectations
"""
from app import app

# Create the Flask application instance
application = app

# This creates the expected 'application' object that Render/Gunicorn is looking for
# Also make it available as wsgi application
wsgi = application