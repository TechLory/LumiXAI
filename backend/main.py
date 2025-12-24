"""
FastAPI Backend Server Module

This module initializes the FastAPI application and configures
the necessary middleware for cross-origin resource sharing (CORS).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI application instance.
app = FastAPI()

# Configure CORS middleware to enable communication between the frontend
# and the backend server running on different ports.
app.add_middleware(
    CORSMiddleware,
    # Restrict allowed origins to the frontend development server.
    allow_origins=["http://localhost:3000"],
    # Enable credentials support for authenticated requests.
    allow_credentials=True,
    # Permit all HTTP methods (GET, POST, PUT, DELETE, etc.).
    allow_methods=["*"],
    # Accept all request headers from the client.
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """
    Root endpoint handler.

    Returns:
        dict: A JSON response containing the server status and framework name.
    """
    return {"status": "Backend Python is active!", "framework": "FastAPI"}