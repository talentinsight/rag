#!/usr/bin/env python3
"""
Startup script for the RAG API server
"""

import os
import sys
import subprocess
import time

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting RAG API Server...")
    
    # Change to src directory
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    os.chdir(src_dir)
    
    # Get virtual environment python
    venv_python = os.path.join("..", "rag_env", "bin", "python")
    
    try:
        # Start the server
        print(f"📍 Working directory: {os.getcwd()}")
        print(f"🐍 Using Python: {venv_python}")
        print("🌐 Server will be available at: http://localhost:8000")
        print("📚 API documentation at: http://localhost:8000/docs")
        print("\\n🔄 Starting server (Ctrl+C to stop)...")
        
        # Run the API server
        subprocess.run([venv_python, "api.py"], check=True)
        
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    start_server()
