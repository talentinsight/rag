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
    
    # Change to project root directory
    project_root = os.path.dirname(__file__)
    os.chdir(project_root)
    
    # Get virtual environment python and uvicorn
    venv_python = os.path.join("rag_env", "bin", "python")
    venv_uvicorn = os.path.join("rag_env", "bin", "uvicorn")
    
    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    try:
        # Start the server
        print(f"📍 Working directory: {os.getcwd()}")
        print(f"🐍 Using Python: {venv_python}")
        print("🌐 Server will be available at: http://localhost:8000")
        print("📚 API documentation at: http://localhost:8000/docs")
        print("\\n🔄 Starting server (Ctrl+C to stop)...")
        
        # Run the API server with uvicorn
        subprocess.run([
            venv_uvicorn, 
            "src.api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], check=True, env=env)
        
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    start_server()
