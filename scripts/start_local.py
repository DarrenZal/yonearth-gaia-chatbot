#!/usr/bin/env python3
"""
Local development startup script for YonEarth Gaia Chatbot
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required dependencies are installed"""
    try:
        import uvicorn
        import fastapi
        import openai
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists with required keys"""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("❌ .env file not found")
        print("Please copy .env.example to .env and add your API keys")
        return False
    
    # Check for required keys
    required_keys = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    with open(env_path) as f:
        content = f.read()
        
    missing_keys = []
    for key in required_keys:
        if f"{key}=" not in content or f"{key}=your_" in content:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing API keys in .env: {', '.join(missing_keys)}")
        return False
    
    print("✅ Environment configuration found")
    return True

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    try:
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
        print("✅ API server started on http://localhost:8000")
        print("📖 API docs available at http://localhost:8000/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def open_frontend():
    """Open the frontend in browser"""
    frontend_path = Path(__file__).parent.parent / "web" / "index.html"
    if frontend_path.exists():
        file_url = f"file://{frontend_path.absolute()}"
        print(f"🌐 Opening frontend: {file_url}")
        webbrowser.open(file_url)
    else:
        print("❌ Frontend files not found")

def main():
    """Main startup function"""
    print("🌍 YonEarth Gaia Chatbot - Local Development Setup")
    print("=" * 50)
    
    # Pre-flight checks
    if not check_requirements():
        return 1
        
    if not check_env_file():
        return 1
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        return 1
    
    # Give server time to start
    import time
    time.sleep(3)
    
    # Open frontend
    open_frontend()
    
    print("\n🎉 Setup complete!")
    print("💬 Chat interface should open in your browser")
    print("🔧 API server running at http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Wait for user to stop
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        api_process.terminate()
        api_process.wait()
        print("✅ Server stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
