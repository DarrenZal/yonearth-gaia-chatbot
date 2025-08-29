#!/usr/bin/env python3
"""
Production server startup script with voice support
"""
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Run the main API with full features including voice
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=80,
        reload=False,
        log_level="info"
    )