#!/usr/bin/env python3
"""
Run the API server in debug mode with verbose logging.
"""
import subprocess
import sys
import time
import os

def main():
    """Run the API server in debug mode."""
    print("🚀 Starting Bot Detection API - Debug Mode")
    print("=" * 50)
    
    # Set environment variables for debug mode
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    # Run the development server
    try:
        print("Starting server with debug logging...")
        print("Logs will be written to 'bot_detection.log'")
        print("Server will be available at http://localhost:8000")
        print("API docs at http://localhost:8000/docs")
        print("=" * 50)
        
        # Run the server
        subprocess.run([
            sys.executable, "run_dev.py"
        ], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
