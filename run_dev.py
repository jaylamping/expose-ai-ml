#!/usr/bin/env python3
"""
Development server for bot detection API.
Optimized for local development and testing.
"""
from api.expose import ExposeAPI

def main():
    """Run the development server."""
    print("🚀 Starting Bot Detection API - Development Mode")
    print("=" * 50)
    
    # Development settings
    host = "0.0.0.0"  # Allow external connections (for browser extension)
    port = 8000
    debug = True
    
    print("📍 Server will be available at:")
    print(f"   • Local: http://localhost:{port}")
    print(f"   • Network: http://0.0.0.0:{port}")
    print(f"   • API Docs: http://localhost:{port}/docs")
    print(f"   • Health Check: http://localhost:{port}/health")
    print()
    print("🔧 Development mode enabled (debug=True)")
    print("📱 Ready for browser extension testing!")
    print("=" * 50)
    
    # Initialize and run API
    api = ExposeAPI()
    api.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
