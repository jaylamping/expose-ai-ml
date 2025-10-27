#!/usr/bin/env python3
"""
Test the API after it's running to identify the 500 error.
"""
import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def wait_for_server(max_attempts=30, delay=2):
    """Wait for the server to be ready."""
    base_url = "http://localhost:8000"
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            logger.info(f"⏳ Waiting for server... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
        except Exception as e:
            logger.warning(f"Unexpected error checking server: {e}")
            time.sleep(delay)
    
    logger.error("❌ Server did not start within the expected time")
    return False

def test_api():
    """Test the API with a simple request."""
    base_url = "http://localhost:8000"
    
    # Test data
    test_request = {
        "user_id": "test_user",
        "comments": [
            {
                "comment_id": "1",
                "comment": "This is a test comment to see if the API works.",
                "created_at": "2024-01-01T00:00:00Z",
                "parent_comment": None,
                "child_comment": None
            },
            {
                "comment_id": "2", 
                "comment": "Another test comment for debugging purposes.",
                "created_at": "2024-01-01T00:01:00Z",
                "parent_comment": None,
                "child_comment": None
            }
        ],
        "options": {
            "fast_only": True,
            "include_breakdown": True,
            "use_context": False,
            "force_full_analysis": False
        }
    }
    
    try:
        # Test health endpoint first
        logger.info("Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health", timeout=10)
        logger.info(f"Health check status: {health_response.status_code}")
        logger.info(f"Health check response: {health_response.json()}")
        
        # Test the analyze endpoint
        logger.info("Testing analyze endpoint...")
        logger.info(f"Sending request: {json.dumps(test_request, indent=2)}")
        
        response = requests.post(
            f"{base_url}/api/v1/analyze/user",
            json=test_request,
            timeout=60  # Longer timeout for model loading
        )
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            logger.info("✅ API request successful!")
            logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            logger.error(f"❌ API request failed with status {response.status_code}")
            logger.error(f"Response text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to API. Make sure the server is running on localhost:8000")
    except requests.exceptions.Timeout:
        logger.error("❌ Request timed out")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

def main():
    """Main test function."""
    logger.info("🧪 Starting API debug test...")
    
    # Wait for server to be ready
    if not wait_for_server():
        return
    
    # Test the API
    test_api()

if __name__ == "__main__":
    main()
