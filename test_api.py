#!/usr/bin/env python3
"""
Test script for the bot detection API.
Run this to verify the API is working correctly.
"""
import requests
import json
import time

def test_api():
    """Test the bot detection API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Bot Detection API")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   📊 Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
            print(f"   📊 Response: {response.json()}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Root endpoint failed: {e}")
    
    # Test 3: Bot analysis endpoint
    print("\n3. Testing bot analysis endpoint...")
    test_request = {
        "user_id": "test_user",
        "comments": [
            "This is a great post!",
            "I completely agree with your point.",
            "Thanks for sharing this information."
        ],
        "options": {
            "fast_only": True,  # Use fast mode for testing
            "include_breakdown": True,
            "use_context": False,
            "force_full_analysis": False
        }
    }
    
    try:
        print("   📤 Sending test request...")
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/v1/analyze-user",
            json=test_request,
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Bot analysis successful!")
            print(f"   📊 Bot Score: {result.get('bot_score', 'N/A')}%")
            print(f"   📊 Confidence: {result.get('confidence', 'N/A')}%")
            print(f"   📊 Is Bot: {result.get('is_likely_bot', 'N/A')}")
            print(f"   ⏱️  Processing Time: {result.get('processing_time_ms', 'N/A')}ms")
            print(f"   ⏱️  Total Time: {(end_time - start_time)*1000:.1f}ms")
        else:
            print(f"   ❌ Bot analysis failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Bot analysis failed: {e}")
        return False
    
    # Test 4: Models info endpoint
    print("\n4. Testing models info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/models/info", timeout=10)
        if response.status_code == 200:
            print("   ✅ Models info retrieved")
            models_info = response.json()
            if "error" in models_info:
                print(f"   ⚠️  Models not initialized: {models_info['error']}")
            else:
                print("   📊 Models are initialized and ready")
        else:
            print(f"   ❌ Models info failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Models info failed: {e}")
    
    print("\n" + "=" * 40)
    print("🎉 API testing completed!")
    print("📱 Your browser extension can now connect to:")
    print(f"   • http://localhost:8000")
    print(f"   • http://0.0.0.0:8000")
    return True

if __name__ == "__main__":
    test_api()
