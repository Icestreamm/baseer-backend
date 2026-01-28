"""
Quick test script to verify server endpoints
Run this locally to test before deploying
"""

import requests
import json

BASE_URL = "https://baseer-backend.onrender.com"

def test_endpoint(url, method="GET", data=None):
    """Test an endpoint and print results"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, data=data, timeout=30)
        
        print(f"\n{'='*60}")
        print(f"Testing: {method} {url}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Error testing {url}: {e}")
        return False

if __name__ == "__main__":
    print("Testing Render Server Endpoints...")
    
    # Test root
    test_endpoint(f"{BASE_URL}/")
    
    # Test health
    test_endpoint(f"{BASE_URL}/health")
    
    # Test predict (will fail if endpoint doesn't exist)
    test_endpoint(f"{BASE_URL}/predict", method="POST", data="test")
    
    print("\n" + "="*60)
    print("Testing complete!")
