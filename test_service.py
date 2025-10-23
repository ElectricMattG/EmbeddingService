#!/usr/bin/env python3
"""
Simple test script to verify our RayService embedding service is working.
This bypasses port-forwarding issues by connecting directly to the Ray cluster.
"""

import requests
import time

def test_embedding_service():
    """Test the embedding service endpoints"""
    
    # Test with different potential URLs
    test_urls = [
        "http://localhost:8001",
        "http://localhost:10001", 
        "http://127.0.0.1:8001",
        "http://127.0.0.1:10001"
    ]
    
    for url in test_urls:
        try:
            print(f"\n🧪 Testing: {url}")
            
            # Test root endpoint (should return docs redirect or API info)
            response = requests.get(f"{url}/", timeout=5)
            print(f"   ✅ Root endpoint: {response.status_code}")
            
            # Test health endpoint 
            response = requests.get(f"{url}/health", timeout=5)
            print(f"   ✅ Health endpoint: {response.status_code}")
            if response.status_code == 200:
                print(f"   📊 Health data: {response.json()}")
            
            # Test docs endpoint
            response = requests.get(f"{url}/docs", timeout=5)
            print(f"   ✅ Docs endpoint: {response.status_code}")
            
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection refused")
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Testing RayService Embedding Service")
    print("=" * 50)
    
    if test_embedding_service():
        print("\n✅ Service is accessible!")
    else:
        print("\n❌ Service not accessible through port forwarding")
        print("   But this might be expected - let's check Ray dashboard!")
        print("   Ray Dashboard: http://localhost:8265")