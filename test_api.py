"""
Test script for the RAG API
"""

import requests
import json
import time

def test_api():
    """Test the RAG API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing RAG API...")
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test root endpoint
        print("\\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test health endpoint
        print("\\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test stats endpoint
        print("\\n3. Testing stats endpoint...")
        response = requests.get(f"{base_url}/stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test query endpoint
        print("\\n4. Testing query endpoint...")
        query_data = {
            "question": "What is multi-head attention?",
            "num_chunks": 3,
            "min_score": 0.1
        }
        response = requests.post(f"{base_url}/query", json=query_data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
        print(f"Chunks found: {result.get('chunks_found', 0)}")
        print(f"Processing time: {result.get('processing_time_ms', 0):.1f}ms")
        
        # Test search endpoint
        print("\\n5. Testing search endpoint...")
        response = requests.post(f"{base_url}/search?query=attention&limit=2")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Chunks found: {result.get('chunks_found', 0)}")
        
        print("\\n✅ API tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_api()
