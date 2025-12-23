"""
Client to test the FastAPI inference endpoint
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n=== HEALTH CHECK ===")
    response = requests.get(f"{BASE_URL}/health")
    print(response.json())

def test_info():
    """Test model info endpoint."""
    print("\n=== MODEL INFO ===")
    response = requests.get(f"{BASE_URL}/info")
    print(json.dumps(response.json(), indent=2))

def test_predict():
    """Test prediction endpoint."""
    print("\n=== PREDICTION TEST ===")
    
    test_cases = [
        {
            "item_id": "FOODS_3_090",
            "store_id": "CA_3",
            "date": "2016-04-25"
        },
        {
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "date": "2016-04-24"
        },
        {
            "item_id": "HOUSEHOLD_1_116",
            "store_id": "TX_1",
            "date": "2016-04-23"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nRequest: {test_case}")
        response = requests.post(f"{BASE_URL}/predict", json=test_case)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Prediction: {result['prediction']:.4f}")
            print(f"    Model: {result['model_used']} (v{result['model_version']})")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"    {response.json()}")

if __name__ == "__main__":
    try:
        test_health()
        test_info()
        test_predict()
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API. Is the server running?")
        print("Start the server with:")
        print("  python -m uvicorn inference.app:app --reload")
