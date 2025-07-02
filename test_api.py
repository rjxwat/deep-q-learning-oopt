import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test all API endpoints"""
    print("Testing DQN Inventory Optimization API...")
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Model info
    print("\n3. Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Single optimization
    print("\n4. Testing single optimization...")
    optimization_request = {
        "sku": "SKU_001",
        "city": "Berlin",
        "current_stock": 100,
        "base_demand": 20.0,
        "demand_std": 5.0,
        "classical_eoq": 60.0,
        "optimal_multiplier": 1.0,
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "transport_cost": 10.0,
        "unit_value": 25.0,
        "lead_time_mean": 3.0,
        "lead_time_std": 1.0,
        "stock_turnover": 0.5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/optimize",
            json=optimization_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Batch optimization
    print("\n5. Testing batch optimization...")
    batch_request = {
        "items": [
            {
                "sku": "SKU_001",
                "city": "Berlin",
                "current_stock": 100,
                "base_demand": 20.0,
                "demand_std": 5.0,
                "classical_eoq": 60.0,
                "optimal_multiplier": 1.0,
                "ordering_cost": 50.0,
                "holding_cost": 2.0,
                "transport_cost": 10.0,
                "unit_value": 25.0,
                "lead_time_mean": 3.0,
                "lead_time_std": 1.0,
                "stock_turnover": 0.5
            },
            {
                "sku": "SKU_002",
                "city": "Munich",
                "current_stock": 50,
                "base_demand": 15.0,
                "demand_std": 3.0,
                "classical_eoq": 45.0,
                "optimal_multiplier": 0.8,
                "ordering_cost": 40.0,
                "holding_cost": 1.5,
                "transport_cost": 8.0,
                "unit_value": 20.0,
                "lead_time_mean": 2.0,
                "lead_time_std": 0.5,
                "stock_turnover": 0.6
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/optimize-batch",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait a bit for the server to start
    print("Waiting for server to start...")
    time.sleep(2)
    test_api() 