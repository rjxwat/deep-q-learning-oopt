from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import pandas as pd
import uvicorn
from dqn2 import DQNAgent, InventoryEnvironment

# Initialize FastAPI app
app = FastAPI(
    title="DQN Inventory Optimization API",
    description="API for inventory optimization using Deep Q-Learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and environment
agent = None
env = None
model_loaded = False

# Pydantic models for request/response
class InventoryState(BaseModel):
    stock_level: float
    demand_mean: float
    demand_cv: float
    seasonality_factor: float
    period_progress: float
    stock_turnover: float
    days_since_last_order: float

class OptimizationRequest(BaseModel):
    sku: str
    city: str
    current_stock: int
    base_demand: float
    demand_std: float
    classical_eoq: float
    optimal_multiplier: float
    ordering_cost: float
    holding_cost: float
    transport_cost: float
    unit_value: float
    lead_time_mean: float
    lead_time_std: float
    stock_turnover: float

class OptimizationResponse(BaseModel):
    recommended_action: int
    action_multiplier: float
    order_quantity: int
    confidence_score: float
    q_values: List[float]
    state_features: Dict[str, float]
    message: str

class BatchOptimizationRequest(BaseModel):
    items: List[OptimizationRequest]

class BatchOptimizationResponse(BaseModel):
    results: List[OptimizationResponse]
    summary: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    action_multipliers: List[float]
    state_dimension: int
    action_dimension: int
    training_episodes: int
    final_epsilon: float
    final_service_level: float

def load_model():
    """Load the trained DQN model"""
    global agent, env, model_loaded
    
    try:
        print("Loading trained DQN model...")
        
        # Create minimal environment for model loading
        sample_inventory = {
            'SKU': ['SKU_1'] * 5,
            'City': ['Berlin'] * 5,
            'Stock_Begin': [100] * 5,
            'Stock_End': [80] * 5,
            'Demand_Mean_7P': [20] * 5,
            'Demand_Std_7P': [5] * 5,
            'Demand_CV': [0.25] * 5,
            'Seasonality_Factor': [1.0] * 5,
            'Ordering_Cost': [50] * 5,
            'Holding_Cost_Per_Unit': [2] * 5,
            'Transportation_Cost': [10] * 5,
            'Unit_Value': [25] * 5,
            'Lead_Time_Mean': [3] * 5,
            'Lead_Time_Std': [1] * 5,
            'Classical_EOQ': [60] * 5,
            'Optimal_Multiplier': [1.0] * 5,
            'Stock_Turnover': [0.5] * 5
        }
        inventory_df = pd.DataFrame(sample_inventory)
        sales_df = pd.DataFrame({
            'SKU': ['SKU_1'] * 10,
            'City': ['Berlin'] * 10,
            'Units_Sold': np.random.normal(20, 5, 10).astype(int)
        })
        
        env = InventoryEnvironment(sales_df, inventory_df, max_periods=5)
        agent = DQNAgent(state_dim=env.n_states, action_dim=env.n_actions, learning_rate=0.001)
        
        # Load the trained model
        agent.load_model("dqn_model_weights.pth")
        model_loaded = True
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model_loaded = False
        return False

def create_state_from_request(request: OptimizationRequest) -> np.ndarray:
    """Create state vector from optimization request"""
    # Normalization parameters (should match training)
    norm_params = {
        'stock_max': 1298,
        'demand_mean_max': 50,
        'stock_turnover_max': 2
    }
    
    state = np.zeros(7)  # 7 state features
    
    # Normalize features (matching training normalization)
    state[0] = min(request.current_stock / norm_params['stock_max'], 2.0)  # stock_level_norm
    state[1] = request.base_demand / norm_params['demand_mean_max']  # demand_mean_norm
    state[2] = min(request.demand_std / max(request.base_demand, 1), 2.0) / 2.0  # demand_cv
    state[3] = 0.0  # seasonality_factor (default to 0 for inference)
    state[4] = 0.5  # period_progress (default to middle of period)
    state[5] = min(request.stock_turnover / norm_params['stock_turnover_max'], 2.0)  # stock_turnover
    state[6] = 0.5  # days_since_last_order (default to middle)
    
    return state

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "message": "DQN Inventory Optimization API",
        "status": "running",
        "model_loaded": model_loaded
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if not model_loaded or agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_loaded=model_loaded,
        action_multipliers=env.action_multipliers.tolist() if env is not None else [],
        state_dimension=agent.state_dim if agent is not None else 0,
        action_dimension=agent.action_dim if agent is not None else 0,
        training_episodes=len(agent.episode_rewards) if agent is not None else 0,
        final_epsilon=agent.epsilon if agent is not None else 0.0,
        final_service_level=float(np.mean(agent.service_levels[-50:])) if agent is not None and agent.service_levels else 0.0)

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_inventory(request: OptimizationRequest):
    """Get inventory optimization recommendation for a single item"""
    if not model_loaded or agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create state from request
        state = create_state_from_request(request)
        
        # Get Q-values for all actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor).cpu().numpy().flatten()
        
        # Select best action
        recommended_action = int(np.argmax(q_values))
        action_multiplier = env.action_multipliers[recommended_action] if env is not None else 1.0
        order_quantity = int(request.classical_eoq * action_multiplier)
        
        # Calculate confidence score (normalized Q-value difference)
        max_q = np.max(q_values)
        min_q = np.min(q_values)
        confidence_score = (max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-8)
        
        # Create state features dictionary
        state_features = {
            "stock_level_norm": float(state[0]),
            "demand_mean_norm": float(state[1]),
            "demand_cv": float(state[2]),
            "seasonality_factor": float(state[3]),
            "period_progress": float(state[4]),
            "stock_turnover": float(state[5]),
            "days_since_last_order": float(state[6])
        }
        
        return OptimizationResponse(
            recommended_action=recommended_action,
            action_multiplier=float(action_multiplier),
            order_quantity=order_quantity,
            confidence_score=float(confidence_score),
            q_values=q_values.tolist(),
            state_features=state_features,
            message=f"Recommended action {recommended_action} with multiplier {action_multiplier:.2f}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/optimize-batch", response_model=BatchOptimizationResponse)
async def optimize_inventory_batch(request: BatchOptimizationRequest):
    """Get inventory optimization recommendations for multiple items"""
    if not model_loaded or agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        total_items = len(request.items)
        
        for i, item_request in enumerate(request.items):
            # Create state from request
            state = create_state_from_request(item_request)
            
            # Get Q-values for all actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor).cpu().numpy().flatten()
            
            # Select best action
            recommended_action = int(np.argmax(q_values))
            action_multiplier = env.action_multipliers[recommended_action] if env is not None else 1.0
            order_quantity = int(item_request.classical_eoq * action_multiplier)
            
            # Calculate confidence score
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            confidence_score = (max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-8)
            
            # Create state features dictionary
            state_features = {
                "stock_level_norm": float(state[0]),
                "demand_mean_norm": float(state[1]),
                "demand_cv": float(state[2]),
                "seasonality_factor": float(state[3]),
                "period_progress": float(state[4]),
                "stock_turnover": float(state[5]),
                "days_since_last_order": float(state[6])
            }
            
            result = OptimizationResponse(
                recommended_action=recommended_action,
                action_multiplier=float(action_multiplier),
                order_quantity=order_quantity,
                confidence_score=float(confidence_score),
                q_values=q_values.tolist(),
                state_features=state_features,
                message=f"Item {i+1}: Action {recommended_action} with multiplier {action_multiplier:.2f}"
            )
            results.append(result)
        
        # Create summary
        avg_confidence = np.mean([r.confidence_score for r in results])
        action_distribution = {}
        for r in results:
            action = r.recommended_action
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        summary = {
            "total_items": total_items,
            "average_confidence": float(avg_confidence),
            "action_distribution": action_distribution,
            "processing_time": "batch_processed"
        }
        
        return BatchOptimizationResponse(results=results, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch optimization failed: {str(e)}")

@app.get("/reload-model", response_model=Dict[str, Any])
async def reload_model():
    """Reload the trained model"""
    success = load_model()
    return {
        "success": success,
        "message": "Model reloaded successfully" if success else "Failed to reload model",
        "model_loaded": model_loaded
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 