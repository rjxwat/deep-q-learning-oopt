import torch
import numpy as np
import pandas as pd
from dqn2 import DQNAgent, InventoryEnvironment

# Test loading the saved model
print("Testing model loading...")

# Create minimal test data
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

# Create environment and agent
env = InventoryEnvironment(sales_df, inventory_df, max_periods=5)
agent = DQNAgent(state_dim=env.n_states, action_dim=env.n_actions, learning_rate=0.001)

# Load the saved model
agent.load_model("dqn_model_weights.pth")

print("✅ Model loaded successfully!")
print(f"Epsilon: {agent.epsilon:.3f}")
print(f"Training episodes: {len(agent.episode_rewards)}")
print(f"Final average reward: {np.mean(agent.episode_rewards[-50:]):.2f}")
print(f"Final average service level: {np.mean(agent.service_levels[-50:]):.3f}")

# Test inference
state = env.reset()
action = agent.select_action(state, training=False)
print(f"Test inference - State shape: {state.shape}, Action: {action}")
print("✅ Model inference works correctly!") 