import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network for EOQ optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 32]):
        super(DQNNetwork, self).__init__() 
        
        # Smaller network to prevent overfitting
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Reduced dropout
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):  # Smaller buffer
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class InventoryEnvironment:
    """CORRECTED EOQ Inventory Management Environment"""
    
    def __init__(self, sales_df: pd.DataFrame, inventory_df: pd.DataFrame, max_periods: int = 26):
        self.sales_df = sales_df
        self.inventory_df = inventory_df
        self.max_periods = max_periods  # Shorter episodes for faster learning
        
        # CORRECTED: Create reasonable action space
        # Instead of using all multipliers, create discrete reasonable options
        self.action_multipliers = np.array([0.5, 0.75, 1.0,1.25, 1.5])
        self.n_actions = len(self.action_multipliers)
        
        # Simplified state space
        self.state_features = [
            'stock_level_norm', 'demand_mean_norm', 'demand_cv', 'seasonality_factor',
            'period_progress', 'stock_turnover', 'days_since_last_order'
        ]
        self.n_states = len(self.state_features)
        
        # Calculate normalization parameters
        self._calculate_normalization_params()
        
        # Get unique combinations
        self.sku_city_combinations = list(
            inventory_df[['SKU', 'City']].drop_duplicates().itertuples(index=False, name=None)
        )
        
        # Add debugging
        self.debug_info = {
            'rewards_breakdown': [],
            'actions_taken': [],
            'stock_levels': [],
            'demands': []
        }
        
        self.reset()
    
    def _calculate_normalization_params(self):
        """Calculate normalization parameters from actual inventory data"""
        self.norm_params = {
            'stock_max': max(self.inventory_df['Stock_End'].max(), 100),
            'demand_mean_max': max(self.inventory_df['Demand_Mean_7P'].max(), 50),
            'stock_turnover_max': max(self.inventory_df['Stock_Turnover'].max(), 2),
        }
        print(f"Normalization params: {self.norm_params}")
    
    def reset(self, sku: str = "", city: str = "") -> np.ndarray:
        """Reset environment for new episode"""
        if sku == "" or city == "":
            self.current_sku, self.current_city = random.choice(self.sku_city_combinations)
        else:
            self.current_sku, self.current_city = sku, city
        
        self.period = 0
        self.days_since_last_order = 0
        
        # Get reference data for this SKU-City
        self.reference_data = self.inventory_df[
            (self.inventory_df['SKU'] == self.current_sku) & 
            (self.inventory_df['City'] == self.current_city)
        ]
        
        if len(self.reference_data) == 0:
            # Fallback to any data for this SKU
            self.reference_data = self.inventory_df[
                self.inventory_df['SKU'] == self.current_sku
            ]
        
        if len(self.reference_data) == 0:
            raise ValueError(f"No data found for SKU {self.current_sku}")
        
        # Initialize with realistic values
        initial_row = self.reference_data.iloc[0]
        self.current_stock = int(initial_row['Stock_Begin'])
        self.base_demand = initial_row['Demand_Mean_7P']
        self.demand_std = initial_row['Demand_Std_7P']
        self.classical_eoq = initial_row['Classical_EOQ']
        self.optimal_multiplier = initial_row['Optimal_Multiplier']
        
        # Initialize tracking
        self.pending_orders = []
        self.demand_history = [self.base_demand] * 7  # Initialize with base demand
        self.total_cost = 0
        self.total_revenue = 0
        
        # Clear debug info for new episode
        self.debug_info = {
            'rewards_breakdown': [],
            'actions_taken': [],
            'stock_levels': [],
            'demands': []
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        if len(self.reference_data) == 0:
            return np.zeros(self.n_states)
        
        # Get current data or use reference
        current_row = self.reference_data.iloc[self.period % len(self.reference_data)]
        
        state = np.zeros(self.n_states)
        
        # Normalize features
        state[0] = min(self.current_stock / self.norm_params['stock_max'], 2.0)  # stock_level_norm
        state[1] = current_row['Demand_Mean_7P'] / self.norm_params['demand_mean_max']  # demand_mean_norm
        state[2] = min(current_row['Demand_CV'], 2.0) / 2.0  # demand_cv
        state[3] = (current_row['Seasonality_Factor'] - 0.5) / 1.5  # seasonality_factor
        state[4] = self.period / self.max_periods  # period_progress
        
        # Stock turnover
        recent_demand = np.mean(self.demand_history[-7:])
        if self.current_stock > 0:
            turnover = recent_demand / self.current_stock
        else:
            turnover = 0
        state[5] = min(turnover / self.norm_params['stock_turnover_max'], 2.0)  # stock_turnover
        
        state[6] = min(self.days_since_last_order / 10.0, 1.0)  # days_since_last_order
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        # Get current period data
        current_row = self.reference_data.iloc[self.period % len(self.reference_data)]
        
        # Generate demand
        demand = self._generate_demand(current_row)
        
        # Process arriving orders
        arriving_stock = self._process_arriving_orders()
        self.current_stock += arriving_stock
        
        # Fulfill demand
        sold = min(demand, self.current_stock)
        self.current_stock -= sold
        
        # Calculate order quantity based on action
        order_quantity = self._calculate_order_quantity(action, current_row)
        order_placed = False
        
        # CORRECTED: Don't order every period - use reorder point logic
        reorder_point = self.base_demand * 2  # Simple reorder point
        if self.current_stock <= reorder_point and order_quantity > 0:
            self._place_order(order_quantity, current_row)
            order_placed = True
            self.days_since_last_order = 0
        else:
            self.days_since_last_order += 1
        
        # CORRECTED: Better reward calculation
        reward = self._calculate_reward(sold, demand, order_quantity, order_placed, current_row, action)
        
        # Update tracking
        self.demand_history.append(demand)
        if len(self.demand_history) > 14:
            self.demand_history = self.demand_history[-14:]
        
        # Update debug info
        self.debug_info['demands'].append(demand)
        self.debug_info['stock_levels'].append(self.current_stock)
        self.debug_info['actions_taken'].append(action)
        
        # Move to next period
        self.period += 1
        done = self.period >= self.max_periods
        
        # Calculate service level for this episode
        total_demand = sum(self.debug_info['demands'])
        total_sold = sum(min(d, s) for d, s in zip(self.debug_info['demands'], self.debug_info['stock_levels']))
        service_level = total_sold / max(total_demand, 1)
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'demand': demand,
            'sold': sold,
            'stock_level': self.current_stock,
            'order_quantity': order_quantity if order_placed else 0,
            'service_level': service_level,
            'stockout': sold < demand,
            'period': self.period,
            'action_multiplier': self.action_multipliers[action],
            'order_placed': order_placed,
            'optimal_multiplier': self.optimal_multiplier
        }
        
        return next_state, reward, done, info
    
    def _generate_demand(self, current_row) -> int:
        """Generate realistic demand"""
        base_demand = current_row['Demand_Mean_7P']
        demand_std = current_row['Demand_Std_7P']
        seasonality = current_row['Seasonality_Factor']
        
        # Add some randomness but keep it reasonable
        demand = np.random.normal(base_demand * seasonality, demand_std * 0.5)
        return max(1, int(demand))  # Ensure minimum demand of 1
    
    def _calculate_order_quantity(self, action: int, current_row) -> int:
        """Calculate order quantity based on action"""
        classical_eoq = current_row['Classical_EOQ']
        multiplier = self.action_multipliers[action]
        return int(classical_eoq * multiplier)
    
    def _place_order(self, quantity: int, current_row):
        """Place order with lead time"""
        lead_time = max(1, int(np.random.normal(
            current_row['Lead_Time_Mean'], 
            current_row['Lead_Time_Std']
        )))
        
        arrival_period = self.period + lead_time
        self.pending_orders.append((arrival_period, quantity))
    
    def _process_arriving_orders(self) -> int:
        """Process orders arriving this period"""
        arriving_stock = 0
        remaining_orders = []
        
        for arrival_period, quantity in self.pending_orders:
            if arrival_period <= self.period:
                arriving_stock += quantity
            else:
                remaining_orders.append((arrival_period, quantity))
        
        self.pending_orders = remaining_orders
        return arriving_stock
    
    def _calculate_reward(self, sold: int, demand: int, order_quantity: int, 
                         order_placed: bool, current_row, action: int) -> float:
        """CORRECTED: Better reward function"""
        
        # Get costs from data
        ordering_cost = current_row['Ordering_Cost']
        holding_cost = current_row['Holding_Cost_Per_Unit']
        transport_cost = current_row['Transportation_Cost']
        unit_value = current_row['Unit_Value']
        
        reward_components = {}
        
        # 1. Service level reward (main objective)
        service_level = sold / max(demand, 1)
        if service_level >= 0.95:
            reward_components['service'] = 50  # Good service
        elif service_level >= 0.90:
            reward_components['service'] = 20
        else:
            reward_components['service'] = -30 * (0.95 - service_level)  # Penalty for poor service
        
        # 2. Inventory efficiency (penalize excessive stock)
        optimal_stock = self.base_demand * 3  # 3 weeks of stock
        if self.current_stock > optimal_stock * 2:
            reward_components['excess_stock'] = -20  # Too much stock
        elif self.current_stock < self.base_demand:
            reward_components['low_stock'] = -10  # Too little stock
        else:
            reward_components['stock_level'] = 5  # Good stock level
        
        # 3. Ordering cost (only when order is placed)
        if order_placed:
            total_order_cost = ordering_cost + transport_cost
            reward_components['ordering'] = -total_order_cost / 100  # Scale down
        else:
            reward_components['ordering'] = 0
        
        # 4. Holding cost
        reward_components['holding'] = -self.current_stock * holding_cost / 100
        
        # 5. Action quality (compare to optimal multiplier)
        chosen_multiplier = self.action_multipliers[action]
        optimal_multiplier = current_row['Optimal_Multiplier']
        multiplier_diff = abs(chosen_multiplier - optimal_multiplier)
        
        if multiplier_diff <= 0.25:
            reward_components['action_quality'] = 10  # Good choice
        else:
            reward_components['action_quality'] = -multiplier_diff * 5  # Poor choice
        
        # 6. Prevent over-ordering
        if order_placed and order_quantity > self.base_demand * 4:
            reward_components['over_order'] = -15
        
        total_reward = sum(reward_components.values())
        
        # Store for debugging
        self.debug_info['rewards_breakdown'].append({
            'total': total_reward,
            'components': reward_components.copy(),
            'service_level': service_level,
            'chosen_multiplier': chosen_multiplier,
            'optimal_multiplier': optimal_multiplier
        })
        
        return total_reward

class DQNAgent:
    """CORRECTED DQN Agent with better hyperparameters"""
    
        
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001, hidden_dims: List[int] = [64, 32]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # CORRECTED: Better hyperparameters
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998
        self.gamma = 0.95
        self.batch_size = 32
        self.target_update_freq = 100
        self.learn_start = 200
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        self.service_levels = []
        
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def learn(self):
        """Learn from batch of experiences"""
        if len(self.replay_buffer) < self.learn_start:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
    
    def save_model(self, filepath: str = "dqn_model_weights.pth"):
        """Save the trained model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'service_levels': self.service_levels,
            'losses': self.losses,
            'model_config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dims': [64, 32]  # Default hidden dimensions
            }
        }, filepath)
        print(f"Model weights saved to {filepath}")
    
    def load_model(self, filepath: str = "dqn_model_weights.pth"):
        """Load the trained model weights"""
        import torch
        try:
            # Try with weights_only=False for backward compatibility
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception as e:
            # If that fails, try with weights_only=True and safe globals
            from numpy.core import multiarray
            import torch.serialization
            torch.serialization.add_safe_globals([multiarray.scalar])
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.service_levels = checkpoint['service_levels']
        self.losses = checkpoint['losses']
        print(f"Model weights loaded from {filepath}")
    
    def train(self, env: InventoryEnvironment, episodes: int = 1000):
        """Train the DQN agent with better monitoring"""
        print(f"Training DQN agent for {episodes} episodes...")
        print(f"Action multipliers: {env.action_multipliers}")
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_info = {
                'stockouts': 0,
                'total_demand': 0,
                'total_sold': 0,
                'orders_placed': 0
            }
            
            while True:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                self.store_experience(state, action, reward, next_state, done)
                self.learn()
                
                episode_reward += reward
                episode_info['total_demand'] += info['demand']
                episode_info['total_sold'] += info['sold']
                if info['stockout']:
                    episode_info['stockouts'] += 1
                if info['order_placed']:
                    episode_info['orders_placed'] += 1
                
                state = next_state
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            service_level = episode_info['total_sold'] / max(1, episode_info['total_demand'])
            self.service_levels.append(service_level)
            
            # IMPROVED: Better progress reporting
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_service = np.mean(self.service_levels[-50:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                
                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"  Average Reward (last 50): {avg_reward:.2f}")
                print(f"  Average Service Level: {avg_service:.3f}")
                print(f"  Stockouts: {episode_info['stockouts']}")
                print(f"  Orders Placed: {episode_info['orders_placed']}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Average Loss: {avg_loss:.4f}")
                
                # Show sample reward breakdown
                if len(env.debug_info['rewards_breakdown']) > 0:
                    last_reward = env.debug_info['rewards_breakdown'][-1]
                    print(f"  Last reward components: {last_reward['components']}")
                    print(f"  Chosen vs Optimal multiplier: {last_reward['chosen_multiplier']:.2f} vs {last_reward['optimal_multiplier']:.2f}")
        
        # Save model weights at the end of training
        self.save_model()


# Example usage with debugging
# from test_dqn_eoq import run_comprehensive_demo
if __name__ == "__main__":
    # Load your datasets
    print("Loading datasets...")
    
    # Create sample data for testing if files don't exist
    try:
        sales_df = pd.read_csv("synthetic_retail_Sales_Data.csv")
        inventory_df = pd.read_csv("EOQ_Training_Data.csv")
    except:
        print("Creating sample data for testing...")
        # Create minimal sample data for testing
        sample_inventory = {
            'SKU': ['SKU_1'] * 10,
            'City': ['Berlin'] * 10,
            'Stock_Begin': [100] * 10,
            'Stock_End': [80] * 10,
            'Demand_Mean_7P': [20] * 10,
            'Demand_Std_7P': [5] * 10,
            'Demand_CV': [0.25] * 10,
            'Seasonality_Factor': [1.0] * 10,
            'Ordering_Cost': [50] * 10,
            'Holding_Cost_Per_Unit': [2] * 10,
            'Transportation_Cost': [10] * 10,
            'Unit_Value': [25] * 10,
            'Lead_Time_Mean': [3] * 10,
            'Lead_Time_Std': [1] * 10,
            'Classical_EOQ': [60] * 10,
            'Optimal_Multiplier': [1.0] * 10,
            'Stock_Turnover': [0.5] * 10
        }
        inventory_df = pd.DataFrame(sample_inventory)
        sales_df = pd.DataFrame({
            'SKU': ['SKU_1'] * 20,
            'City': ['Berlin'] * 20,
            'Units_Sold': np.random.normal(20, 5, 20).astype(int)
        })
    
    # Create environment
    env = InventoryEnvironment(sales_df, inventory_df, max_periods=26)
    
    # Create and train agent
    agent = DQNAgent(state_dim=env.n_states, action_dim=env.n_actions, learning_rate=0.001)
    
    # Train
    agent.train(env, episodes=500)
    
    print("\nTraining completed!")
    print(f"Final average reward: {np.mean(agent.episode_rewards[-50:]):.2f}")
    print(f"Final average service level: {np.mean(agent.service_levels[-50:]):.3f}")