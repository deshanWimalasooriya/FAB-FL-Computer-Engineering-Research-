import numpy as np
import math

class BSFLScheduler:
    def __init__(self, num_clients=20, m=5, ucb_c=2.0, alpha_reward=0.5, beta_reward=0.5):
        """
        Initializes the BSFL Scheduler.
        
        Args:
            num_clients: Total number of clients in the system (N=20).
            m: Number of clients to select in the final cohort (m=5).
            ucb_c: Exploration parameter 'c' for the UCB algorithm.
            alpha_reward: Weight for the data quality (FREQSEL distance) in the reward function.
            beta_reward: Weight for the computational speed in the reward function.
        """
        self.num_clients = num_clients
        self.m = m
        self.ucb_c = ucb_c
        
        # Reward function balancing weights
        self.alpha_reward = alpha_reward
        self.beta_reward = beta_reward
        
        # --- UCB Tracking Variables ---
        # N_k(t): Number of times client k has been selected
        self.selection_counts = np.zeros(num_clients)
        
        # \mu_k(t): Estimated average speed (e.g., 1 / latency) for client k
        self.estimated_speeds = np.zeros(num_clients)
        
        # Total number of communication rounds executed (t)
        self.total_rounds = 0
        
    def update_speed_records(self, selected_clients, client_latencies):
        """
        Updates the historical speed records for the clients selected in the previous round.
        Must be called after each communication round completes.
        """
        self.total_rounds += 1
        
        for client_id in selected_clients:
            latency = client_latencies[client_id]
            # Speed is inversely proportional to computational latency
            # We use max(latency, 1e-5) to prevent division by zero
            speed = 1.0 / max(latency, 1e-5)
            
            # Update selection count N_k(t)
            self.selection_counts[client_id] += 1
            
            # Update moving average of speed \mu_k(t)
            n = self.selection_counts[client_id]
            old_speed = self.estimated_speeds[client_id]
            self.estimated_speeds[client_id] = old_speed + (speed - old_speed) / n

    def calculate_ucb_speed(self, candidate_pool):
        r"""
        Calculates the UCB speed metric for the K candidates.
        
        Math:
        UCB_{k,t} = \mu_k(t) + c * \sqrt{ \ln(t) / N_k(t) }
        """
        ucb_scores = {}
        for k in candidate_pool:
            if self.selection_counts[k] == 0:
                # If a client has never been selected, grant infinite UCB score 
                # to strongly encourage exploration of unknown clients.
                ucb_scores[k] = float('inf')
            else:
                exploitation = self.estimated_speeds[k]
                exploration = self.ucb_c * math.sqrt(math.log(self.total_rounds) / self.selection_counts[k])
                ucb_scores[k] = exploitation + exploration
                
        return ucb_scores

    def schedule(self, candidate_pool, candidate_distances):
        """
        Phase 3: Scheduling Stage (BSFL).
        Selects the final m clients from the K candidates based on the combined reward A_t.
        
        Math:
        Reward A_{k,t} = \alpha * (1 - normalized_distance_k) + \beta * normalized_ucb_speed_k
        We maximize A_{k,t} to balance fast convergence (low distance) and low latency (high speed).
        """
        # 1. Calculate UCB speeds for the candidate pool
        ucb_scores = self.calculate_ucb_speed(candidate_pool)
        
        selected = []
        remaining_m = self.m
        
        # 2. Edge Case Handling: Initial rounds where some UCB scores are infinity (unexplored)
        # Prioritize exploring these clients, breaking ties using their FREQSEL distances.
        unexplored = [k for k in candidate_pool if ucb_scores[k] == float('inf')]
        if len(unexplored) > 0:
            # Sort unexplored clients by distance (lower distance is better)
            unexplored.sort(key=lambda k: candidate_distances[k])
            
            if len(unexplored) >= self.m:
                return unexplored[:self.m]
            else:
                # Take all unexplored, and fill the rest by maximizing reward for explored
                selected = unexplored
                remaining_m = self.m - len(unexplored)
                # Filter candidate pool to only explored clients for reward calculation
                candidate_pool = [k for k in candidate_pool if k not in unexplored]
                
        # 3. Normalize Distances and UCB speeds for fair combination in the reward function
        if len(candidate_pool) > 0:
            d_values = np.array([candidate_distances[k] for k in candidate_pool])
            u_values = np.array([ucb_scores[k] for k in candidate_pool])
            
            def min_max_norm(arr):
                ptp = np.ptp(arr)
                return (arr - np.min(arr)) / ptp if ptp > 0 else np.zeros_like(arr)
                
            norm_d = min_max_norm(d_values)
            norm_u = min_max_norm(u_values)
            
            # 4. Calculate Combined Reward A_{k,t}
            # We want to minimize distance (so we maximize 1 - norm_d) and maximize speed (norm_u)
            rewards = self.alpha_reward * (1.0 - norm_d) + self.beta_reward * norm_u
            
            # 5. Select the top 'remaining_m' clients maximizing the reward
            # np.argsort sorts ascending, so we take the last elements and reverse them
            best_indices = np.argsort(rewards)[-remaining_m:][::-1]
            
            for idx in best_indices:
                selected.append(candidate_pool[idx])
                
        return selected

# Example execution if run directly
if __name__ == "__main__":
    print("Executing Phase 3: bsfl_scheduler.py (BSFL Stage)")
    scheduler = BSFLScheduler(num_clients=20, m=5)
    
    # Mock data arriving from Phase 2 (FREQSEL)
    mock_candidates = [2, 5, 8, 12, 15, 18, 19, 1, 3, 7] # K=10
    mock_distances = {k: np.random.uniform(0.1, 0.9) for k in mock_candidates}
    
    # First round - selects top 5 based purely on distance (since speeds are UCB=inf)
    selected_m = scheduler.schedule(mock_candidates, mock_distances)
    print(f"Round 1 (Exploration) Selected clients: {selected_m}")
    
    # Simulate finishing the round and recording simulated latencies
    mock_latencies = {k: np.random.uniform(10.0, 50.0) for k in selected_m}
    scheduler.update_speed_records(selected_m, mock_latencies)
    print(f"Updated speed estimations for selected: {scheduler.estimated_speeds[selected_m]}")
