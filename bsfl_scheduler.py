import numpy as np
import math

class BSFLScheduler:
    def __init__(self, num_clients=20, m=5, ucb_c=2.0):
        """
        Initializes the BSFL Scheduler.
        
        Args:
            num_clients: Total number of clients in the system (N=20).
            m: Number of clients to select in the final cohort (m=5).
            ucb_c: Exploration parameter 'c' for the UCB algorithm.
        """
        self.num_clients = num_clients
        self.m = m
        self.ucb_c = ucb_c
        
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

    def calculate_ucb_scores(self, candidate_pool):
        r"""
        Calculates the UCB score for clients in the candidate pool.
        
        Math:
        A_t = R_i + c * \sqrt{ \ln(t) / N_i }
        Where R_i is normalized historical speed.
        """
        ucb_scores = {}
        
        # Normalize speeds (R_i)
        speeds = np.array([self.estimated_speeds[k] for k in candidate_pool])
        def min_max_norm(arr):
            ptp = np.ptp(arr)
            return (arr - np.min(arr)) / ptp if ptp > 0 else np.zeros_like(arr)
            
        R_i = min_max_norm(speeds)
        
        for idx, k in enumerate(candidate_pool):
            if self.selection_counts[k] == 0:
                # If a client has never been selected, grant infinite UCB score 
                # to strongly encourage exploration of unknown clients.
                ucb_scores[k] = float('inf')
            else:
                exploitation = R_i[idx]
                exploration = self.ucb_c * math.sqrt(math.log(self.total_rounds) / self.selection_counts[k])
                ucb_scores[k] = exploitation + exploration
                
        return ucb_scores

    def schedule(self, candidate_pool):
        """
        Phase 3: Scheduling Stage (BSFL).
        Selects the final m clients from the dynamically scaled candidate pool K_new.
        """
        # Ensure we don't try to select more than what's available
        actual_m = min(self.m, len(candidate_pool))
        if actual_m == 0:
            return []
            
        # 1. Calculate UCB scores
        ucb_scores = self.calculate_ucb_scores(candidate_pool)
        
        # 2. Select top m clients
        # Sort clients by UCB score descending
        sorted_candidates = sorted(candidate_pool, key=lambda k: ucb_scores[k], reverse=True)
        
        return sorted_candidates[:actual_m]

# Example execution if run directly
if __name__ == "__main__":
    print("Executing Phase 3: bsfl_scheduler.py (BSFL Stage)")
    scheduler = BSFLScheduler(num_clients=20, m=5)
    
    # Mock data arriving from Phase 2 (FREQSEL)
    mock_candidate_pool = list(range(10)) # K_new has 10 clients
    
    # First round
    selected_m = scheduler.schedule(mock_candidate_pool)
    print(f"Round 1 Selected clients: {selected_m}")
    
    # Simulate finishing the round
    mock_latencies = {k: np.random.uniform(10.0, 50.0) for k in selected_m}
    scheduler.update_speed_records(selected_m, mock_latencies)
    print(f"Updated speed estimations for selected: {scheduler.estimated_speeds[selected_m]}")
