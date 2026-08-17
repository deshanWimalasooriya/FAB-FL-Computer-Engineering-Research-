import numpy as np
from scipy.stats import entropy
import itertools

def compute_local_frequencies(client_class_counts, num_classes=10):
    r"""
    Computes local class frequencies (P_i(c)) for all clients.
    """
    num_clients = len(client_class_counts)
    f_i = np.zeros((num_clients, num_classes))
    
    for i in range(num_clients):
        counts = np.array([client_class_counts[i][c] for c in range(num_classes)])
        total_samples = np.sum(counts)
        if total_samples > 0:
            f_i[i] = counts / total_samples
        else:
            f_i[i] = np.ones(num_classes) / num_classes
    return f_i

def compute_kl_divergences(f_i, F_pop):
    """
    Calculates KL Divergence D_KL(P_i || Q) between each client's 
    local frequency distribution P_i and target global Q (F_pop).
    """
    num_clients = f_i.shape[0]
    distances = np.zeros(num_clients)
    for i in range(num_clients):
        distances[i] = entropy(pk=f_i[i], qk=F_pop)
    return distances

def select_candidates(client_class_counts, num_classes=10, k=7, lambda_pen=1e6, gamma=0.5):
    """
    Executes Phase 2: Metadata Collection & Evaluation (FREQSEL).
    Filters total pool of N clients down to candidate pool K.
    Implements Penalty for Class Starvation and Dynamic K-Scaling.
    """
    num_clients = len(client_class_counts)
    f_i = compute_local_frequencies(client_class_counts, num_classes)
    
    # Target uniform distribution Q
    F_pop = np.ones(num_classes) / num_classes
    
    # KL Divergence for all clients
    distances = compute_kl_divergences(f_i, F_pop)
    
    # 1. Candidate Filtering (With Penalty)
    best_S = None
    min_penalty = float('inf')
    
    # To prevent exploding combinations if N is very large, limit to a greedy search 
    # if combinations are too massive, but for N=10,20 it's fine.
    if num_clients > 25:
        # Simple greedy fallback just sorts by distance
        best_S = tuple(np.argsort(distances)[:k])
    else:
        for S in itertools.combinations(range(num_clients), k):
            div_sum = np.sum(distances[list(S)])
            
            # Starvation Penalty
            # sum_{i in S} P_i(c)
            class_densities = np.sum(f_i[list(S)], axis=0) 
            penalty = lambda_pen * np.sum(np.maximum(0, gamma - class_densities))
            
            total_cost = div_sum + penalty
            if total_cost < min_penalty:
                min_penalty = total_cost
                best_S = S
                
    K_pool = list(best_S)
    
    # 2. Algorithmic Fallback: Dynamic K-Scaling
    # Verify Coverage: UK = {c | P_i(c) > 0 for i in K}
    covered_classes = set()
    for i in K_pool:
        for c in range(num_classes):
            if client_class_counts[i][c] > 0:
                covered_classes.add(c)
                
    K_new = set(K_pool)
    
    while len(covered_classes) < num_classes:
        missing_classes = set(range(num_classes)) - covered_classes
        c_miss = list(missing_classes)[0]
        
        # Find client j not in K with max density of c_miss
        excluded_pool = [j for j in range(num_clients) if j not in K_new]
        if not excluded_pool:
            break # Exhausted all clients
            
        best_j = max(excluded_pool, key=lambda j: f_i[j][c_miss])
        K_new.add(best_j)
        
        # Update covered classes with best_j's data
        for c in range(num_classes):
            if client_class_counts[best_j][c] > 0:
                covered_classes.add(c)
                
    final_K = list(K_new)
    all_distances_dict = {client_id: distances[client_id] for client_id in range(num_clients)}
    
    return final_K, all_distances_dict

if __name__ == "__main__":
    print("Executing Phase 2: freqsel_filter.py (FREQSEL Stage)")
    mock_class_counts = {
        0: {c: 500 if c == 0 else 55 for c in range(10)},
        1: {c: 100 for c in range(10)},
        2: {c: 200 if c in [1, 2] else 66 for c in range(10)},
        3: {c: 100 if c == 3 else 0 for c in range(10)},
        4: {c: 100 if c == 4 else 0 for c in range(10)},
        5: {c: 100 if c == 5 else 0 for c in range(10)},
        6: {c: 100 if c == 6 else 0 for c in range(10)},
        7: {c: 100 if c == 7 else 0 for c in range(10)},
        8: {c: 100 if c == 8 else 0 for c in range(10)},
        9: {c: 100 if c == 9 else 0 for c in range(10)},
    }
    
    final_pool, dists = select_candidates(mock_class_counts, k=5)
    print("\nFiltered candidate pool:", final_pool)
    print("Distances:", dists)
