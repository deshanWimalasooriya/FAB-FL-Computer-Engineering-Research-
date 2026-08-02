import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_local_frequencies(client_class_counts, num_classes=10):
    r"""
    Computes local class frequencies (f_i^c) for all clients.
    
    Math:
    f_i^c = n_i^c / \sum_{c'} n_i^{c'}
    where n_i^c is the number of samples for class c in client i's local data.
    """
    num_clients = len(client_class_counts)
    f_i = np.zeros((num_clients, num_classes))
    
    for i in range(num_clients):
        # Extract class counts for client i into an array
        counts = np.array([client_class_counts[i][c] for c in range(num_classes)])
        total_samples = np.sum(counts)
        
        if total_samples > 0:
            f_i[i] = counts / total_samples
        else:
            # Fallback for empty client partitions, safely defaults to uniform.
            f_i[i] = np.ones(num_classes) / num_classes
            
    return f_i

def compute_distances(f_i, F_pop):
    """
    Calculates distance d_i between each client's local frequency distribution f_i
    and the target global population frequency F_pop.
    
    Math:
    d_i = Distance(f_i, F_pop)
    We use Jensen-Shannon distance as a symmetric, smoothed metric well-suited for 
    comparing probability distributions, especially since Non-IID Dirichlet splits 
    frequently produce zero-probability classes that would break standard KL-Divergence.
    """
    num_clients = f_i.shape[0]
    distances = np.zeros(num_clients)
    
    for i in range(num_clients):
        # JS distance returns a metric bounded between [0, 1] 
        # where 0 means distributions are identical.
        distances[i] = jensenshannon(f_i[i], F_pop)
        
    return distances

def evaluate_all_distances(client_class_counts, num_classes=10):
    """
    Executes Phase 2: Metadata Collection & Evaluation (FREQSEL).
    Evaluates the distributional distance d_i for all N clients without truncation.
    """
    # 1. Compute local class frequency vectors f_i^c for all clients
    f_i = compute_local_frequencies(client_class_counts, num_classes)
    
    # 2. Compute the global population frequency vector F_pop^c.
    # Because CIFAR-10's training set is perfectly balanced globally, 
    # the ideal target F_pop is a uniform distribution: F_pop^c = 1 / C
    F_pop = np.ones(num_classes) / num_classes
    
    # 3. Calculate distributional distance d_i for all clients
    distances = compute_distances(f_i, F_pop)
    
    # 4. Return distance metadata for all N clients
    candidate_distances = {client_id: distances[client_id] for client_id in range(len(client_class_counts))}
    
    return candidate_distances

# Example execution if run directly
if __name__ == "__main__":
    print("Executing Phase 2: freqsel_filter.py (FREQSEL Stage)")
    # Mocking some data for demonstration
    mock_class_counts = {
        0: {c: 500 if c == 0 else 55 for c in range(10)},  # Highly skewed
        1: {c: 100 for c in range(10)},                    # Perfectly IID
        2: {c: 200 if c in [1, 2] else 66 for c in range(10)} # Moderately skewed
    }
    
    # We evaluate all clients directly
    dist_info = evaluate_all_distances(mock_class_counts)
    print("\nEvaluated all clients.")
    print("Distances:", dist_info)
