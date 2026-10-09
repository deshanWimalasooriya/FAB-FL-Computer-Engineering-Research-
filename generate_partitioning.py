import numpy as np
import json
import torchvision
import os

def generate_dirichlet_partition(dataset, num_clients, alpha):
    """
    Partitions the dataset into num_clients using a Dirichlet distribution over the classes.
    """
    # CIFAR-10 targets
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    
    # Initialize the dictionary to store client indices
    client_indices = {f"client_{i}": [] for i in range(num_clients)}
    
    # For each class, distribute the indices among clients based on Dirichlet distribution
    for k in range(num_classes):
        class_idx = np.where(targets == k)[0]
        np.random.shuffle(class_idx)
        
        # Generate proportions for each client using Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Convert proportions to integer counts
        counts = np.round(proportions * len(class_idx)).astype(int)
        
        # Fix any rounding errors to ensure all samples are assigned
        diff = len(class_idx) - np.sum(counts)
        if diff > 0:
            for i in range(diff):
                counts[np.random.randint(0, num_clients)] += 1
        elif diff < 0:
            for i in range(-diff):
                # Only decrement if count > 0
                non_zero = np.where(counts > 0)[0]
                if len(non_zero) > 0:
                    counts[np.random.choice(non_zero)] -= 1
                    
        # Assign indices to clients based on counts
        current_idx = 0
        for i in range(num_clients):
            assigned_indices = class_idx[current_idx:current_idx + counts[i]]
            client_indices[f"client_{i}"].extend(assigned_indices.tolist())
            current_idx += counts[i]
            
    return client_indices

def main():
    print("Downloading/Loading CIFAR-10 dataset...")
    # We only need the dataset targets, no complex transforms are required for partitioning
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    num_clients = 10
    alpha = 0.5
    
    print(f"Partitioning data for {num_clients} clients with Dirichlet alpha={alpha}...")
    # Generate the partition mapping
    mapping = generate_dirichlet_partition(dataset, num_clients=num_clients, alpha=alpha)
    
    # Save the mapping to a JSON file
    output_file = "client_data_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Partitioning complete! Saved mapping to {output_file}")
    
    # Verify the partition sizes
    for client_id, indices in mapping.items():
        print(f"{client_id}: {len(indices)} samples assigned")

if __name__ == "__main__":
    main()
