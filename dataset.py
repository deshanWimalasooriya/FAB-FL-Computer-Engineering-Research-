import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset

def get_cifar10(root_dir='./data'):
    """
    Downloads and prepares the CIFAR-10 dataset with standard augmentations.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=root_dir, train=False, download=True, transform=transform_test)

    return trainset, testset


def dirichlet_partition_cifar10(dataset, num_clients=20, alpha=0.5):
    """
    Partitions the dataset across num_clients using a Dirichlet distribution 
    to simulate Non-IID label distribution skew.
    
    Math/Logic: 
    For each class c, we sample a probability vector from a Dirichlet distribution:
        p_c ~ Dir(alpha)
    where p_c is a vector of length `num_clients`.
    This vector dictates the proportion of class c's samples assigned to each client.
    Smaller alpha values lead to higher heterogeneity (Non-IID).
    """
    num_classes = 10
    
    # Retrieve all targets (labels) from the dataset
    targets = np.array(dataset.targets)
    
    # Initialize lists to hold indices assigned to each client
    client_indices = [[] for _ in range(num_clients)]
    
    # Metadata dictionary to track class frequencies per client (f_i^c calculation precursor)
    client_class_counts = {i: {c: 0 for c in range(num_classes)} for i in range(num_clients)}
    
    for c in range(num_classes):
        # 1. Identify all sample indices for the current class c
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        num_samples_c = len(idx_c)
        
        # 2. Sample allocation proportions from Dirichlet distribution
        # p_c ~ Dir(alpha * [1, 1, ..., 1])
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # 3. Convert proportions to absolute sample counts per client
        counts = (proportions * num_samples_c).astype(int)
        
        # 4. Handle remainder due to integer truncation to ensure all samples are assigned
        remainder = num_samples_c - counts.sum()
        if remainder > 0:
            counts[np.random.choice(num_clients, remainder, replace=False)] += 1
            
        # 5. Split the class indices according to the computed counts
        split_indices = np.split(idx_c, np.cumsum(counts)[:-1])
        
        # 6. Assign split subsets to respective clients and update metadata
        for i in range(num_clients):
            client_indices[i].extend(split_indices[i].tolist())
            client_class_counts[i][c] = len(split_indices[i])
            
    # Shuffle the assigned indices within each client's partition to ensure mixed batches during training
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    # Wrap partitioned indices into PyTorch Subset objects
    client_datasets = {i: Subset(dataset, client_indices[i]) for i in range(num_clients)}
    
    return client_datasets, client_class_counts


def prepare_federated_data(num_clients=20, alpha=0.5, root_dir='./data'):
    """
    Main entry point for Phase 1 data preparation.
    """
    # Fix seed for reproducibility in simulation
    np.random.seed(42)
    
    trainset, testset = get_cifar10(root_dir=root_dir)
    client_datasets, client_class_counts = dirichlet_partition_cifar10(
        trainset, num_clients=num_clients, alpha=alpha)
        
    return client_datasets, client_class_counts, testset

# Example usage/test execution if run directly
if __name__ == "__main__":
    print("Executing Phase 1: dataset.py (Data Utilities)")
    print("Simulating N=20 clients with Dirichlet alpha=0.5...")
    client_datasets, client_class_counts, testset = prepare_federated_data(num_clients=20, alpha=0.5)
    
    print("\nSample Class Counts for Client 0:")
    print(client_class_counts[0])
    print(f"Total samples for Client 0: {len(client_datasets[0])}")
