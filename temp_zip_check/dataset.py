import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import ssl

# Fix for PyInstaller SSL certificate issues when downloading datasets
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import os

def get_cifar10(root_dir=None):
    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
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


def dirichlet_partition_cifar10(dataset, num_clients=20, alpha=0.5, progress_callback=None):
    """
    Partitions the dataset across num_clients using a Dirichlet distribution 
    to simulate Non-IID label distribution skew.
    Runs on CUDA to accelerate the array permutations and Dirichlet sampling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Data partitioning using device: {device}")
    
    num_classes = 10
    
    # Retrieve all targets (labels) from the dataset, move to CUDA
    targets = torch.tensor(dataset.targets, device=device)
    
    # Initialize lists to hold indices assigned to each client
    client_indices = [[] for _ in range(num_clients)]
    
    # Metadata dictionary to track class frequencies per client
    client_class_counts = {i: {c: 0 for c in range(num_classes)} for i in range(num_clients)}
    
    # PyTorch Dirichlet distribution on GPU
    dirichlet_dist = torch.distributions.dirichlet.Dirichlet(torch.full((num_clients,), alpha, device=device))
    
    for c in range(num_classes):
        if progress_callback:
            progress_callback(c / num_classes)
            
        # 1. Identify all sample indices for the current class c
        idx_c = torch.where(targets == c)[0]
        idx_c = idx_c[torch.randperm(len(idx_c), device=device)]
        num_samples_c = len(idx_c)
        
        # 2. Sample allocation proportions from Dirichlet distribution
        proportions = dirichlet_dist.sample()
        
        # 3. Convert proportions to absolute sample counts per client
        counts = (proportions * num_samples_c).to(torch.int32)
        
        # 4. Handle remainder due to integer truncation to ensure all samples are assigned
        remainder = num_samples_c - counts.sum().item()
        if remainder > 0:
            add_idx = torch.randperm(num_clients, device=device)[:remainder]
            counts[add_idx] += 1
            
        # 5. Split the class indices according to the computed counts
        split_indices = torch.split(idx_c, counts.tolist())
        
        # 6. Assign split subsets to respective clients and update metadata
        for i in range(num_clients):
            client_indices[i].extend(split_indices[i].cpu().tolist())
            client_class_counts[i][c] = len(split_indices[i])
            
    # Shuffle the assigned indices within each client's partition to ensure mixed batches during training
    for i in range(num_clients):
        idx_tensor = torch.tensor(client_indices[i])
        shuffled = idx_tensor[torch.randperm(len(idx_tensor))]
        client_indices[i] = shuffled.tolist()
        
    if progress_callback:
        progress_callback(1.0)
        
    # Wrap partitioned indices into PyTorch Subset objects
    client_datasets = {i: Subset(dataset, client_indices[i]) for i in range(num_clients)}
    
    return client_datasets, client_class_counts


def prepare_federated_data(num_clients=20, alpha=0.5, root_dir=None, progress_callback=None):
    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    """
    Main entry point for Phase 1 data preparation.
    """
    # Fix seed for reproducibility in simulation
    np.random.seed(42)
    
    trainset, testset = get_cifar10(root_dir=root_dir)
    client_datasets, client_class_counts = dirichlet_partition_cifar10(
        trainset, num_clients=num_clients, alpha=alpha, progress_callback=progress_callback)
        
    return client_datasets, client_class_counts, testset

def plot_client_data_distribution(client_class_counts, num_classes=10):
    """
    Plots the data distribution across clients using a stacked bar chart.
    Each of the 10 classes is represented by a different color.
    """
    num_clients = len(client_class_counts)
    
    # Prepare data for plotting: matrix of shape (num_classes, num_clients)
    distribution_matrix = np.zeros((num_classes, num_clients))
    for client_id, counts in client_class_counts.items():
        for class_id, count in counts.items():
            distribution_matrix[class_id, client_id] = count
            
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 10 distinct colors for the 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    bottom = np.zeros(num_clients)
    client_indices = np.arange(num_clients)
    
    for class_id in range(num_classes):
        ax.bar(client_indices, distribution_matrix[class_id], bottom=bottom, 
               color=colors[class_id], label=f'Class {class_id}')
        bottom += distribution_matrix[class_id]
        
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('CIFAR-10 Data Distribution among Clients (Dirichlet)')
    ax.set_xticks(client_indices)
    # Put legend outside the plot
    ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('client_data_distribution.png')
    print("Saved data distribution plot to 'client_data_distribution.png'")
    # Note: Using block=False or skipping show() might be better for automated scripts,
    # but plt.show() allows the user to see it. 
    # plt.show()


# Example usage/test execution if run directly
if __name__ == "__main__":
    print("Executing Phase 1: dataset.py (Data Utilities)")
    print("Simulating N=20 clients with Dirichlet alpha=0.5...")
    client_datasets, client_class_counts, testset = prepare_federated_data(num_clients=20, alpha=0.5)
    
    print("\nSample Class Counts for Client 0:")
    print(client_class_counts[0])
    print(f"Total samples for Client 0: {len(client_datasets[0])}")
    
    print("\nPlotting Data Distribution...")
    plot_client_data_distribution(client_class_counts)
