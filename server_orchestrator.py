import ray
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import copy
import gc

# Import custom modules from previous phases
from dataset import prepare_federated_data
from freqsel_filter import freqsel_filter
from bsfl_scheduler import BSFLScheduler
from client_node import ClientNode

def fedavg_aggregate(global_model, client_weights_list):
    r"""
    Performs Federated Averaging (FedAvg) on the collected client model weights.
    
    Math: w_{t+1} = \sum_{i \in S_t} \frac{1}{m} w_i^t
    In this implementation, we use uniform weighting across the selected m clients.
    """
    m = len(client_weights_list)
    avg_weights = OrderedDict()
    
    # Initialize avg_weights with zeros mapping to global model architecture
    for k in client_weights_list[0].keys():
        # Ensure tensor is cast correctly and detached
        avg_weights[k] = torch.zeros_like(torch.as_tensor(client_weights_list[0][k]), dtype=torch.float32)
        
    # Sum all weights from the selected cohort
    for client_weights in client_weights_list:
        for k in client_weights.keys():
            avg_weights[k] += torch.as_tensor(client_weights[k])
            
    # Divide by m to get the average
    for k in avg_weights.keys():
        avg_weights[k] = torch.div(avg_weights[k], m)
        
    # Apply aggregated weights to the global model
    global_model.load_state_dict(avg_weights)

def create_global_model():
    """Instantiates the ResNet-18 global model modified for CIFAR-10."""
    model = models.resnet18(num_classes=10)
    # Adjust for CIFAR-10's 32x32 image size (Standard ResNet expects 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def main():
    # 1. Initialize Ray cluster (Local Mode for laptop CPU simulation)
    # ignore_reinit_error prevents crashes if run multiple times in an interactive session
    ray.init(ignore_reinit_error=True)
    
    N = 10  # Total clients
    K = 5  # Candidate pool size (FREQSEL)
    m = 3   # Selected clients per round (BSFL)
    num_rounds = 10
    
    print("=== FAB-FL System Initialization ===")
    print(f"Total Clients (N): {N}, Candidates (K): {K}, Selected (m): {m}")
    
    # 2. Prepare Data (Phase 1)
    print("\n[Phase 1] Partitioning CIFAR-10 data (Dirichlet alpha=0.5)...")
    client_datasets, client_class_counts, testset = prepare_federated_data(num_clients=N, alpha=0.5)
    
    # 3. Initialize Global Model and Server State
    global_model = create_global_model()
    
    # 4. Instantiate Ray Remote Client Nodes (Phase 4)
    print("\nInitializing Ray Remote Client Nodes...")
    client_nodes = {}
    for i in range(N):
        client_nodes[i] = ClientNode.remote(
            client_id=i, 
            dataset_split=client_datasets[i],
            batch_size=32,
            local_epochs=1,
            learning_rate=0.01
        )
        
    # 5. Initialize the BSFL Scheduler (Phase 3)
    scheduler = BSFLScheduler(num_clients=N, m=m)
    
    print("\n=== Starting FAB-FL Training Loop ===")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        # --- Stage 1: FREQSEL Filtering (Phase 2) ---
        candidate_pool, candidate_distances = freqsel_filter(client_class_counts, K=K)
        print(f"FREQSEL Candidate Pool (K={K}): {candidate_pool}")
        
        # --- Stage 2: BSFL Scheduling (Phase 3) ---
        selected_clients = scheduler.schedule(candidate_pool, candidate_distances)
        print(f"BSFL Selected Clients (m={m}): {selected_clients}")
        
        # --- Stage 3: Local Training via Ray ---
        global_state_dict = global_model.state_dict()
        
        # Dispatch training tasks to the selected Ray actors asynchronously
        futures = []
        for client_id in selected_clients:
            node = client_nodes[client_id]
            # .remote() triggers the function on the Ray worker process
            futures.append(node.train.remote(global_state_dict))
            
        # Wait for all selected clients to finish training
        results = ray.get(futures)
        
        # Extract updated weights and simulated metrics
        client_weights_list = []
        round_latencies = {}
        round_loss = 0.0
        round_acc = 0.0
        
        for updated_weights, metrics in results:
            client_weights_list.append(updated_weights)
            c_id = metrics['client_id']
            round_latencies[c_id] = metrics['latency']
            round_loss += metrics['loss']
            round_acc += metrics['accuracy']
            
        avg_loss = round_loss / m
        avg_acc = round_acc / m
        # The bottleneck latency determines the speed of this communication round
        max_latency = max(round_latencies.values())
        
        print(f"Round {round_num} Local Training -> Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}% | Bottleneck Latency: {max_latency:.2f}s")
        
        # Feedback loop: Update Scheduler with the actual simulated latencies
        scheduler.update_speed_records(selected_clients, round_latencies)
        
        # --- Stage 4: Federated Aggregation (FedAvg) ---
        fedavg_aggregate(global_model, client_weights_list)
        
        # Clear Ray Object Store and Collect Garbage
        del futures
        del results
        gc.collect()
        
    print("\n=== FAB-FL Training Complete ===")
    ray.shutdown()

if __name__ == "__main__":
    main()
