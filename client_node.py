import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import ray
import numpy as np
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ray.remote
class ClientNode:
    def __init__(self, client_id, dataset_split, batch_size=32, local_epochs=10, learning_rate=0.01):
        """
        Initializes the Ray remote client node for Federated Learning.
        
        Args:
            client_id: Unique identifier for the client.
            dataset_split: PyTorch Subset containing this client's specific Non-IID data.
            batch_size: Batch size for local training.
            local_epochs: Number of epochs to train locally before sending weights back.
            learning_rate: Learning rate for local SGD optimizer.
        """
        self.client_id = client_id
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = learning_rate
        
        # DataLoader for local training
        self.dataloader = DataLoader(self.dataset_split, batch_size=self.batch_size, shuffle=True)
        
        # --- Simulated Hardware Constraints ---
        # Simulated hardware capability factor (e.g., FLOPS). 
        # Lower factor means a stronger/faster machine.
        # We assign this deterministically based on client_id to keep the simulation stable.
        np.random.seed(self.client_id + 42)
        self.hardware_factor = np.random.uniform(0.5, 2.0)
        
        # Simulated network latency (e.g., ping time in milliseconds)
        self.network_delay = np.random.uniform(10.0, 100.0)
        
        # Instantiate model once to avoid memory fragmentation per round
        self.model = self._create_model()

    def _simulate_latency(self, num_samples):
        """
        Calculates a deterministic simulated latency rather than using noisy physical clock time.
        
        Math/Logic:
        Total Latency = (Data Size * Hardware Factor * Computation Constant) + Network Delay
        This ensures our BSFL scheduler receives stable, realistic feedback even when running
        on a heavily multiplexed local laptop CPU.
        """
        # Assume processing one sample takes some arbitrary constant time unit (e.g., 2ms)
        computation_constant_ms = 2.0
        
        computation_time = num_samples * self.hardware_factor * computation_constant_ms * self.local_epochs
        total_latency_ms = computation_time + self.network_delay
        
        # Convert to seconds for standard logging and scheduler usage
        return total_latency_ms / 1000.0

    def _create_model(self):
        """
        Instantiates the ResNet-18 model locally to avoid sending the whole architecture 
        class over Ray serialization. Modifies the architecture specifically for CIFAR-10.
        """
        model = models.resnet18(num_classes=10)
        # Modify for CIFAR-10 32x32 image size (default ResNet expects 224x224)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    def train(self, global_model_state_dict):
        """
        Executes local training on the client's data.
        
        Args:
            global_model_state_dict: The state_dict of the global ResNet-18 model.
            
        Returns:
            updated_state_dict: The new weights after local training.
            metrics: A dictionary containing loss, accuracy, and simulated latency.
        """
        model = self.model
        model.load_state_dict(global_model_state_dict)
        model.to(device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        
        for epoch in range(self.local_epochs):
            total_loss = 0.0
            correct = 0
            total_samples = 0
            
            for data, target in self.dataloader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_samples += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        # Calculate local metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = 100. * correct / total_samples if total_samples > 0 else 0.0
        
        # Calculate deterministic simulated latency
        simulated_latency_sec = self._simulate_latency(total_samples)
        
        # Extract updated weights to send back to the server (move to CPU for Ray transfer)
        updated_state_dict = {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
        
        metrics = {
            'client_id': self.client_id,
            'num_samples': total_samples,
            'loss': avg_loss,
            'accuracy': accuracy,
            'latency': simulated_latency_sec
        }
        
        # Explicitly delete local variables that hold PyTorch memory
        # Note: We do NOT delete `self.model` to avoid memory fragmentation across rounds
        del criterion
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return updated_state_dict, metrics

if __name__ == "__main__":
    print("Executing Phase 4: client_node.py (Ray Remote Client Node)")
    print("This module defines the @ray.remote actor and is designed to be executed via Ray from the server orchestrator.")
