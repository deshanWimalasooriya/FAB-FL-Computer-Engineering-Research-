import os
import io
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import gc
from torch.utils.data import DataLoader

from dataset import prepare_federated_data

EPOCHS = 1
BATCH_SIZE = 32

client_state = {
    "local_clients": [], # Will store dicts with global_id, class_counts
    "current_status": "Initializing...",
    "logs": [],
    "active_training_gid": None,
    "memory_cleared_msg": "",
    "partition_progress": 0.0,
    "stop_flag": False
}

def log(msg):
    print(msg)
    client_state["logs"].append(msg)
    if len(client_state["logs"]) > 100:
        client_state["logs"] = client_state["logs"][-100:]

def update_progress(val):
    client_state["partition_progress"] = val

def connect_device(server_url="http://127.0.0.1:8000", N=20, device_name="Laptop-2"):
    log(f"=== FAB-FL Client Simulator ({device_name}) ===")
    
    # Connect Device First
    client_state["current_status"] = "Connecting to Server..."
    log(f"Connecting device '{device_name}' to server at {server_url}...")
    try:
        res = requests.post(f"{server_url}/connect_device", json={
            "device_name": device_name,
            "num_clients": N
        }).json()
        if res.get("status") == "success":
            client_state["current_status"] = "Connected!"
            log("Successfully connected to Server.")
            return True
        else:
            log(f"Connection failed: {res}")
            client_state["current_status"] = "Connection Failed"
            return False
    except Exception as e:
        log(f"Failed to reach Server: {e}")
        client_state["current_status"] = "Server Unreachable"
        return False

def start_process(server_url="http://127.0.0.1:8000", N=20):
    client_state["stop_flag"] = False
    
    # 1. Initialize dataset and non-IID partitioning
    log(f"Initializing {N} virtual clients locally and partitioning data...")
    client_state["current_status"] = "Partitioning Data (Downloading if needed)..."
    client_datasets, client_class_counts, _ = prepare_federated_data(
        num_clients=N, alpha=0.5, progress_callback=update_progress
    )
    
    if client_state["stop_flag"]:
        log("Process stopped.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # 2. Registration Handshake
    local_idx_to_global = {}
    log(f"Registering clients with Server at {server_url}...")
    client_state["current_status"] = "Registering..."
    
    # Clear local clients for fresh start
    client_state["local_clients"] = []
    
    for i in range(N):
        if client_state["stop_flag"]:
            log("Process stopped during registration.")
            return
            
        temp_id = f"FFC-{i:04d}"
        
        # Simulate initial hardware speed for registration
        import random
        initial_speed = random.uniform(20.0, 50.0)
        
        payload = {
            "temp_id": temp_id,
            "class_counts": client_class_counts[i],
            "hardware_speed_ms": initial_speed
        }
        
        try:
            res = requests.post(f"{server_url}/register", json=payload).json()
            global_id = res["global_id"]
            local_idx_to_global[i] = global_id
            log(f"  Registered {temp_id} -> {global_id}")
            
            client_state["local_clients"].append({
                "global_id": global_id,
                "class_counts": client_class_counts[i]
            })
        except Exception as e:
            log(f"Failed to register client {i}: {e}")
            return
            
    # 3. Polling Loop
    local_round = 0
    log("\nWaiting for Server to start training rounds...")
    client_state["current_status"] = "Waiting for Server..."
    
    while not client_state["stop_flag"]:
        try:
            status_res = requests.get(f"{server_url}/status").json()
            server_round = status_res.get("round", 0)
            in_progress = status_res.get("round_in_progress", False)
            selected_gids = status_res.get("selected_clients", [])
            
            if in_progress and server_round > local_round:
                log(f"\n--- Round {server_round} Started ---")
                local_round = server_round
                client_state["current_status"] = f"Round {server_round} in progress..."
                
                # Find which of our local clients were selected
                our_selected = []
                for idx, gid in local_idx_to_global.items():
                    if gid in selected_gids:
                        our_selected.append((idx, gid))
                        
                if not our_selected:
                    log("None of our local clients were selected for this round.")
                    client_state["current_status"] = "Idle (Not Selected)"
                    time.sleep(2)
                    continue
                    
                log(f"Our selected clients: {[gid for _, gid in our_selected]}")
                
                # 4. Download Global Model
                representative_gid = our_selected[0][1]
                log(f"Downloading global model via {representative_gid}...")
                model_res = requests.get(f"{server_url}/get_model", params={"global_id": representative_gid})
                
                if model_res.status_code != 200:
                    log("Failed to download global model.")
                    time.sleep(2)
                    continue
                    
                global_model_path = "local_global_model.pt"
                with open(global_model_path, "wb") as f:
                    f.write(model_res.content)
                    
                # 5. Sequential Training & Strict Memory Management
                for idx, gid in our_selected:
                    if client_state["stop_flag"]:
                        break
                        
                    client_state["active_training_gid"] = gid
                    client_state["current_status"] = f"Training {gid}..."
                    log(f"[{gid}] Starting local training...")
                    start_time = time.time()
                    
                    # Instantiate fresh model
                    model = models.resnet18(num_classes=10)
                    model.load_state_dict(torch.load(global_model_path, weights_only=True))
                    model = model.to(device)
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                    
                    dataloader = DataLoader(client_datasets[idx], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                    
                    model.train()
                    for epoch in range(EPOCHS):
                        if client_state["stop_flag"]:
                            break
                        for inputs, labels in dataloader:
                            if client_state["stop_flag"]:
                                break
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            
                    train_time_ms = (time.time() - start_time) * 1000
                    log(f"[{gid}] Training completed in {train_time_ms:.2f} ms")
                    
                    # Save to memory buffer
                    buffer = io.BytesIO()
                    model = model.cpu()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    
                    # Memory Cleanup
                    del model
                    del optimizer
                    del criterion
                    del inputs
                    del labels
                    del outputs
                    del loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    client_state["memory_cleared_msg"] = f"Memory cleared for {gid}"
                    log(f"[{gid}] Executed: del model, torch.cuda.empty_cache(), gc.collect()")
                    
                    if client_state["stop_flag"]:
                        break
                        
                    # Upload model
                    client_state["current_status"] = f"Uploading {gid}..."
                    files = {"file": (f"{gid}.pt", buffer, "application/octet-stream")}
                    params = {"global_id": gid, "hardware_speed_ms": train_time_ms}
                    upload_res = requests.post(f"{server_url}/upload_model", params=params, files=files)
                    
                    if upload_res.status_code == 200:
                        log(f"[{gid}] Successfully uploaded weights.")
                    else:
                        log(f"[{gid}] Failed to upload weights: {upload_res.text}")
                
                client_state["active_training_gid"] = None
                client_state["current_status"] = "Waiting for Server..."
                        
        except requests.exceptions.ConnectionError:
            pass # Server might be down, just wait
            
        time.sleep(2) # Poll every 2 seconds
        
    client_state["current_status"] = "Stopped"
    log("Loop terminated.")

def stop_process():
    client_state["stop_flag"] = True
    client_state["current_status"] = "Stopping..."
    log("Stop requested.")

if __name__ == "__main__":
    pass
