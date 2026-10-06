import sys
import os
import io
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import math
import uuid
import torch
import uvicorn
import threading
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any
import numpy as np

from freqsel_filter import select_candidates
from dataset import get_cifar10
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

try:
    _, testset = get_cifar10()
except Exception as e:
    print(f"Failed to load testset: {e}")
    testset = None

app = FastAPI(title="FAB-FL Server", version="2.0")

# --- Global State ---
class ServerState:
    def __init__(self):
        self.clients: Dict[str, Dict] = {}       # Maps FAB-FL-XXXX -> client info
        self.temp_to_global: Dict[str, str] = {} # Maps FFC-XXXX -> FAB-FL-XXXX
        self.connected_devices: List[Dict] = []  # Stores connected physical devices (Edge Nodes)
        
        self.current_round = 0
        self.max_rounds = 50
        self.m = 3 # number of clients to select per round
        self.k = 1 # k parameter
        
        self.candidate_pool = []
        self.selected_clients = []
        self.uploaded_models: Dict[str, bytes] = {}
        self.global_model_path = "global_model.pt"
        self.round_in_progress = False
        self.stop_requested = False
        self.is_freqsel_enabled = True
        self.round_metrics = {}
        self.aggregating = False

state = ServerState()

# Pydantic schemas for request validation
class RegisterRequest(BaseModel):
    temp_id: str
    class_counts: Dict[int, int]
    hardware_speed_ms: float

class ConnectDeviceRequest(BaseModel):
    device_name: str
    num_clients: int
    hardware_name: str = "Unknown HW"

class SetParamsRequest(BaseModel):
    max_rounds: int
    m: int
    k: float

class ToggleFreqselRequest(BaseModel):
    enabled: bool


# --- 1. Registration Handshake ---
@app.post("/connect_device")
def connect_device(req: ConnectDeviceRequest):
    # Check if already connected
    for device in state.connected_devices:
        if device["device_name"] == req.device_name:
            device["num_clients"] = req.num_clients
            device["hardware_name"] = req.hardware_name
            return {"status": "success", "message": "Device updated"}
            
    state.connected_devices.append({
        "device_name": req.device_name,
        "num_clients": req.num_clients,
        "hardware_name": req.hardware_name
    })
    print(f"[SERVER] New device connected: {req.device_name} ({req.hardware_name}) hosting {req.num_clients} clients.")
    return {"status": "success"}

@app.post("/set_params")
def set_params(req: SetParamsRequest):
    state.max_rounds = req.max_rounds
    state.m = req.m
    state.k = req.k
    print(f"[SERVER] Parameters updated: max_rounds={state.max_rounds}, m={state.m}, k={state.k}")
    return {"status": "success"}

@app.post("/toggle_freqsel")
def toggle_freqsel(req: ToggleFreqselRequest):
    state.is_freqsel_enabled = req.enabled
    mode_str = "ENABLED" if state.is_freqsel_enabled else "DISABLED (Baseline Random Mode)"
    print(f"[SERVER] FREQSEL Filter: {mode_str}")
    return {"status": "success"}

@app.post("/register")
def register_client(req: RegisterRequest):
    if req.temp_id in state.temp_to_global:
        return {"global_id": state.temp_to_global[req.temp_id]}

    # Assign permanent ID
    client_num = len(state.clients) + 1
    global_id = f"FAB-FL-{client_num:04d}"
    
    state.temp_to_global[req.temp_id] = global_id
    state.clients[global_id] = {
        "class_counts": req.class_counts,
        "hardware_speed_ms": req.hardware_speed_ms,
        "N_k": 0, # Times selected
        "distance": 0.0, # Will be computed
        "ucb_score": 0.0 # Will be computed
    }
    print(f"[REGISTER] {req.temp_id} -> {global_id}")
    return {"global_id": global_id}

def auto_trigger_round():
    time.sleep(3)
    start_round()

@app.post("/reset")
def reset_server():
    state.clients.clear()
    state.temp_to_global.clear()
    state.connected_devices.clear()
    state.current_round = 0
    state.candidate_pool = []
    state.selected_clients = []
    state.uploaded_models.clear()
    state.round_in_progress = False
    state.stop_requested = False
    state.aggregating = False
    state.round_metrics.clear()
    # Don't reset is_freqsel_enabled so user settings persist
    if os.path.exists(state.global_model_path):
        try:
            os.remove(state.global_model_path)
        except Exception:
            pass
    print("\n[SERVER] State has been completely reset.")
    return {"status": "success"}

# --- 2. UCB Joint Scoring ---
def run_ucb_selection():
    N = len(state.clients)
    if N == 0:
        return []
    
    idx_to_gid = list(state.clients.keys())
    formatted_counts = {idx: state.clients[gid]["class_counts"] for idx, gid in enumerate(idx_to_gid)}
    
    if not getattr(state, 'is_freqsel_enabled', True):
        # BASELINE MODE: Candidate pool is ALL connected clients. BSFL will select m from this pool.
        for gid in idx_to_gid:
            state.clients[gid]["distance"] = 0.0
            
        candidate_gids = idx_to_gid.copy()
    else:
        # 1. FREQSEL Candidate Filtering
        # Use user-defined state.k, bounded by N-1
        initial_k = max(1, min(int(state.k), N - 1)) if N > 1 else 1
        
        candidate_pool_idx, all_distances_dict = select_candidates(formatted_counts, k=initial_k)
        
        # Update GUI distances for ALL clients immediately
        for idx, gid in enumerate(idx_to_gid):
            state.clients[gid]["distance"] = all_distances_dict[idx]
            
        candidate_gids = [idx_to_gid[idx] for idx in candidate_pool_idx]
        
    state.candidate_pool = candidate_gids
    
    # 2. Extract arrays for UCB (BSFL)
    speeds = np.array([state.clients[gid]["hardware_speed_ms"] for gid in candidate_gids])
    N_k_counts = np.array([state.clients[gid]["N_k"] for gid in candidate_gids])
    
    def min_max_norm(arr):
        if np.max(arr) == np.min(arr): return np.ones_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    # R_i is normalized computational speed (inverse latency)
    speed_inv = 1.0 / (speeds + 1e-9)
    R_i = min_max_norm(speed_inv)
    
    # UCB Parameters
    c = 1.0
    t = state.current_round + 1 # avoid ln(0)
    
    ucb_scores = []
    for i in range(len(candidate_gids)):
        gid = candidate_gids[i]
        
        exploitation = R_i[i]
        exploration = c * math.sqrt(math.log(t) / (N_k_counts[i] + 1e-9))
        
        # If never selected, force exploration by assigning infinity
        score = float('inf') if N_k_counts[i] == 0 else exploitation + exploration
        ucb_scores.append(score)
        
        # Save to state for GUI
        state.clients[gid]["ucb_score"] = score
        
    # Select top m + delta clients for reserve strategy
    delta = 2
    actual_m = min(state.m + delta, len(candidate_gids))
    top_m_idx = np.argsort(ucb_scores)[-actual_m:]
        
    selected_gids = [candidate_gids[idx] for idx in top_m_idx]
    
    # Update N_k for selected clients
    for gid in selected_gids:
        state.clients[gid]["N_k"] += 1
        
    return selected_gids

# --- 3. Round Orchestration ---
@app.post("/start_round")
def start_round():
    if state.round_in_progress:
        return {"error": "Round already in progress"}
        
    state.current_round += 1
    state.uploaded_models.clear()
    
    state.selected_clients = run_ucb_selection()
    state.round_in_progress = True
    state.aggregating = False
    
    print(f"\n[ROUND {state.current_round}] Started. Selected Clients (with reserve): {state.selected_clients}")
    return {"status": "started", "round": state.current_round, "selected": state.selected_clients}

@app.get("/status")
def get_status():
    return {
        "round": state.current_round,
        "round_in_progress": state.round_in_progress,
        "selected_clients": state.selected_clients,
        "clients_registered": len(state.clients),
        "stop_flag": state.stop_requested
    }

@app.post("/stop_training")
def stop_training():
    state.stop_requested = True
    state.round_in_progress = False
    print("\n[SERVER] Training stop requested. Devices will be notified.")
    return {"status": "stopped"}

# --- 4. Model Distribution & Aggregation ---
@app.get("/get_model")
def get_model(global_id: str):
    if global_id not in state.selected_clients:
        return JSONResponse(status_code=403, content={"error": "Client not selected for this round"})
        
    # If round 1 and no global model exists, send a dummy/initial model
    if not os.path.exists(state.global_model_path):
        model = models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        torch.save(model.state_dict(), state.global_model_path)
        
    with open(state.global_model_path, "rb") as f:
        model_bytes = f.read()
        
    return Response(content=model_bytes, media_type="application/octet-stream")

@app.post("/upload_model")
async def upload_model(global_id: str, hardware_speed_ms: float, file: UploadFile = File(...)):
    if global_id not in state.selected_clients:
        return JSONResponse(status_code=403, content={"error": "Client not selected"})
        
    try:
        model_bytes = await file.read()
    except Exception as e:
        # Catch ConnectionResetError: [WinError 10054] and network timeouts
        print(f"[SERVER] Client {global_id} dropped during upload: {e}")
        return JSONResponse(status_code=500, content={"error": "Client disconnected"})

    state.uploaded_models[global_id] = model_bytes
    
    # Update client's speed metric for next round's UCB
    state.clients[global_id]["hardware_speed_ms"] = hardware_speed_ms
    
    print(f"[{global_id}] Uploaded model weights. Time: {hardware_speed_ms:.2f}ms")
    
    # Reserve Clients Strategy: Proceed to FedAvg as soon as the first 'm' updates arrive
    target_m_clients = state.m
    if len(state.uploaded_models) >= target_m_clients:
        if not state.aggregating:
            state.aggregating = True
            threading.Thread(target=perform_fed_avg).start()
            
    return {"status": "success"}

def perform_fed_avg():
    # STRICT ROUND VALIDATION (Zero-Data Aggregation Prevention)
    target_m_clients = state.m
    successful_client_updates = list(state.uploaded_models.keys())
    
    if len(successful_client_updates) < target_m_clients:
        print(f"[ROUND VALIDATION] ABORT ROUND: Only {len(successful_client_updates)} updates received (needed {target_m_clients}).")
        # DO NOT increment current_round (it was preemptively incremented in start_round, so decrement it)
        state.current_round -= 1
        state.round_in_progress = False
        state.aggregating = False
        print(f"[SERVER] Restarting client selection phase for Round {state.current_round + 1}...")
        threading.Thread(target=auto_trigger_round).start()
        return

    print(f"[ROUND {state.current_round}] First {target_m_clients} clients uploaded. Discarding stragglers. Performing FedAvg...")
    
    # Load state dicts (limit to first 'm' models if we received more simultaneously)
    state_dicts = []
    for gid in successful_client_updates[:target_m_clients]:
        model_bytes = state.uploaded_models[gid]
        buffer = io.BytesIO(model_bytes)
        state_dict = torch.load(buffer, weights_only=True)
        state_dicts.append(state_dict)
        
    # Average them
    global_dict = state_dicts[0]
    for key in global_dict.keys():
        # Sum
        for i in range(1, len(state_dicts)):
            global_dict[key] += state_dicts[i][key]
        # Divide
        # For float/complex types use torch.div, for integer types use floor_divide
        if torch.is_floating_point(global_dict[key]):
            global_dict[key] = torch.div(global_dict[key], len(state_dicts))
        else:
            global_dict[key] = torch.div(global_dict[key], len(state_dicts), rounding_mode='trunc')
        
    # Save back
    torch.save(global_dict, state.global_model_path)
    print(f"[ROUND {state.current_round}] FedAvg Complete. Global model saved.")
    
    if testset is not None:
        print(f"[ROUND {state.current_round}] Evaluating global model on Test Set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.load_state_dict(global_dict)
        model.to(device)
        model.eval()
        
        testloader = DataLoader(testset, batch_size=128, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_samples += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = 100. * correct / total_samples if total_samples > 0 else 0.0
        
        state.round_metrics[state.current_round] = {"acc": accuracy, "loss": avg_loss}
        print(f"[ROUND {state.current_round}] Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")
        
        model.to('cpu')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    state.round_in_progress = False
    
    # Auto-trigger next round
    if state.current_round < state.max_rounds:
        print(f"[SERVER] Auto-starting Round {state.current_round + 1} in 3 seconds...")
        threading.Thread(target=auto_trigger_round).start()
    else:
        print(f"\n[SERVER] Maximum rounds ({state.max_rounds}) reached! Training complete.")

if __name__ == "__main__":
    print("=== FAB-FL Server (Laptop 1) ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)
