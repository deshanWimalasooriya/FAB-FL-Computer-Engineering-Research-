import asyncio
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FAB-FL Server API")

# Allow CORS for React frontend (Vite default is 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State to Simulate Training
training_active = False

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.post("/api/start")
async def start_training():
    global training_active
    if not training_active:
        training_active = True
        # Start a background task for the dummy training loop
        asyncio.create_task(training_loop())
    return {"status": "Training started"}

@app.post("/api/stop")
async def stop_training():
    global training_active
    training_active = False
    return {"status": "Training stopped"}

@app.post("/api/restart")
async def restart_training():
    global training_active
    training_active = False
    await asyncio.sleep(1) # Allow current loop to exit
    training_active = True
    asyncio.create_task(training_loop())
    return {"status": "Training restarted"}

async def training_loop():
    """
    Simulates the FAB-FL Server Training Loop.
    Generates dummy metrics and broadcasts via WebSockets.
    """
    global training_active
    round_num = 1
    total_rounds = 100
    
    while training_active and round_num <= total_rounds:
        # Simulate time taken for one round
        await asyncio.sleep(2)
        
        # Generate dummy metrics
        metrics = {
            "type": "metrics",
            "round": round_num,
            "progress": int((round_num / total_rounds) * 100),
            "nodes": [
                {
                    "id": "Node-01",
                    "ip": "192.168.1.10",
                    "hardware": "NVIDIA Jetson Orin",
                    "clients": random.randint(10, 20),
                    "status": "Online",
                    "data_skew": round(random.uniform(0.1, 0.9), 2),
                    "latency": f"{random.randint(20, 150)}ms",
                    "ucb_score": round(random.uniform(0.5, 1.5), 3),
                    "network_speed": f"{random.randint(100, 500)} Mbps"
                },
                {
                    "id": "Node-02",
                    "ip": "192.168.1.11",
                    "hardware": "NVIDIA Jetson Nano",
                    "clients": random.randint(5, 15),
                    "status": "Online",
                    "data_skew": round(random.uniform(0.1, 0.9), 2),
                    "latency": f"{random.randint(40, 200)}ms",
                    "ucb_score": round(random.uniform(0.5, 1.5), 3),
                    "network_speed": f"{random.randint(50, 300)} Mbps"
                },
                {
                    "id": "Node-03",
                    "ip": "192.168.1.12",
                    "hardware": "Raspberry Pi 4",
                    "clients": random.randint(2, 8),
                    "status": "Offline",
                    "data_skew": 0.0,
                    "latency": "N/A",
                    "ucb_score": 0.0,
                    "network_speed": "0 Mbps"
                }
            ],
            "global_accuracy": round(0.5 + (round_num * 0.004) + random.uniform(-0.02, 0.02), 4),
            "global_loss": round(2.0 - (round_num * 0.015) + random.uniform(-0.05, 0.05), 4)
        }
        
        await manager.broadcast(metrics)
        round_num += 1

    if round_num > total_rounds:
        training_active = False
        await manager.broadcast({"type": "status", "message": "Training Complete"})

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We can receive messages from the client if needed (e.g., config updates)
            data = await websocket.receive_text()
            print(f"Received from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
