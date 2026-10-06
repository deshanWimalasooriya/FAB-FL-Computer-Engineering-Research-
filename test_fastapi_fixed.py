import asyncio
from fastapi.testclient import TestClient
import Server
from Server import app, state
import torch
import io
import torchvision.models as models
import torch.nn as nn

Server.testset = None

client = TestClient(app)

def make_model_bytes():
    m = models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    buffer = io.BytesIO()
    torch.save(m.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

r1_reg = client.post("/register", json={"temp_id": "FFC-0018", "class_counts": {0:1}, "hardware_speed_ms": 100}).json()
r2_reg = client.post("/register", json={"temp_id": "FFC-0019", "class_counts": {0:1}, "hardware_speed_ms": 100}).json()
r3_reg = client.post("/register", json={"temp_id": "FFC-0020", "class_counts": {0:1}, "hardware_speed_ms": 100}).json()

state.selected_clients = [r1_reg["global_id"], r2_reg["global_id"], r3_reg["global_id"]]

gids = list(state.selected_clients)

r1 = client.post("/upload_model", params={"global_id": gids[0], "hardware_speed_ms": 100}, files={"file": ("model.pt", make_model_bytes(), "application/octet-stream")})
print("Upload 1:", r1.status_code, r1.text)

r2 = client.post("/upload_model", params={"global_id": gids[1], "hardware_speed_ms": 100}, files={"file": ("model.pt", make_model_bytes(), "application/octet-stream")})
print("Upload 2:", r2.status_code, r2.text)

try:
    r3 = client.post("/upload_model", params={"global_id": gids[2], "hardware_speed_ms": 100}, files={"file": ("model.pt", make_model_bytes(), "application/octet-stream")})
    print("Upload 3:", r3.status_code, r3.text)
except Exception as e:
    import traceback
    traceback.print_exc()
