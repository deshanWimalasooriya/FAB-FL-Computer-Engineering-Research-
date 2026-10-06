import asyncio
from fastapi.testclient import TestClient
import Server
from Server import app, state
import torch
import io
import torchvision.models as models
import torch.nn as nn

Server.testset = None  # Mock testset to None to bypass evaluation!

client = TestClient(app)

def make_model_bytes():
    m = models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    buffer = io.BytesIO()
    torch.save(m.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

state.selected_clients = ["FAB-FL-0018", "FAB-FL-0019", "FAB-FL-0020"]

r1 = client.post("/upload_model", params={"global_id": "FAB-FL-0018", "hardware_speed_ms": 100}, files={"file": ("model.pt", make_model_bytes(), "application/octet-stream")})
print("Upload 1:", r1.status_code, r1.text)

r2 = client.post("/upload_model", params={"global_id": "FAB-FL-0019", "hardware_speed_ms": 100}, files={"file": ("model.pt", make_model_bytes(), "application/octet-stream")})
print("Upload 2:", r2.status_code, r2.text)

try:
    r3 = client.post("/upload_model", params={"global_id": "FAB-FL-0020", "hardware_speed_ms": 100}, files={"file": ("model.pt", make_model_bytes(), "application/octet-stream")})
    print("Upload 3:", r3.status_code, r3.text)
except Exception as e:
    import traceback
    traceback.print_exc()
