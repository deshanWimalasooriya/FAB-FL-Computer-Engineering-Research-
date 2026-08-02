import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def get_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(num_classes=10)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def predict_image(model_path, image_path):
    if not os.path.exists(image_path):
        return "Error: Image not found."
    if not os.path.exists(model_path):
        return "Error: Model not found. Start training first."
        
    try:
        model, device = get_model(model_path)
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        return CIFAR10_CLASSES[predicted.item()]
    except Exception as e:
        return f"Prediction Error: {str(e)}"
