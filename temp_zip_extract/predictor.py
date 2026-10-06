import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
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
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(probabilities, 0)
            
        probs_dict = {CIFAR10_CLASSES[i]: probabilities[i].item() for i in range(10)}
        return {"class": CIFAR10_CLASSES[predicted.item()], "probabilities": probs_dict}
    except Exception as e:
        return f"Prediction Error: {str(e)}"

def predict_multiple_and_save(model_path, image_paths, save_dir):
    """
    Predicts multiple images and saves each original image with its prediction text overlay.
    Returns a list of dictionaries containing paths and results.
    """
    if not os.path.exists(model_path):
        return [{"error": "Error: Model not found. Start training first."}]
        
    try:
        model, device = get_model(model_path)
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        results = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                results.append({"path": img_path, "error": "Image not found."})
                continue
                
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(probabilities, 0)
                
            best_class = CIFAR10_CLASSES[predicted.item()]
            prob = probabilities[predicted.item()].item()
            
            # Create a composite image
            # We'll add a white panel at the bottom for text
            padded_img = Image.new('RGB', (img.width, img.height + 60), color='white')
            padded_img.paste(img, (0, 0))
            
            draw = ImageDraw.Draw(padded_img)
            text = f"Pred: {best_class.upper()}\nConf: {prob*100:.1f}%"
            # Try to load a default font, otherwise default to basic
            try:
                # Use a larger font if possible
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
                
            draw.text((10, img.height + 10), text, fill="black", font=font)
            
            save_name = f"pred_{os.path.basename(img_path)}"
            save_path = os.path.join(save_dir, save_name)
            padded_img.save(save_path)
            
            results.append({
                "path": img_path,
                "saved_path": save_path,
                "class": best_class,
                "confidence": prob
            })
            
        return results
    except Exception as e:
        return [{"error": f"Prediction Error: {str(e)}"}]

