from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import numpy as np
import os

app = FastAPI()

# Configuración
IMG_SIZE = 224
NUM_CLASSES = 101

# Definir el modelo
class FoodNet(torch.nn.Module):
    def __init__(self):
        super(FoodNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 26 * 26, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Cargar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FoodNet().to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'food_recognition_model.pth')
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Lista de etiquetas
labels_path = os.path.join(os.path.dirname(__file__), '..', 'labels.txt')
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(None), request: Request = None, top_k: int = 5):
    try:
        if file:
            image_data = await file.read()
        else:
            image_data = await request.body()
        
        # Procesar imagen
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = transform(image).unsqueeze(0).to(device)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.topk(probabilities, k=top_k)
            
            # Preparar resultados
            predictions = []
            for i in range(top_k):
                predictions.append({
                    "class": labels[top_class[0][i].item()],
                    "confidence": float(top_prob[0][i].item())
                })
        
        return {
            "predictions": predictions
        }
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["src"]
    ) 