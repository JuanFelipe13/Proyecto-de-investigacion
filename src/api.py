from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import numpy as np
import os

app = FastAPI()

IMG_SIZE = 128
NUM_CLASSES = 101

class FoodNet(torch.nn.Module):
    def __init__(self):
        super(FoodNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 16 * 16, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FoodNet().to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'food_recognition_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

labels_path = os.path.join(os.path.dirname(__file__), '..', 'labels.txt')
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

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

        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
 
        with torch.no_grad():
            outputs = model(image_tensor)
            print("Outputs:", outputs) 
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print("Probabilities:", probabilities) 
            
            top_prob, top_class = torch.topk(probabilities, k=min(top_k, len(labels)))
            print("Top classes:", top_class) 
            print("Top probabilities:", top_prob)  

            predictions = []
            for i in range(min(top_k, len(labels))):
                class_idx = top_class[0][i].item()
                if class_idx < len(labels):
                    predictions.append({
                        "class": labels[class_idx],
                        "confidence": float(top_prob[0][i].item())
                    })
        
        return {
            "predictions": predictions
        }
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
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