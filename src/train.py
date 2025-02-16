import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from tqdm import tqdm

def main():
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuración
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    NUM_CLASSES = 101
    LEARNING_RATE = 0.001

    # Cargar etiquetas
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Transformaciones de datos
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar datasets
    train_dataset = ImageFolder(root='data/food-101/train', transform=train_transform)
    test_dataset = ImageFolder(root='data/food-101/test', transform=test_transform)

    # Usar num_workers=0 para evitar problemas de multiprocesamiento
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Modelo CNN
    class FoodNet(nn.Module):
        def __init__(self):
            super(FoodNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 26 * 26, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, NUM_CLASSES)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # Crear modelo y moverlo a GPU
    model = FoodNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Función de entrenamiento
    def train_epoch(model, loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(loader, desc='Training')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        return running_loss/len(loader), 100.*correct/total

    # Función de evaluación
    def evaluate(model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc='Evaluating'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss/len(loader), 100.*correct/total

    # Entrenamiento principal
    print("Iniciando entrenamiento...")
    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Guardar el mejor modelo
        if test_acc > best_acc:
            print(f"Accuracy mejorada ({best_acc:.2f}% -> {test_acc:.2f}%). Guardando modelo...")
            best_acc = test_acc
            torch.save(model.state_dict(), 'food_recognition_model.pth')

    print("Entrenamiento completado!")

if __name__ == '__main__':
    main() 