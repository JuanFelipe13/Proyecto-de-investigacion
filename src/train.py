import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import random
import os
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    IMG_SIZE = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    NUM_CLASSES = 101
    LEARNING_RATE = 0.001
    SAMPLES_PER_CLASS = 100

    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = ImageFolder(root='data/food-101/train', transform=train_transform)
    full_test_dataset = ImageFolder(root='data/food-101/test', transform=test_transform)

    indices_per_class = {}
    for idx, (_, label) in enumerate(full_train_dataset.samples):
        if label not in indices_per_class:
            indices_per_class[label] = []
        indices_per_class[label].append(idx)

    selected_indices = []
    for label in indices_per_class:
        selected_indices.extend(random.sample(indices_per_class[label], 
                                           min(SAMPLES_PER_CLASS, len(indices_per_class[label]))))

    train_dataset = Subset(full_train_dataset, selected_indices)
    test_dataset = Subset(full_test_dataset, range(min(1000, len(full_test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class FastFoodNet(nn.Module):
        def __init__(self):
            super(FastFoodNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, NUM_CLASSES)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = FastFoodNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    print("Iniciando entrenamiento rápido...")
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            print(f"Guardando modelo ({best_acc:.2f}% -> {val_acc:.2f}%)")
            best_acc = val_acc
            torch.save(model.state_dict(), 'food_recognition_model.pth')

    print(f"Entrenamiento rápido completado! Mejor precisión: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 