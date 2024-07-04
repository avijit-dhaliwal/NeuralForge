# neuralforge/transfer/transfer_learning.py
import torch
import torch.nn as nn
from torchvision import models

def create_transfer_model(base_model_name, num_classes, freeze_base=True):
    if base_model_name == 'resnet50':
        base_model = models.resnet50(pretrained=True)
    elif base_model_name == 'vgg16':
        base_model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    if freeze_base:
        for param in base_model.parameters():
            param.requires_grad = False
    
    if base_model_name.startswith('resnet'):
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, num_classes)
    elif base_model_name.startswith('vgg'):
        num_ftrs = base_model.classifier[6].in_features
        base_model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    return base_model

def fine_tune(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")

# Usage example:
model = create_transfer_model('resnet50', num_classes=10, freeze_base=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
fine_tune(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=torch.device('cuda'))