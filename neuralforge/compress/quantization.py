# neuralforge/compress/quantization.py
import torch
import torch.nn as nn
import torch.quantization

def quantize_model(model, backend='fbgemm'):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model

def quantization_aware_training(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    qat_model = torch.quantization.prepare_qat(model)
    
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    qat_model = torch.quantization.convert(qat_model)
    return qat_model

# neuralforge/compress/pruning.py
import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# Usage example:
quantized_model = quantize_model(my_model)
pruned_model = prune_model(my_model, amount=0.3)