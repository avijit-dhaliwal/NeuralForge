# neuralforge/benchmark/performance.py
import time
import torch
import torch.nn as nn
import torchvision.models as models
from neuralforge.core.model import Model as NFModel

def benchmark_forward_pass(model, input_size, num_iterations=1000, device='cuda'):
    model.to(device)
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)
    
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

def compare_frameworks(input_size, num_classes):
    # PyTorch model
    pytorch_model = models.resnet50(pretrained=False, num_classes=num_classes)
    pytorch_time = benchmark_forward_pass(pytorch_model, input_size)
    
    # NeuralForge model (assuming similar architecture to ResNet50)
    nf_model = NFModel()  # Define your NeuralForge model here
    nf_time = benchmark_forward_pass(nf_model, input_size)
    
    print(f"PyTorch forward pass time: {pytorch_time:.6f} seconds")
    print(f"NeuralForge forward pass time: {nf_time:.6f} seconds")

# Usage example:
# compare_frameworks((3, 224, 224), num_classes=1000)