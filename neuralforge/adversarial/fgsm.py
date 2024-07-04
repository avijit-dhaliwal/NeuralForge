# neuralforge/adversarial/fgsm.py
import numpy as np

def fgsm_attack(model, x, y, epsilon):
    x_adv = x.copy()
    x_adv.requires_grad = True
    
    output = model(x_adv)
    loss = model.loss_fn(output, y)
    model.zero_grad()
    loss.backward()
    
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv

def adversarial_train_step(model, x, y, optimizer, epsilon):
    model.train()
    
    # Generate adversarial examples
    x_adv = fgsm_attack(model, x, y, epsilon)
    
    # Train on both clean and adversarial examples
    optimizer.zero_grad()
    output_clean = model(x)
    output_adv = model(x_adv)
    loss_clean = model.loss_fn(output_clean, y)
    loss_adv = model.loss_fn(output_adv, y)
    loss = 0.5 * (loss_clean + loss_adv)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Usage example:
for epoch in range(num_epochs):
    for batch_x, batch_y in data_loader:
        loss = adversarial_train_step(model, batch_x, batch_y, optimizer, epsilon=0.01)
    print(f"Epoch {epoch}, Loss: {loss}")