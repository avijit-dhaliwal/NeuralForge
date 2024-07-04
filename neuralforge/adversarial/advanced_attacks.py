# neuralforge/adversarial/advanced_attacks.py
import torch
import torch.nn as nn

def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    loss = nn.CrossEntropyLoss()
    
    ori_images = images.data
    
    for i in range(num_iter):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
    return images

def carlini_wagner_l2(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
    def f(x):
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(x.device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        if targeted:
            return torch.clamp(i-j, min=-kappa)
        else:
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    
    for step in range(max_iter):
        a = 1/2*(torch.tanh(w) + 1)
        
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))
        loss = loss1 + loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    attack_images = 1/2*(torch.tanh(w) + 1)
    return attack_images

# Usage:
# model = YourModel()
# images, labels = next(iter(dataloader))
# adversarial_images = pgd_attack(model, images, labels, epsilon=0.3, alpha=2/255, num_iter=40)
# or
# adversarial_images = carlini_wagner_l2(model, images, labels)