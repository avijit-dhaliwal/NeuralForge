# neuralforge/distributed/federated.py
import copy
import torch

def federated_average(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def train_client(model, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model

def federated_learning(global_model, client_data, num_rounds, local_epochs):
    for round in range(num_rounds):
        client_models = []
        for client_loader in client_data:
            client_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
            client_model = train_client(client_model, optimizer, client_loader, local_epochs)
            client_models.append(client_model)
        
        global_model = federated_average(global_model, client_models)
    
    return global_model

# Usage:
# global_model = YourModel()
# client_data = [dataloader1, dataloader2, dataloader3]  # Dataloaders for each client
# final_model = federated_learning(global_model, client_data, num_rounds=10, local_epochs=5)