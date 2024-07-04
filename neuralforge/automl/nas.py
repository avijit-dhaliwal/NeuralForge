# neuralforge/automl/nas.py
import random
import torch
import torch.nn as nn

class SearchSpace:
    def __init__(self):
        self.ops = ['conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3']
        self.num_layers = [2, 3, 4, 5]

def random_architecture(search_space):
    num_layers = random.choice(search_space.num_layers)
    return [random.choice(search_space.ops) for _ in range(num_layers)]

def mutate_architecture(arch, search_space):
    mutation_type = random.choice(['add', 'remove', 'change'])
    if mutation_type == 'add' and len(arch) < max(search_space.num_layers):
        arch.append(random.choice(search_space.ops))
    elif mutation_type == 'remove' and len(arch) > min(search_space.num_layers):
        arch.pop(random.randint(0, len(arch) - 1))
    elif mutation_type == 'change':
        idx = random.randint(0, len(arch) - 1)
        arch[idx] = random.choice(search_space.ops)
    return arch

def create_model(arch, input_channels, num_classes):
    layers = []
    in_channels = input_channels
    for op in arch:
        if op == 'conv3x3':
            layers.append(nn.Conv2d(in_channels, in_channels*2, 3, padding=1))
            in_channels *= 2
        elif op == 'conv5x5':
            layers.append(nn.Conv2d(in_channels, in_channels*2, 5, padding=2))
            in_channels *= 2
        elif op == 'maxpool3x3':
            layers.append(nn.MaxPool2d(3, stride=1, padding=1))
        elif op == 'avgpool3x3':
            layers.append(nn.AvgPool2d(3, stride=1, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_channels, num_classes))
    return nn.Sequential(*layers)

def evolutionary_search(search_space, num_generations, population_size, input_channels, num_classes, train_fn):
    population = [random_architecture(search_space) for _ in range(population_size)]
    for generation in range(num_generations):
        models = [create_model(arch, input_channels, num_classes) for arch in population]
        fitness = [train_fn(model) for model in models]
        
        # Select top performers
        elite = [population[i] for i in sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:2]]
        
        # Create new population
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent = random.choice(elite)
            child = mutate_architecture(parent.copy(), search_space)
            new_population.append(child)
        
        population = new_population
    
    best_arch = max(zip(population, fitness), key=lambda x: x[1])[0]
    return best_arch

# Usage:
# def train_and_evaluate(model):
#     # Train the model and return validation accuracy
#     pass
# 
# search_space = SearchSpace()
# best_architecture = evolutionary_search(search_space, num_generations=10, population_size=20, input_channels=3, num_classes=10, train_fn=train_and_evaluate)
# best_model = create_model(best_architecture, input_channels=3, num_classes=10)