import numpy as np
from scipy.stats import uniform, randint

def random_search(model_class, x_train, y_train, param_distributions, n_iter=10, cv=3):
    best_score = float('-inf')
    best_params = None

    for _ in range(n_iter):
        params = {k: v.rvs() for k, v in param_distributions.items()}
        model = model_class(**params)
        scores = []

        for fold in range(cv):
            train_idx = np.random.choice(len(x_train), size=int(0.8*len(x_train)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(x_train)), train_idx)
            
            x_train_fold, y_train_fold = x_train[train_idx], y_train[train_idx]
            x_val_fold, y_val_fold = x_train[val_idx], y_train[val_idx]

            model.fit(x_train_fold, y_train_fold)
            score = model.evaluate(x_val_fold, y_val_fold)
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score

# Usage example:
param_distributions = {
    'learning_rate': uniform(0.0001, 0.1),
    'batch_size': randint(32, 256),
    'num_layers': randint(1, 5)
}

best_params, best_score = random_search(MyNeuralNetworkModel, x_train, y_train, param_distributions)