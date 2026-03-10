import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
"""from IPython import display
display.set_matplotlib_formats("svg")""" 

iris = pd.read_csv("iris.csv")
print(iris.head())

X = torch.tensor(iris.drop("variety", axis=1).values, dtype=torch.float)
y = torch.tensor(
    [0 if vty == "Setosa" else 1 if vty == "Versicolor" else 2 for vty in iris["variety"]], 
    dtype=torch.long
)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, shuffle=True, batch_size=12)
test_loader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))

print("Training data batches:")
for X, y in train_loader:
    print(X.shape, y.shape)
    
print("\nTest data batches:")
for X, y in test_loader:
    print(X.shape, y.shape)


    
def train_model(train_loader, test_loader, model, lr=0.01, num_epochs=200):
    train_accuracies, test_accuracies = [], []
    losses = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        correct_preds=0
        total_preds=0

        for X, y in train_loader:
            preds = model(X)
            pred_labels = torch.argmax(preds, axis=1)
            loss = loss_function(preds, y)
            losses.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_preds += (pred_labels == y).sum().item()
            total_preds += y.size(0)
            
        train_accuracies.append(
            100 * correct_preds / total_preds
        )
        
        X, y = next(iter(test_loader))
        pred_labels = torch.argmax(model(X), axis=1)
        test_accuracies.append(
            100 * torch.mean((pred_labels == y).float()).item()
        )
 
    return train_accuracies[-1], test_accuracies[-1]


class Net2(nn.Module):
    def __init__(self, n_units, n_layers):
        super().__init__()
        self.n_layers = n_layers
        
        self.layers = nn.ModuleDict()
        self.layers["input"] = nn.Linear(in_features=4, out_features=n_units)
        
        for i in range(self.n_layers):
            self.layers[f"hidden_{i}"] = nn.Linear(in_features=n_units, out_features=n_units)
            
        self.layers["output"] = nn.Linear(in_features=n_units, out_features=3)
        
    def forward(self, x):
        x = self.layers["input"](x)
        
        for i in range(self.n_layers):
            x = F.relu(self.layers[f"hidden_{i}"](x))
            
        return self.layers["output"](x)
    
n_layers = np.arange(1, 5)
n_units = np.arange(8, 65, 8)
train_accuracies, test_accuracies = [], []

for i in range(len(n_units)):
    for j in range(len(n_layers)):
        model = Net2(n_units=n_units[i], n_layers=n_layers[j])
        train_acc, test_acc = train_model(train_loader, test_loader, model)
        train_accuracies.append({
            "n_layers": n_layers[j],
            "n_units": n_units[i],
            "accuracy": train_acc
        })
        test_accuracies.append({
            "n_layers": n_layers[j],
            "n_units": n_units[i],
            "accuracy": test_acc
        })
        
        
train_accuracies = pd.DataFrame(train_accuracies).sort_values(by=["n_layers", "n_units"]).reset_index(drop=True)
test_accuracies = pd.DataFrame(test_accuracies).sort_values(by=["n_layers", "n_units"]).reset_index(drop=True)
print(test_accuracies.head())    

print(test_accuracies[test_accuracies["accuracy"] == test_accuracies["accuracy"].max()])