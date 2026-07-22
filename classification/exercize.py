import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn
from accuracy import accuracy  # Assuming this takes Y_true and Y_pred
from helper_functions import plot_decision_boundary 
import matplotlib.pyplot as plt# Assuming this takes a model and data
random_state = 42

# 1. Generate and split data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 2. Convert cleanly to Tensors on your hardware target
x_train = torch.from_numpy(x_train).type(torch.float).to("xpu")
x_test  = torch.from_numpy(x_test).type(torch.float).to("xpu")
y_train = torch.from_numpy(y_train).type(torch.float).to("xpu")
y_test  = torch.from_numpy(y_test).type(torch.float).to("xpu")

# 3. Define structure
class MakeMoonsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

# INSTANTIATE ONCE
model = MakeMoonsModel().to("xpu")
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)

# 4. Training Loop
epochs = 1000
for epoch in range(epochs):
    # Set model to training mode
    model.train()

    # 1. Forward pass using the assigned variable
    y_logits = model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) 

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy(Y_true=y_train, Y_pred=y_pred)

    # 3. Optimization pipeline
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. Evaluation Loop Block
    if epoch % 100 == 0:
        model.eval() # Use instance variable here
        with torch.inference_mode():
            test_logits = model(x_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy(Y_true=y_test, Y_pred=test_pred)
            
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

plot_decision_boundary(model=model, X=x_test, y=y_test)
plt.show()