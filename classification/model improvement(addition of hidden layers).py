#%%
from torch import nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from accuracy import accuracy
import matplotlib.pyplot as plt
n_samples = 1000
device = "xpu"
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
x = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train,y_train = x_train.to(device) , y_train.to(device)
x_test,y_test = x_test.to(device) , y_test.to(device)
#%%
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x): 
    
        return self.layer_3(self.layer_2(self.layer_1(x)))
#%%
model_1 = CircleModelV1().to(device)
model_1
loss_fn = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
torch.manual_seed(42)
#%%
epochs = 1000 
#%%
X_train, y_train = x_train.to(device), y_train.to(device)
X_test, y_test = x_test.to(device), y_test.to(device)

for epoch in range(epochs):
    
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

    loss = loss_fn(y_logits, y_train)
    acc = accuracy(Y_true=y_train, 
                      Y_pred=y_pred)


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy(Y_true=y_test,
                               Y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
#%%        plt.figure(figsize=(12, 6))
import requests
from pathlib import Path 

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
#%%
from helper_functions import plot_predictions, plot_decision_boundary

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
# %%
