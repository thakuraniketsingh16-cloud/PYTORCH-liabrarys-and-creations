# %%
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
# %%
model_0 = nn.Sequential(nn.Linear(in_features = 2,out_features=5),nn.Linear(in_features = 5,out_features= 1)).to("xpu")
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(lr=0.1,params = model_0.parameters())
## training time baby
#%%
torch.xpu.manual_seed(42)
epoch = 200
for epoch in range (epoch):
    model_0.train() # set the model to train mode (not inference)
    y_logits = model_0 (x_train).squeeze()
    train_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn (y_logits, y_train)
    train_acc = accuracy(Y_pred=train_preds, Y_true=y_train)
     
    optimizer.zero_grad()
    loss.backward() 
    #testing
    model_0.eval()
    with torch.inference_mode():
        y_test_logit = model_0 (x_test).squeeze()
        preds = torch.round(torch.sigmoid(y_test_logit))
        test_loss = loss_fn(y_test_logit, y_test)
        test_acc = accuracy(Y_pred=preds, Y_true=y_test)
        
        
        
        if epoch % 10:
           print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
#%%
import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary



# %%
plt.figure(figsize=(7, 3))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, x_train, y_train)
plt.figure(figsize=(7, 3))
plt.subplot(1,2,1)
plt.title("TEST")
plot_decision_boundary(model_0, x_test, y_test)
# %%
