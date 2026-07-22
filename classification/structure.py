# %%
from torch import nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from accuracy import accuracy

# Make 1000 samples
n_samples = 1000
device = "xpu"
# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
x = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(type(x), type(y))
#test and train split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(len(x_train),type(x_train), len(y_train), len(x_test), len(y_test))
# class circleV1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(in_features=2, out_features=5)
#         self.layer2 = nn.Linear(in_features=5, out_features=1)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:#doing linear 2ice
#         return self.layer2(self.layer1(x))
    
# %%
# circle = circleV1().to("xpu")
# print(circle)
# %%
#using nn.sequential

model_0 = nn.Sequential(nn.Linear(in_features = 2,out_features=5),nn.Linear(in_features = 5,out_features= 1)).to("xpu")
print(model_0)
print(list(model_0.parameters()))
loss_fn = nn.BCEWithLogitsLoss()
model_0.eval()
with torch.inference_mode():
    Y_logits = model_0(x_test.to(device)[:5])
    print(Y_logits)
    Y_pred_prob = torch.sigmoid(Y_logits)
    print(Y_pred_prob,torch.round(Y_pred_prob))

