#%%
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
import accuracy 

NUM_CLASS = 4
NUM_FEATURE = 2
RANDOM_SEED = 42
device = "xpu" if torch.xpu.is_available() else "cpu" # Falls back safely if XPU isn't active

X_blob, Y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURE, 
                            centers=NUM_CLASS,
                            cluster_std=1.5,       
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(Y_blob).type(torch.LongTensor) # CrossEntropyLoss expects LongTensor for labels

x_train, x_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

X_blob_train, y_blob_train = x_train.to(device), y_train.to(device)
X_blob_test, y_blob_test = x_test.to(device), y_test.to(device)

class BlobModel(nn.Module):
    def __init__(self, input_feature, output_feature, hidden_units=8):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_feature)
        )
    def forward(self, x: torch.Tensor):
        return self.linear_stack(x)

model_4 = BlobModel(input_feature=NUM_FEATURE, 
                    output_feature=NUM_CLASS, 
                    hidden_units=8).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)


torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train) 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) 
    loss = loss_fn(y_logits, y_blob_train) 
    acc = accuracy.accuracy(Y_true=y_blob_train, 
                            Y_pred=y_pred) 

    optimizer.zero_grad()


    loss.backward()


    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy.accuracy(Y_true=y_blob_test, Y_pred=test_pred)


    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
        # Make predictions
#%%
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

print(y_logits[:10])
y_pred_probs = torch.softmax(y_logits, dim=1)

y_preds = y_pred_probs.argmax(dim=1)

print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy.accuracy(
    Y_true=y_blob_test, Y_pred=y_preds)}%")
# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()
# %%
