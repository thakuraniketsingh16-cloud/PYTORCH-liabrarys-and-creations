import torch
import pathlib as path
import torch.nn as nn
from model_str import LinearRegressionModel
from data import x_train, y_train, x_test, y_test
from plots import plot_predictions
import matplotlib.pyplot as plt
torch.manual_seed(42)
model_0 = LinearRegressionModel().to("xpu")
y_pred = model_0(x_train)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
print(next(model_0.parameters()).device)
torch.manual_seed(42)
epoch_count, loss_vals, test_loss_vals = [], [], []
test_vals = []
epochs = 200
for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")
        print(f"weights: {list(model_0.parameters())[0].item():.4f}, bias: {list(model_0.parameters())[1].item():.4f}")
        epoch_count.append(epoch)
        loss_vals.append(loss.item())
    
        model_0.eval() 
        with torch.inference_mode():
            test_preds = model_0(x_test)
            test_vals.append(test_preds)
            test_loss = loss_fn(test_preds, y_test)
            test_loss_vals.append(test_loss.item())
                        
# Send all tensors back to CPU and cast them to numpy right as you pass them in!
with torch.inference_mode():
   # Strip away the incomplete loss line variables to focus purely on the data scatter plot
    plot_predictions(
        train_data=x_train.cpu().numpy(),
        train_labels=y_train.cpu().numpy(),
        test_data=x_test.cpu().numpy(),
        test_labels=y_test.cpu().numpy(),
        predictions=test_preds.cpu().numpy())
plt.plot(epoch_count,loss_vals,color="red")
plt.plot(epoch_count,test_loss_vals,color="blue")

plt.show()

path = "models/model_1.pth"
torch.save(obj=model_0.state_dict(), f=path)
model_1 = LinearRegressionModel().to("xpu")
model_1.load_state_dict(torch.load(f=path))
print(model_1.state_dict())
print(model_1.eval())
with torch.inference_mode():
    y_preds_new = model_1(x_test)
    y_preds = model_0(x_test)
    print(y_preds_new == y_preds)
