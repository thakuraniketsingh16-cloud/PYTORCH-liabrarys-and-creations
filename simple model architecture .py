
import torch
from torch import nn
import matplotlib.pyplot as plt


w = 0.7
b = 0.3
x = torch.arange(0, 100, 1).unsqueeze(dim=1)
y = w * x + b

train_size = int(0.8 * len(x))
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias



if __name__ == "__main__":
    print(x[:5], y[:5]) # Move this inside!
    
    torch.manual_seed(42)
    model_0 = LinearRegressionModel()
    
    with torch.inference_mode():
        y_preds = model_0(x_test)
        print(y_preds)
        
    plt.scatter(x_train, y_train, c="b", s=4, label="Training data")
    plt.scatter(x_test, y_test, c="g", s=4, label="Testing data")
    plt.scatter(x_test, y_preds, c="r", s=4, label="Predictions")
    plt.legend()
    plt.show() # This won't trigger anymore!