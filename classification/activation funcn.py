import torch
import matplotlib.pyplot as plt
A = torch.arange(0-10,10,1,dtype=torch.float32)
print(A)
def Relu(x:torch.tensor):
    return torch.maximum(torch.tensor(0),x)
def sigmoid(x:torch.tensor):
    return 1/(1+torch.exp(-x))
plt.plot(sigmoid(A))
plt.plot(Relu(A))
plt.show()