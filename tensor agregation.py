import torch
#tensor agregation
x = torch.arange(0,100,5)
print(x,x.dtype)
#min
print(x.min(),torch.min(x))
#max
print(x.max(),torch.max(x))
#mean - require a tensor of float32 or lower
print(x.type(torch.float32).mean(),torch.mean(x.type(torch.float32)))
#sum
print(x.sum(),torch.sum(x))
#position of minimum and maximum value
y = torch.argmin(x)
z = torch.argmax(x)  
print(f"Minimum value is at index {y} and maximum value is at index {z}")
print(f"Minimum value is {x[y]} and maximum value is {x[z]}")