import torch
#before 
random_a = torch.rand(3,4)
random_b = torch.rand(3,4)
print(random_a)
print(random_b)
print(random_a == random_b)
#after
torch.manual_seed(42)    
random_a = torch.rand(3,4)
torch.manual_seed(42)
random_b = torch.rand(3,4)
print(random_a)
print(random_b) 
print(random_a == random_b)
