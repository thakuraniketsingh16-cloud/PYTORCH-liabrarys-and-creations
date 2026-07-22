import torch 
x= torch.arange(0, 1,0.02).unsqueeze(dim=1)
w,b = 0.7, 0.3
y = w*x+b
training_set= int(0.8*len(x))

x_train, y_train = x[:training_set].to("xpu"),y[:training_set].to("xpu")
x_test, y_test = x[training_set:].to("xpu"), y[training_set:].to("xpu")

