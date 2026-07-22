#%%
import torch
def accuracy(Y_true,Y_pred):
    correct = torch.eq(Y_true,Y_pred).sum().item()
    acc = (correct/len(Y_pred))*100
    return acc
# %%
