#%%
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

NUM_CLASS = 4
NUM_FEATURE = 2
RANDOM_SEED = 42

X_blob,Y_blob = make_blobs(n_samples = 1000,
                           n_features = NUM_FEATURE, 
                           centers = NUM_CLASS,
                           cluster_std = 1.5,       
                           random_state = RANDOM_SEED)
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(Y_blob).type(torch.float)
print(X_blob[:5], y_blob[:5])
                           
x_train,y_train,x_test,y_test = train_test_split(X_blob, Y_blob, test_size = 0.2, random_state = RANDOM_SEED)
# plt.figure(figsize = (10,7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1],c = Y_blob,cmap = plt.cm.RdYlBu)
# plt.show()
# %%
