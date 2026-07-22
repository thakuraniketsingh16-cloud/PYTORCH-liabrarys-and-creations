from sklearn import datasets
import torch
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
print(X[:5], y[:5])

# Turn data into a dataframe
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print(circles.head())
import numpy as np

# 1. Plot the dataset points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')

# 2. Generate points for a perfect separating circle (Radius = 0.9)
theta = np.linspace(0, 2 * np.pi, 200)
radius = 0.9

circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)

# 3. Plot the decision boundary line
plt.plot(circle_x, circle_y, color="black", linestyle="--", linewidth=2, label="Decision Boundary")

# 4. Clean up the plot display
plt.gca().set_aspect('equal', adjustable='box') # Keeps the circle from looking like an oval
plt.legend()
plt.show()
#check dtyepe of X and y
# print(type(X), type(y))
# convert to tensors
x = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(type(x), type(y))
#test and train split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(len(x_train),type(x_train), len(y_train), len(x_test), len(y_test))
