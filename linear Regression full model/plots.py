import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None, epochs=None, loss_vals=None, test_loss_vals=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    # This block safely skips line plotting if we don't pass epochs or loss metrics
    if epochs is not None and loss_vals is not None and test_loss_vals is not None:
        plt.plot(epochs, loss_vals, c="b", label="Training Loss")
        plt.plot(epochs, test_loss_vals, c="g", label="Testing Loss")
        
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend()
    plt.show()