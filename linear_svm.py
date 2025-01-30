import matplotlib.pyplot as plt
import numpy as np

data = np.load(f"../data/toy-data.npz")  # Make sure you haved cd'ed into scripts for this relative path to work
training_data = data["training_data"]
labels = data["training_labels"]
w = np.array([-0.4528, -0.5190])         # Weight vector
b = 0.1471                               # Bias

# Plot the data points
def plot_data_points(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels)

# Plot the decision boundary
def plot_decision_boundary(w, b):
    x = np.linspace(-5, 5, 100)
    y = -(w[0] * x + b) / w[1]
    plt.plot(x, y, 'k')

# Plot the margins
def plot_margins(w, b):
    x = np.linspace(-5, 5, 100)
    y_positive_margin = -(w[0] * x + b - 1) / w[1]
    y_negative_margin = -(w[0] * x + b + 1) / w[1]
    plt.plot(x, y_positive_margin, 'r--')
    plt.plot(x, y_negative_margin, 'r--')

# Indicate the support vectors
def plot_support_vectors(data, w, b):
    decision_function = np.dot(data, w) + b
    support_vector_indices = np.where(np.abs(decision_function) - 1 < 1e-2)[0] # Support vectors are at the margin
    plt.scatter(data[support_vector_indices, 0], data[support_vector_indices, 1],
                edgecolor='k', s=100, marker='o', label="Support Vectors", alpha=0.3)


# Plot the data and results
plt.figure(figsize=(6, 6))
plot_data_points(training_data, labels)
plot_decision_boundary(w, b)
plot_margins(w, b)
plot_support_vectors(training_data, w, b)
plt.title("Hard-Margin SVM Visualization")
plt.legend()
plt.show()
