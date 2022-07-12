import numpy as np

def load_fake_data(dim=10, num_points=100, num_classes=10, seed=12345):
    np.random.seed(seed)
    points = np.random.multivariate_normal(np.zeros([dim]), np.eye(dim), size=num_points)
    labels = np.random.choice(10, size=num_points, replace=True)
    return points, labels
