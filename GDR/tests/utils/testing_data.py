import numpy as np

def load_fake_data(dim=10, num_points=100, num_classes=10, seed=12345):
    np.random.seed(seed)
    points = np.random.multivariate_normal(np.zeros([dim]), np.eye(dim), size=num_points)
    labels = np.random.choice(10, size=num_points, replace=True)
    return points, labels

def load_fake_clusters(dim=10, num_points=100, num_classes=2, seed=12345):
    np.random.seed(seed)
    points_per_class = int(num_points / num_classes)
    all_points = []
    all_labels = []
    for c in range(num_classes):
        class_points = np.random.multivariate_normal(
            np.ones([dim]) * c,
            np.eye(dim) / 10,
            size=points_per_class
        )
        class_labels = np.ones([points_per_class]) * c
        all_points.append(class_points)
        all_labels.append(class_labels)

    points = np.concatenate(all_points)
    labels = np.concatenate(all_labels)

    return points, labels
