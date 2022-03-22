import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

def cluster_distances(embedding, labels):
    """ Get 10-th percentile of distances between classes in embedding """
    # Standardize clusters to [-0.5, 0.5]
    normed_embeddings = embedding - np.min(embedding)
    normed_embeddings /= np.max(normed_embeddings)
    normed_embeddings -= np.array([[0.5, 0.5]])

    num_classes = int(np.unique(labels).shape[0])
    cluster_distances = []
    for label_i in tqdm(range(0, num_classes - 1), total=num_classes-1):
        for label_j in range(label_i + 1, num_classes):
            embeddings_i = normed_embeddings[np.where(labels == label_i)]
            embeddings_j = normed_embeddings[np.where(labels == label_j)]
            vecs = np.expand_dims(embeddings_i, 0) - \
                   np.expand_dims(embeddings_i, 1)
            dists = np.sqrt(np.sum(np.square(vecs), axis=-1))
            dists = dists.flatten()
            dists = np.sort(dists)
            cluster_distances.append(dists[int(len(dists) / 10)])

    return cluster_distances, normed_embeddings

def cluster_quality(embedding, labels):
    num_classes = int(np.unique(labels).shape[0])
    model = KMeans(n_clusters=num_classes).fit(embedding)
    return v_measure_score(labels, model.labels_)

def classifier_accuracy(embedding, labels, k=100, cross_val_steps=10):
    """ cross-validated kNN classifier accuracy """
    model = KNeighborsClassifier(n_neighbors=k)
    slice_size = int(embedding.shape[0] / cross_val_steps)
    accuracies = np.zeros([cross_val_steps - 1])
    for i in range(cross_val_steps - 1):
        X_train = np.concatenate([embedding[:slice_size * i, :], embedding[slice_size * (i + 1):, :]])
        Y_train = np.concatenate([labels[:slice_size * i], labels[slice_size * (i + 1):]])
        X_test = embedding[slice_size * i : slice_size * (i + 1), :]
        Y_test = labels[slice_size * i : slice_size * (i + 1)]
        model.fit(X_train, Y_train)
        accuracies[i] = model.score(X_test, Y_test)

    return np.mean(accuracies)
