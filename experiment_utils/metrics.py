import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
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

def cluster_quality(embedding, labels, cluster_model='kmeans'):
    num_classes = int(np.unique(labels).shape[0])
    labels = np.squeeze(labels)
    if cluster_model == 'kmeans':
        print('Fitting kmeans...')
        model = KMeans(n_clusters=num_classes).fit(embedding)
        pred_labels = model.labels_
        true_labels = labels
    elif cluster_model == 'dbscan':
        print('Fitting dbscan...')
        model = DBSCAN().fit(embedding)
        pred_labels = model.labels_
        nonnegative_inds = np.where(pred_labels >= 0)
        pred_labels = pred_labels[nonnegative_inds]
        true_labels = labels[nonnegative_inds]
    elif cluster_model == 'spectral':
        print('Fitting spectral clustering...')
        model = SpectralClustering(n_clusters=num_classes).fit(embedding)
        pred_labels = model.labels_
        true_labels = labels
    else:
        raise ValueError('Unrecognized clustering algorithm %s' % cluster_model)

    return v_measure_score(true_labels, pred_labels)

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
