import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from scipy import linalg
from sklearn import cluster, mixture

COLORS = ['red', 'navy', 'c', 'cornflowerblue', 'gold', 'darkorange',
          'magenta', 'yellow', 'green', 'blue', 'skyblue', 'forestgreen',
          'purple', 'darkmagenta', 'crimson', 'sienna', 'lime', 'brown',
          'darkviolet', 'maroon']
COLORS_ITER = itertools.cycle(COLORS)


def fit(method, X, n_clusters, samples_by_cluster, max_iter):
    if X.shape[0] <= samples_by_cluster * n_clusters:
        n_clusters = int(X.shape[0] / samples_by_cluster)

    if X.shape[0] < n_clusters or n_clusters == 0:
        n_clusters = 1

    if method == "kmeans":
        model = cluster.KMeans(n_clusters=n_clusters, max_iter=max_iter)
    elif method == "kmeans":
        model = mixture.GaussianMixture(n_components=n_clusters,
                                        max_iter=max_iter)
    elif method == "bgmm":
        model = mixture.BayesianGaussianMixture(
            n_components=n_clusters, max_iter=max_iter)
    else:
        model = cluster.Birch(n_clusters=n_clusters, compute_labels=False)

    while True:
        try:
            model.fit(X)
            return model
        except Exception as e:
            if type(model) == cluster.birch.Birch:
                model = cluster.Birch(n_clusters=n_clusters,
                                      compute_labels=False)
                if n_clusters > 1:
                    n_clusters -= 1
                continue
            else:
                raise(e)
    return model


def predict_proba(model, X):
    if type(model) == mixture.bayesian_mixture.BayesianGaussianMixture or \
            type(model) == mixture.gaussian_mixture.GaussianMixture:
        probs = model.predict_proba(X)
    else:
        probs = model.transform(X)
        probs = 1.0 - probs / np.max(probs, axis=1, keepdims=True) + 0.0000001
        probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs


def plot_results(X, Y_, clusters, title):
    for index, color in zip(range(clusters), COLORS_ITER):
        i = index + 1
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    if np.any(Y_ == 0):
        plt.scatter(X[Y_ == 0, 0], X[Y_ == 0, 1], .8, color="gray")
    if np.any(Y_ == -1):
        plt.scatter(X[Y_ == -1, 0], X[Y_ == -1, 1], .8, color="black")

    plt.xticks(()), plt.yticks(())
    plt.title(title)
    plt.show()


def compute_score(probs, c):
    score = probs[c] * (1.0 - np.sum(probs) + probs[c])
    return score


def generate_random_samples(n_samples = 10000, gaussian_distribution = False):
    np.random.seed(2)
    samples, components, variance, bias = [250, int(n_samples / 250), 2, 5]
    X = []

    if gaussian_distribution:
        C = [[1, 0], [0, 1]]

    for _i in range(components):
        if not gaussian_distribution:
            C = np.random.uniform(-variance, variance, size=(2, 2))
        b = np.random.uniform(-bias, bias, size=(2))
        X.append(np.dot(np.random.randn(samples, 2), C) + b)
    X = np.array(X).reshape(-1, 2)
    return X


if __name__ == "__main__":
    # Model parameters.
    n_clusters = 5
    samples_by_cluster = 250
    max_iter = 200
    method = "kmeans"

    # Generate samples.
    X = X_filtered = generate_random_samples(10000, False)
    y = np.zeros(len(X), dtype=np.int)
    y_map = range(len(X))
    scores = [[] for i in range(len(X))]
    clusters = []

    while X_filtered.shape[0] > 0:
        model = fit(method, X_filtered, n_clusters, samples_by_cluster,
                    max_iter)
        probs_all = predict_proba(model, X)
        probs_filtered = np.array(
            [probs_all[y_map[i]] for i in range(len(X_filtered))])

        for c in range(min(probs_filtered.shape[1], n_clusters)):
            cluster_id = c + 1
            cluster_indexes_filtered = np.argsort(
                probs_filtered[:, c])[-samples_by_cluster:]
            cluster_indexes_all = np.argsort(
                probs_all[:, c])[-samples_by_cluster:]
            cluster_set = set()

            for i, j in zip(cluster_indexes_filtered, cluster_indexes_all):
                y[y_map[i]], y[j] = (cluster_id, cluster_id)
                cluster_set.add(y_map[i]), cluster_set.add(j)

            for i in cluster_indexes_all[len(cluster_indexes_filtered):]:
                y[i] = cluster_id
                cluster_set.add(i)

            for i in range(len(probs_all)):
                if y[i] == cluster_id:
                    score = compute_score(probs_all[i], c)
                else:
                    score = 0
                scores[i].append(score)
            clusters.append(cluster_set)

        # Plot result.
        indexes_filtered = np.where(np.equal(y, 0))
        remaining = len(indexes_filtered[0])
        processed = len(X) - remaining
        elements = np.sum([len(clusters[i]) for i in range(len(clusters))])
        status = 'Processed: %d, remaining: %d, clusters: %d, elements: %d' % (
            processed, remaining, len(clusters), elements)
        print(status)
        # plot_results(X, y, n_clusters, status)

        # Update data.
        y[np.where(np.greater(y, 0))] = -1
        X_filtered = X[indexes_filtered]
        y_map = [i for i in indexes_filtered[0]]

    # Plot result.
    scores = np.array(scores)
    scores = scores / (np.sum(scores, axis=0, keepdims=True))
    y = np.argmax(scores, axis=1) + 1
    n_clusters = len(np.unique(y))
    status = "Clusters: %d, Samples: %d" % (n_clusters, len(X))
    plot_results(X, y, scores.shape[1], status)

    # Plot histogram.
    n, bins, patches = plt.hist(y, n_clusters, facecolor='green', alpha=0.75)
    print("Size mean: %.2f, Size Std: %.2f" % (np.mean(n), np.std(n)))
    plt.xlabel('Cluster'), plt.ylabel('Size')
    plt.grid(True)
    plt.show()
