import numpy as np




def get_normal_dist(
        datax: np.ndarray,
        datay: np.ndarray,
        label: int,
        feature: int) -> tuple:

    mu = np.mean(datax[datay == label, feature]) # Mean
    var = np.var(datax[datay == label, feature]) # Variance
    pi = float(sum(datay == label)) / float(len(datay))

    return mu, var, pi


def fit_univariate_generative_model(
        x: np.ndarray,
        y: np.ndarray,
        feature: int,
        labels: list) -> tuple:

    k = len(labels) # Number of classes
    mu = np.zeros(k) # List of means
    var = np.zeros(k) # List of variances
    pi = np.zeros(k) # List of class weights

    for label in labels:
        indices = (y == label)
        mu[label - 1], var[label - 1], pi[label - 1] = get_normal_dist(x, y, label, feature)

    return mu, var, pi