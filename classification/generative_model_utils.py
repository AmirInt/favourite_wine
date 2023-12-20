import numpy as np
from scipy.stats import norm, multivariate_normal
 



def get_univariate_normal_dist(
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
        mu[label - 1], var[label - 1], pi[label - 1] = get_univariate_normal_dist(x, y, label, feature)

    return mu, var, pi


def test_univariate_model(
        mu: np.ndarray,
        var: np.ndarray,
        pi: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        feature: int,
        labels: list,
        features: list) -> float:
    
    scores = np.zeros((len(x), len(labels)))
    
    for i in range(len(x)):
        for label in labels:
            scores[i, label - 1] = np.log(pi[label - 1]) + \
                norm.logpdf(x[i, feature], mu[label - 1], np.sqrt(var[label - 1]))
            
    predictions = np.argmax(scores, axis=1) + 1

    error = np.sum(predictions != y) / float(len(y))

    print(f"Error for feature number {feature} ({features[feature]}): {error}")

    return error


def fit_bivariate_gaussian(
        x: np.ndarray,
        feature_indices: list) -> tuple:
    mu = np.mean(x[:, feature_indices], axis=0)
    covar = np.cov(x[:, feature_indices], rowvar=0, bias=1)
    return mu, covar


def fit_bivariate_generative_model(
        x: np.ndarray,
        y: np.ndarray,
        feature_indices: list,
        labels: list) -> tuple:
    
    k = len(labels) # Number of classes
    d = len(feature_indices) # Number of features
    
    mu = np.zeros((k, d)) # List of means
    covar = np.zeros((k, d, d)) # List of covariance matrices
    pi = np.zeros(k) # List of class weights
    
    for label in labels:
        indices = (y == label)
        mu[label - 1, :], covar[label - 1, :, :] = fit_bivariate_gaussian(x[indices, :], feature_indices)
        pi[label - 1] = float(sum(indices)) / float(len(y))
    
    return mu, covar, pi


def test_bivariate_model(
        mu: np.ndarray,
        covar: np.ndarray,
        pi: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        f1: int,
        f2: int,
        labels: list,
        features: list) -> float:
    
    if f1 == f2:
        print("Choose different features for f1 and f2.")
        return
    
    feature_indices= [f1, f2]
    
    k = len(labels) # Labels 1,2,...,k
    nt = len(y) # Number of test points
    scores = np.zeros((nt, k))

    for i in range(nt):
        for label in labels:
            scores[i, label - 1] = np.log(pi[label - 1]) + \
                multivariate_normal.logpdf(x[i, feature_indices], mean=mu[label - 1, :], cov=covar[label - 1, :, :])
    
    predictions = np.argmax(scores, axis=1) + 1
    
    error = np.sum(predictions != y) / len(y)
    
    print(f"Test error using feature combination {f1} ({features[f1]}) and {f2} ({features[f2]}): {error}")
    print()

    return error


def fit_multivariate_generative_model(
        x: np.ndarray,
        y: np.ndarray,
        labels: list) -> tuple:
    
    k = len(labels) # Number of classes
    d = x.shape[1] # Number of features
    
    mu = np.zeros((k, d)) # List of means
    sigma = np.zeros((k, d, d)) # List of covariance matrices
    pi = np.zeros(k) # List of class weights
    
    for label in labels:
        indices = (y == label)
        pi[label - 1] = float(sum(indices)) / float(len(y))
        mu[label - 1] = np.mean(x[indices, :], axis=0)
        sigma[label - 1] = np.cov(x[indices, :], rowvar=0, bias=1)
    
    return mu, sigma, pi


def test_multivariate_model(
        mu: np.ndarray,
        sigma: np.ndarray,
        pi: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        feature_indices: list,
        labels: list) -> float:
    
    k = len(labels) # Labels 1,2,...,k
    nt = len(y) # Number of test points
    scores = np.zeros((nt, k))
    
    for i in range(nt):
        for label in labels:
            scores[i, label - 1] = np.log(pi[label - 1]) + \
                multivariate_normal.logpdf(x[i, feature_indices], mean=mu[label - 1, feature_indices], cov=sigma[label - 1, feature_indices, feature_indices])
    
    predictions = np.argmax(scores, axis=1) + 1
    
    error = np.sum(predictions != y) / len(y)
    
    return error


def regularise_matrix(
        sigma: np.ndarray,
        c: float,
        k: int,
        d: int) -> np.ndarray:
    
    regulariser = np.array([np.diag(np.array([c for _ in range(d)])) for _ in range(k)])

    sigma += regulariser
    
    return sigma
