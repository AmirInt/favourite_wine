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
        mu: float,
        var: float,
        pi: float,
        x: np.ndarray,
        y: np.ndarray,
        feature: int,
        features: list,
        labels: list) -> float:
    
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

