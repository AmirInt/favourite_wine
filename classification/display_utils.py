import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from classification.generative_model_utils import get_normal_dist, fit_univariate_generative_model




def display_univariate_plot(
        datax: np.ndarray,
        datay: np.ndarray,
        label: int,
        feature: int,
        features: list) -> None:

    plt.hist(datax[datay == label, feature], density=True)

    mu, var, _ = get_normal_dist(datax, datay, label, feature)

    std = np.sqrt(var)

    x_axis = np.linspace(mu - 3 * std, mu + 3 * std, 1000)

    plt.plot(x_axis, norm.pdf(x_axis, mu, std), 'r', lw=2)
    plt.title(f"Winery {label}")
    plt.xlabel(features[feature], fontsize=14, color='red')
    plt.ylabel('Density', fontsize=14, color='red')
    plt.show()


def show_univariate_densities(
        datax: np.ndarray,
        datay: np.ndarray,
        feature: int,
        labels: list,
        features: list) -> None:

    mu, var, pi = fit_univariate_generative_model(
        datax, datay, feature, labels)

    colours = ['r', 'k', 'g']

    for label in labels:
        m = mu[label - 1]
        s = np.sqrt(var[label - 1])
        x_axis = np.linspace(m - 3 * s, m + 3 * s, 1000)
        plt.plot(
            x_axis,
            norm.pdf(x_axis, m, s),
            colours[label - 1],
            label=f"Class {label}")
    
    plt.xlabel(features[feature], fontsize=14, color='red')
    plt.ylabel('Density', fontsize=14, color='red')
    plt.legend()
    plt.show()