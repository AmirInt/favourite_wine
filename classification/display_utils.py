import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from classification.generative_model_utils import get_normal_dist




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