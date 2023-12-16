import numpy as np
import matplotlib.pyplot as plt
import classification.generative_model_utils as gmutils
from scipy.stats import norm, multivariate_normal




def display_univariate_plot(
        datax: np.ndarray,
        datay: np.ndarray,
        label: int,
        feature: int,
        features: list) -> None:

    plt.hist(datax[datay == label, feature], density=True)

    mu, var, _ = gmutils.get_univariate_normal_dist(datax, datay, label, feature)

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

    mu, var, pi = gmutils.fit_univariate_generative_model(
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


def find_range(x: np.ndarray) -> tuple:
    lower = min(x)
    upper = max(x)
    width = upper - lower
    lower = lower - 0.2 * width
    upper = upper + 0.2 * width
    return lower, upper


def plot_bivariate_contours(
        mu: np.ndarray,
        cov: np.ndarray,
        x1g: np.ndarray,
        x2g: np.ndarray,
        colour: any) -> None:
    
    rv = multivariate_normal(mean=mu, cov=cov)
    z = np.zeros((len(x1g), len(x2g)))
    
    for i in range(len(x1g)):
        for j in range(len(x2g)):
            z[j, i] = rv.logpdf([x1g[i], x2g[j]])
    
    sign, logdet = np.linalg.slogdet(cov)
    normalizer = -0.5 * (2 * np.log(6.28) + sign * logdet)
    
    for offset in range(1,4):
        plt.contour(
            x1g,
            x2g,
            z,
            levels=[normalizer - offset],
            colors=colour,
            linewidths=2.0,
            linestyles='solid')


def plot_bivariate_points(
        datax: np.ndarray,
        datay: np.ndarray,
        label: int,
        f1: int,
        f2: int,
        features: list) -> None:
    """
    Displays the points and contours of one class based on the two features given.
    """    

    if f1 == f2:
        print("Choose different features for f1 and f2.")
        return

    # Set up plot
    x1_lower, x1_upper = find_range(datax[datay == label, f1])
    x2_lower, x2_upper = find_range(datax[datay == label, f2])
    plt.xlim(x1_lower, x1_upper) # limit along x1-axis
    plt.ylim(x2_lower, x2_upper) # limit along x2-axis

    # Plot the training points along the two selected features
    plt.plot(datax[datay == label, f1], datax[datay == label, f2], 'ro')

    # Define a grid along each axis; the density will be computed at each grid point
    res = 200 # resolution
    x1g = np.linspace(x1_lower, x1_upper, res)
    x2g = np.linspace(x2_lower, x2_upper, res)

    # Now plot a few contour lines of the density
    mu, cov = fit_gaussian(datax[datay == label, :], [f1, f2])
    plot_contours(mu, cov, x1g, x2g, 'k')

    # Finally, display
    plt.xlabel(features[f1], fontsize=14, color='red')
    plt.ylabel(features[f2], fontsize=14, color='red')
    plt.title(f'Class {label}', fontsize=14, color='blue')
    plt.show()


def plot_bivariate_classes(
        datax: np.ndarray,
        datay: np.ndarray,
        f1: int,
        f2: int,
        labels: list,
        features: list) -> None:

    if f1 == f2: # we need f1 != f2
        print("Choose different features for f1 and f2.")
        return

    # Set up plot
    x1_lower, x1_upper = find_range(datax[:, f1])
    x2_lower, x2_upper = find_range(datax[:, f2])
    plt.xlim(x1_lower, x1_upper) # limit along x1-axis
    plt.ylim(x2_lower, x2_upper) # limit along x2-axis

    # Plot the training points along the two selected features
    colours = ['r', 'k', 'g']
    for label in labels:
        plt.plot(
            datax[datay == label, f1],
            datax[datay == label, f2],
            marker='o',
            ls='None',
            c=colours[label - 1])

    # Define a grid along each axis; the density will be computed at each grid point
    res = 200 # resolution
    x1g = np.linspace(x1_lower, x1_upper, res)
    x2g = np.linspace(x2_lower, x2_upper, res)

    # Show the Gaussian fit to each class, using features f1,f2
    mu, covar, pi = gmutils.fit_bivariate_generative_model(datax, datay, [f1, f2], labels)
    for label in labels:
        gmean = mu[label - 1, :]
        gcov = covar[label - 1, :, :]
        plot_bivariate_contours(
            gmean,
            gcov,
            x1g,
            x2g,
            colours[label - 1])

    # Finally, display
    plt.xlabel(features[f1], fontsize=14, color='red')
    plt.ylabel(features[f2], fontsize=14, color='red')
    plt.title('Wine Data', fontsize=14, color='blue')
    plt.show()