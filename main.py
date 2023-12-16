import sys
import numpy as np
from classification.config_utils import load_config
from classification.dataset_utils import Dataset
from classification.display_utils import display_univariate_plot, show_univariate_densities
from classification.generative_model_utils import get_univariate_normal_dist, fit_univariate_generative_model, test_univariate_model




def compare_feature_stds() -> None:
    config = load_config()

    dataset = Dataset(
        config["dataset"]["path"],
        config["dataset"]["train_count"],
        config["dataset"]["features"],
        config["dataset"]["labels"])
    
    dataset.prepare_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    vars = np.zeros(len(features))
    for label in labels:
        for feature, _ in enumerate(features):
            _, vars[feature], _ = get_univariate_normal_dist(
                dataset.get_trainx(),
                dataset.get_trainy(),
                label,
                feature)
        
        print(f"Minimum standard deviation for Class {label} is for feature {np.argmin(vars)} ({features[np.argmin(vars)]})")

        display_univariate_plot(
            dataset.get_trainx(),
            dataset.get_trainy(),
            label,
            np.argmin(vars),
            features)


def compare_label_dists() -> None:
    config = load_config()

    dataset = Dataset(
        config["dataset"]["path"],
        config["dataset"]["train_count"],
        config["dataset"]["features"],
        config["dataset"]["labels"])
    
    dataset.prepare_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    for feature, _ in enumerate(features):
        show_univariate_densities(
            dataset.get_trainx(),
            dataset.get_trainy(),
            feature,
            labels,
            features)


def compare_feature_tests() -> None:
    config = load_config()

    dataset = Dataset(
        config["dataset"]["path"],
        config["dataset"]["train_count"],
        config["dataset"]["features"],
        config["dataset"]["labels"])
    
    dataset.prepare_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    train_errors = np.zeros(len(features))
    test_errors = np.zeros(len(features))

    for feature, _ in enumerate(features):
        mu, var, pi = fit_univariate_generative_model(
            dataset.get_trainx(),
            dataset.get_trainy(),
            feature,
            labels)

        print("Train:")
        train_errors[feature] = test_univariate_model(
            mu, var, pi,
            dataset.get_trainx(),
            dataset.get_trainy(),
            feature,
            features,
            labels
        )

        print("Test:")
        test_errors[feature] = test_univariate_model(
            mu, var, pi,
            dataset.get_testx(),
            dataset.get_testy(),
            feature,
            features,
            labels
        )
        print()

    optimum_feature = np.argmin(train_errors)
    print(f"Minimum train error and the corresponding test error are respectively {train_errors[optimum_feature]} {test_errors[optimum_feature]}, that belongs to feature {optimum_feature} ({features[optimum_feature]})")


if __name__ == "__main__":
    try:
        if sys.argv[1] == "compare_feature_stds":
            compare_feature_stds()
        elif sys.argv[1] == "compare_label_dists":
            compare_label_dists()
        elif sys.argv[1] == "compare_feature_tests":
            compare_feature_tests()
    except KeyboardInterrupt:
        print("User interrupted, exiting...")
        pass