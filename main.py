import sys
import numpy as np
import classification.display_utils as dutils
import classification.generative_model_utils as gmutils
from classification.config_utils import load_config
from classification.dataset_utils import Dataset




def load_dataset() -> Dataset:
    config = load_config()

    dataset = Dataset(
        config["dataset"]["path"],
        config["dataset"]["train_count"],
        config["dataset"]["features"],
        config["dataset"]["labels"])
    
    dataset.prepare_dataset()

    return dataset


def compare_uni_feature_stds() -> None:
    dataset = load_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    vars = np.zeros(len(features))
    for label in labels:
        for feature, _ in enumerate(features):
            _, vars[feature], _ = gmutils.get_univariate_normal_dist(
                dataset.get_trainx(),
                dataset.get_trainy(),
                label,
                feature)
        
        print(f"Minimum standard deviation for Class {label} is for feature {np.argmin(vars)} ({features[np.argmin(vars)]})")

        dutils.display_univariate_plot(
            dataset.get_trainx(),
            dataset.get_trainy(),
            label,
            np.argmin(vars),
            features)


def compare_uni_label_dists(feature: int) -> None:
    dataset = load_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    dutils.show_univariate_densities(
        dataset.get_trainx(),
        dataset.get_trainy(),
        feature,
        labels,
        features)


def compare_uni_feature_classifications() -> None:
    dataset = load_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    train_errors = np.zeros(len(features))
    test_errors = np.zeros(len(features))

    for feature, _ in enumerate(features):
        mu, var, pi = gmutils.fit_univariate_generative_model(
            dataset.get_trainx(),
            dataset.get_trainy(),
            feature,
            labels)

        print("Train:")
        train_errors[feature] = gmutils.test_univariate_model(
            mu, var, pi,
            dataset.get_trainx(),
            dataset.get_trainy(),
            feature,
            labels,
            features)

        print("Test:")
        test_errors[feature] = gmutils.test_univariate_model(
            mu, var, pi,
            dataset.get_testx(),
            dataset.get_testy(),
            feature,
            labels,
            features)
        print()

    optimum_feature = np.argmin(train_errors)
    print(f"Minimum train error and the corresponding test error are respectively {train_errors[optimum_feature]} {test_errors[optimum_feature]}, that belongs to feature {optimum_feature} ({features[optimum_feature]})")


def compare_bivariate_label_dists(f1: int, f2: int) -> None:
    dataset = load_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    dutils.plot_bivariate_classes(
        dataset.get_trainx(),
        dataset.get_trainy(),
        f1,
        f2,
        labels,
        features)


def report_bivariate_errors(errors: np.ndarray, features: list) -> None:
    for f1 in range(len(features)):
        print(f"\t{f1}", end='')
    print()
    
    for f2 in range(len(features)):
        print(f"{f2}\t", end='')
        for f1 in range(len(features)):
            if f1 >= f2:
                print("\t", end='')
            else:
                print(f"{errors[f1, f2]:.3f}\t", end='')
        print()


def compare_bivariate_feature_classifications() -> None:
    dataset = load_dataset()

    labels = dataset.get_labels()
    features = dataset.get_features()

    train_errors = np.ones((len(features), len(features)))
    test_errors = np.ones((len(features), len(features)))

    for f1 in range(len(features)):
        for f2 in range(f1 + 1, len(features)):
            mu, covar, pi = gmutils.fit_bivariate_generative_model(
                dataset.get_trainx(),
                dataset.get_trainy(),
                [f1, f2],
                labels)
            
            train_errors[f1, f2] = gmutils.test_bivariate_model(
                mu, covar, pi,
                dataset.get_trainx(),
                dataset.get_trainy(),
                f1,
                f2,
                labels,
                features)
            
            test_errors[f1, f2] = gmutils.test_bivariate_model(
                mu, covar, pi,
                dataset.get_testx(),
                dataset.get_testy(),
                f1,
                f2,
                labels,
                features)

    print("Train errors:")
    report_bivariate_errors(train_errors, features)
    print()
    print("Test errors:")
    report_bivariate_errors(test_errors, features)
    print()

    optimum_feature_combination = np.argwhere(
        train_errors == np.min(train_errors))
    optimum_feature_combination = optimum_feature_combination[0, 0], optimum_feature_combination[0, 1]
    print(f"Minimum train error and the corresponding test error are respectively {train_errors[optimum_feature_combination]:0.3f} and {test_errors[optimum_feature_combination]:0.3f}, that belong to the feature combination {optimum_feature_combination} ({features[optimum_feature_combination[0]]}, {features[optimum_feature_combination[1]]})")
    print()
    optimum_feature_combination = np.argwhere(
        test_errors == np.min(test_errors))
    optimum_feature_combination = optimum_feature_combination[0, 0], optimum_feature_combination[0, 1]
    print(f"Minimum test error and the corresponding train error are respectively {test_errors[optimum_feature_combination]:0.3f} and {train_errors[optimum_feature_combination]:0.3f}, that belong to the feature combination {optimum_feature_combination} ({features[optimum_feature_combination[0]]}, {features[optimum_feature_combination[1]]})")


if __name__ == "__main__":
    try:
        if sys.argv[1] == "compare_uni_feature_stds":
            compare_uni_feature_stds()
        elif sys.argv[1] == "compare_uni_label_dists":
            compare_uni_label_dists(int(sys.argv[2]))
        elif sys.argv[1] == "compare_uni_feature_classifications":
            compare_uni_feature_classifications()
        elif sys.argv[1] == "compare_bivariate_label_dists":
            compare_bivariate_label_dists(int(sys.argv[2]), int(sys.argv[3]))
        elif sys.argv[1] == "compare_bivariate_feature_classifications":
            compare_bivariate_feature_classifications()
        else:
            raise IndexError
    except IndexError:
        print("Arguments:")
        print()
        print("- compare_uni_feature_stds (Compare the standard deviations for each feature in the univariate schema)")
        print()
        print("- compare_uni_label_dists [feature_#] (Compare the guassian distributions of classes for the given feature in the univariate schema)")
        print()
        print("- compare_uni_feature_classifications (Fit models on data based on all features and compare the train and test errors in the univariate schema)")
        print()
        print("- compare_bivariate_features [feature_#_1] [feature_#_2] (Compare the guassian distributions of classes for the given feature combination in the bivariate schema)")
        print()
        print("- compare_bivariate_feature_classifications (Fit models on data based on all feature combinations and compare the train and test errors in the bivariate schema)")

    except KeyboardInterrupt:
        print("User interrupted, exiting...")
        pass