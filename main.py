import sys
import numpy as np
from classification.config_utils import load_config
from classification.dataset_utils import Dataset
from classification.display_utils import display_univariate_plot
from classification.generative_model_utils import get_normal_dist




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
            _, vars[feature], _ = get_normal_dist(
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


if __name__ == "__main__":
    if sys.argv[1] == "compare_feature_stds":
        compare_feature_stds()