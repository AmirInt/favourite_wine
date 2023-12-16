import os
import numpy as np
import random




class Dataset:
    def __init__(
            self,
            data_path: os.PathLike,
            train_count: int,
            features: list,
            labels: list) -> None:

        self._data = np.loadtxt(data_path, delimiter=',')
        self._train_count = train_count
        self._features = features
        self._labels = labels


    def prepare_dataset(self) -> None:
        np.random.seed(0)
        perm = np.random.permutation(self._data.shape[0])
        self._trainx = self._data[perm[0:self._train_count], 1:14]
        self._trainy = self._data[perm[0:self._train_count], 0]
        self._testx = self._data[perm[self._train_count:self._data.shape[0]], 1:14]
        self._testy = self._data[perm[self._train_count:self._data.shape[0]], 0]


    def get_features(self) -> list:
        return self._features

    
    def get_labels(self) -> list:
        return self._labels
    

    def get_trainx(self) -> np.ndarray:
        return self._trainx
    
    
    def get_trainy(self) -> np.ndarray:
        return self._trainy
    
    
    def get_testx(self) -> np.ndarray:
        return self._testx
    
    
    def get_testy(self) -> np.ndarray:
        return self._testy