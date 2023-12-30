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

        self.__data = np.loadtxt(data_path, delimiter=',')
        self.__train_count = train_count
        self.__features = features
        self.__labels = labels


    def prepare_dataset(self) -> None:
        np.random.seed(0)
        perm = np.random.permutation(self.__data.shape[0])
        self.__trainx = self.__data[perm[0:self.__train_count], 1:14]
        self.__trainy = self.__data[perm[0:self.__train_count], 0]
        self.__testx = self.__data[perm[self.__train_count:self.__data.shape[0]], 1:14]
        self.__testy = self.__data[perm[self.__train_count:self.__data.shape[0]], 0]


    def get_features(self) -> list:
        return self.__features

    
    def get_labels(self) -> list:
        return self.__labels
    

    def get_trainx(self) -> np.ndarray:
        return self.__trainx
    
    
    def get_trainy(self) -> np.ndarray:
        return self.__trainy
    
    
    def get_testx(self) -> np.ndarray:
        return self.__testx
    
    
    def get_testy(self) -> np.ndarray:
        return self.__testy