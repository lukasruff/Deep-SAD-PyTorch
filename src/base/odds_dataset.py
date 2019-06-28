from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np


class ODDSDataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    urls = {
        'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1',
        'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1',
        'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1',
        'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=1',
        'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1',
        'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1'
    }

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = self.dataset_name + '.mat'
        self.data_file = self.root / self.file_name

        if download:
            self.download()

        mat = loadmat(self.data_file)
        X = mat['X']
        y = mat['y'].ravel()
        idx_norm = y == 0
        idx_out = y == 1

        # 60% data for training and 40% for testing; keep outlier ratio
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                                test_size=0.4,
                                                                                random_state=random_state)
        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                            test_size=0.4,
                                                                            random_state=random_state)
        X_train = np.concatenate((X_train_norm, X_train_out))
        X_test = np.concatenate((X_test_norm, X_test_out))
        y_train = np.concatenate((y_train_norm, y_train_out))
        y_test = np.concatenate((y_test_norm, y_test_out))

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

    def download(self):
        """Download the ODDS dataset if it doesn't exist in root already."""

        if self._check_exists():
            return

        # download file
        download_url(self.urls[self.dataset_name], self.root, self.file_name)

        print('Done!')
