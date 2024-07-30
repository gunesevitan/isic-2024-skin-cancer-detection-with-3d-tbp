import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularImageDataset(Dataset):

    def __init__(self, image_paths, features=None, targets=None, transforms=None):

        self.image_paths = image_paths
        self.features = features
        self.targets = targets
        self.transforms = transforms

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.image_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        volume: torch.FloatTensor of shape (channel, depth, height, width)
            Volume tensor

        targets: torch.LongTensor of shape (5 or 10)
            Tensor of targets as class indices

        targets: torch.FloatTensor of shape (5 or 10)
            Tensor of sample weights
        """

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        else:
            image = torch.as_tensor(image, dtype=torch.float)

        if self.targets is not None:

            target = self.targets[idx]
            target = torch.as_tensor(target, dtype=torch.long)

            if self.features is not None:

                features = self.features[idx]
                features = torch.as_tensor(features, dtype=torch.float)

                return image, features, target

            else:

                return image, target

        else:

            if self.features is not None:

                features = self.features[idx]
                features = torch.as_tensor(features, dtype=torch.float)

                return image, features

            else:

                return image


def prepare_dataset(df, features):

    """
    Prepare inputs and outputs for dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with image path, feature and target columns

    features: list
        List of feature names

    Returns
    -------
    image_paths: numpy.ndarray of shape (n_samples)
        Array of image paths

    features: numpy.ndarray of shape (n_samples, n_features)
        Array of features

    targets: numpy.ndarray of shape (n_samples)
        Array of targets
    """

    image_paths = df['image_path'].values
    if features is not None:
        features = df[features].values
    targets = df['target'].values.reshape(-1, 1)

    return image_paths, features, targets
