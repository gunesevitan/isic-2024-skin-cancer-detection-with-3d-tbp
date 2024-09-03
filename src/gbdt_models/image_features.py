import sys
from tqdm import tqdm
import h5py
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM

sys.path.append('..')
import settings


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'isic-2024-challenge' / 'train-metadata.csv')
    settings.logger.info(f'Dataset Shape {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    train_images = h5py.File(settings.DATA / 'isic-2024-challenge' / 'train-image.hdf5', 'r+')

    image_features = []

    for isic_id in tqdm(train_images):

        image = np.array(Image.open(BytesIO(train_images[isic_id][()])))

        image_mean = np.mean(image)
        image_std = np.std(image)
        image_min = np.min(image)
        image_max = np.max(image)
        image_skew = skew(image, axis=(0, 1, 2))

        image_r_mean = np.mean(image[:, :, 0])
        image_r_std = np.std(image[:, :, 0])
        image_r_min = np.min(image[:, :, 0])
        image_r_max = np.max(image[:, :, 0])
        image_r_skew = skew(image[:, :, 0], axis=(0, 1))

        image_g_mean = np.mean(image[:, :, 1])
        image_g_std = np.std(image[:, :, 1])
        image_g_min = np.min(image[:, :, 1])
        image_g_max = np.max(image[:, :, 1])
        image_g_skew = skew(image[:, :, 1], axis=(0, 1))

        image_b_mean = np.mean(image[:, :, 2])
        image_b_std = np.std(image[:, :, 2])
        image_b_min = np.min(image[:, :, 2])
        image_b_max = np.max(image[:, :, 2])
        image_b_skew = skew(image[:, :, 2], axis=(0, 1))

        image_features.append({
            'isic_id': isic_id,

            'image_mean': image_mean,
            'image_std': image_std,
            'image_min': image_min,
            'image_max': image_max,
            'image_skew': image_skew,

            'image_r_mean': image_r_mean,
            'image_r_std': image_r_std,
            'image_r_min': image_r_min,
            'image_r_max': image_r_max,
            'image_r_skew': image_r_skew,

            'image_g_mean': image_g_mean,
            'image_g_std': image_g_std,
            'image_g_min': image_g_min,
            'image_g_max': image_g_max,
            'image_g_skew': image_g_skew,

            'image_b_mean': image_b_mean,
            'image_b_std': image_b_std,
            'image_b_min': image_b_min,
            'image_b_max': image_b_max,
            'image_b_skew': image_b_skew,
        })

    image_features = pd.DataFrame(image_features)
    image_features.to_parquet(settings.DATA / 'image_features.parquet')
