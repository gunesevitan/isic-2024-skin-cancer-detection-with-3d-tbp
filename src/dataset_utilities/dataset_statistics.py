import os
import sys
import json

import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_metadata = pd.read_parquet(settings.DATA / 'datasets' / 'isic_metadata.parquet')

    pixel_count = 0
    pixel_sum = 0
    pixel_squared_sum = 0

    for idx, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.float32(image) / 255.

        pixel_count += (image.shape[0] * image.shape[1])
        pixel_sum += np.sum(image, axis=(0, 1))
        pixel_squared_sum += np.sum(image ** 2, axis=(0, 1))

    mean = pixel_sum / pixel_count
    var = (pixel_squared_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    dataset_statistics = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    with open(settings.DATA / 'statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)
