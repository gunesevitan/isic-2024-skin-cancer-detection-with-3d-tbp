import sys
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'isic-2024-challenge'
    image_directory = settings.DATA / 'isic-2024-challenge' / 'train-image' / 'image'

    df_train_metadata = pd.read_csv(dataset_directory / 'train-metadata.csv')
    settings.logger.info(f'Dataset Shape: {df_train_metadata.shape}')

    output_directory = settings.DATA / 'datasets'
    output_directory.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_train_metadata.iterrows(), total=df_train_metadata.shape[0]):

        image = cv2.imread(image_directory / f'{row["isic_id"]}.jpg')

        height, width = image.shape[:2]
        df_train_metadata.loc[idx, 'height'] = height
        df_train_metadata.loc[idx, 'width'] = width

    df_train_metadata['height'] = df_train_metadata['height'].astype(int)
    df_train_metadata['width'] = df_train_metadata['width'].astype(int)
    df_train_metadata['image_path'] = df_train_metadata['isic_id'].apply(lambda x: str(image_directory / f'{str(x)}.jpg'))

    df_train_metadata.to_parquet(output_directory / 'isic-2024-metadata.parquet')
