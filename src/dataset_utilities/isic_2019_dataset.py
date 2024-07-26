import sys
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'isic-2019-challenge'
    train_image_directory = dataset_directory / 'ISIC_2019_Training_Input'
    test_image_directory = dataset_directory / 'ISIC_2019_Test_Input'

    df_train = pd.read_csv(dataset_directory / 'ISIC_2019_Training_Metadata.csv')
    df_train_labels = pd.read_csv(dataset_directory / 'ISIC_2019_Training_GroundTruth.csv')
    df_train = df_train.merge(df_train_labels, on='image', how='left')

    df_test = pd.read_csv(dataset_directory / 'ISIC_2019_Test_Metadata.csv')

    output_directory = settings.DATA / 'datasets'
    output_directory.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

        image = cv2.imread(train_image_directory / f'{row["image"]}.jpg')

        height, width = image.shape[:2]
        df_train.loc[idx, 'height'] = height
        df_train.loc[idx, 'width'] = width

    df_train['height'] = df_train['height'].astype(int)
    df_train['width'] = df_train['width'].astype(int)
    df_train['image_path'] = df_train['image'].apply(lambda x: str(train_image_directory / f'{str(x)}.jpg'))

    for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):

        image = cv2.imread(test_image_directory / f'{row["image"]}.jpg')

        height, width = image.shape[:2]
        df_test.loc[idx, 'height'] = height
        df_test.loc[idx, 'width'] = width

    df_test['height'] = df_test['height'].astype(int)
    df_test['width'] = df_test['width'].astype(int)
    df_test['image_path'] = df_test['image'].apply(lambda x: str(test_image_directory / f'{str(x)}.jpg'))

    df_metadata = pd.concat((df_train, df_test), axis=0).reset_index(drop=True).drop(columns=['lesion_id'])
    df_metadata = df_metadata.rename(columns={
        'image': 'isic_id',
        'MEL': 'target',
        'NV': 'nevus',
        'BCC': 'basal_cell_carcinoma',
        'AK': 'actinic_keratosis',
        'BKL': 'benign_keratosis',
        'DF': 'dermatofibroma',
        'VASC': 'vascular_lesion',
        'SCC': 'squamous_cell_carcinoma',
        'UNK': 'other'
    })
    df_metadata['id'] = pd.Series(df_metadata.index).apply(lambda x: f'isic_2019_{x}')
    df_metadata['dataset'] = 'isic_2019'

    columns = [
        'id', 'isic_id', 'target', 'nevus',
        'basal_cell_carcinoma', 'actinic_keratosis',
        'benign_keratosis', 'dermatofibroma', 'vascular_lesion',
        'squamous_cell_carcinoma', 'other',
        'dataset', 'height', 'width', 'image_path'
    ]
    df_metadata = df_metadata[columns].copy(deep=True)

    df_metadata.to_parquet(output_directory / 'isic-2019-metadata.parquet')
    settings.logger.info(f'isic-2019-metadata.parquet is saved to {output_directory}')
