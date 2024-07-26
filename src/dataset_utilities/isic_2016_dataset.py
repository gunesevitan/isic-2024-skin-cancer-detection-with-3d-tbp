import sys
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'isic-2016-challenge'
    train_image_directory = dataset_directory / 'ISBI2016_ISIC_Part3_Training_Data'
    test_image_directory = dataset_directory / 'ISBI2016_ISIC_Part3_Test_Data'

    df_train = pd.read_csv(dataset_directory / 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
    df_test = pd.read_csv(dataset_directory / 'ISBI2016_ISIC_Part3_Test_GroundTruth.csv')

    df_train['target'] = df_train['target'].map({'benign': 0, 'malignant': 1}).astype(int)
    df_test['target'] = df_test['target'].astype(int)

    output_directory = settings.DATA / 'datasets'
    output_directory.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

        image = cv2.imread(train_image_directory / f'{row["id"]}.jpg')

        height, width = image.shape[:2]
        df_train.loc[idx, 'height'] = height
        df_train.loc[idx, 'width'] = width

    df_train['height'] = df_train['height'].astype(int)
    df_train['width'] = df_train['width'].astype(int)
    df_train['image_path'] = df_train['id'].apply(lambda x: str(train_image_directory / f'{str(x)}.jpg'))

    for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):

        image = cv2.imread(test_image_directory / f'{row["id"]}.jpg')

        height, width = image.shape[:2]
        df_test.loc[idx, 'height'] = height
        df_test.loc[idx, 'width'] = width

    df_test['height'] = df_test['height'].astype(int)
    df_test['width'] = df_test['width'].astype(int)
    df_test['image_path'] = df_test['id'].apply(lambda x: str(test_image_directory / f'{str(x)}.jpg'))

    df_metadata = pd.concat((df_train, df_test), axis=0).reset_index(drop=True)
    df_metadata['id'] = pd.Series(df_metadata.index).apply(lambda x: f'isic_2016_{x}')
    df_metadata['dataset'] = 'isic_2016'

    df_metadata.to_parquet(output_directory / 'isic-2016-metadata.parquet')
    settings.logger.info(f'isic-2016-metadata.parquet is saved to {output_directory}')
