import sys
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'isic-2017-challenge'
    train_image_directory = dataset_directory / 'ISIC-2017_Training_Data'
    val_image_directory = dataset_directory / 'ISIC-2017_Validation_Data'
    test_image_directory = dataset_directory / 'ISIC-2017_Test_v2_Data'

    df_train = pd.read_csv(dataset_directory / 'ISIC-2017_Training_Data_metadata.csv')
    df_train_labels = pd.read_csv(dataset_directory / 'ISIC-2017_Training_Part3_GroundTruth.csv')
    df_train = df_train.merge(df_train_labels, on='image_id', how='left')

    df_val = pd.read_csv(dataset_directory / 'ISIC-2017_Validation_Data_metadata.csv')
    df_val_labels = pd.read_csv(dataset_directory / 'ISIC-2017_Validation_Part3_GroundTruth.csv')
    df_val = df_val.merge(df_val_labels, on='image_id', how='left')

    df_test = pd.read_csv(dataset_directory / 'ISIC-2017_Test_v2_Data_metadata.csv')
    df_test_labels = pd.read_csv(dataset_directory / 'ISIC-2017_Test_v2_Part3_GroundTruth.csv')
    df_test = df_test.merge(df_test_labels, on='image_id', how='left')

    output_directory = settings.DATA / 'datasets'
    output_directory.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

        image = cv2.imread(train_image_directory / f'{row["image_id"]}.jpg')

        height, width = image.shape[:2]
        df_train.loc[idx, 'height'] = height
        df_train.loc[idx, 'width'] = width

    df_train['height'] = df_train['height'].astype(int)
    df_train['width'] = df_train['width'].astype(int)
    df_train['image_path'] = df_train['image_id'].apply(lambda x: str(train_image_directory / f'{str(x)}.jpg'))

    for idx, row in tqdm(df_val.iterrows(), total=df_val.shape[0]):

        image = cv2.imread(val_image_directory / f'{row["image_id"]}.jpg')

        height, width = image.shape[:2]
        df_val.loc[idx, 'height'] = height
        df_val.loc[idx, 'width'] = width

    df_val['height'] = df_val['height'].astype(int)
    df_val['width'] = df_val['width'].astype(int)
    df_val['image_path'] = df_val['image_id'].apply(lambda x: str(val_image_directory / f'{str(x)}.jpg'))

    for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):

        image = cv2.imread(test_image_directory / f'{row["image_id"]}.jpg')

        height, width = image.shape[:2]
        df_test.loc[idx, 'height'] = height
        df_test.loc[idx, 'width'] = width

    df_test['height'] = df_test['height'].astype(int)
    df_test['width'] = df_test['width'].astype(int)
    df_test['image_path'] = df_test['image_id'].apply(lambda x: str(test_image_directory / f'{str(x)}.jpg'))

    df_metadata = pd.concat((df_train, df_val, df_test), axis=0).reset_index(drop=True)
    df_metadata = df_metadata.rename(columns={'image_id': 'isic_id', 'melanoma': 'target'})
    df_metadata['target'] = df_metadata['target'].astype(int)
    df_metadata['seborrheic_keratosis'] = df_metadata['seborrheic_keratosis'].astype(int)
    df_metadata['id'] = pd.Series(df_metadata.index).apply(lambda x: f'isic_2017_{x}')
    df_metadata['dataset'] = 'isic_2017'

    columns = [
        'id', 'isic_id', 'target', 'seborrheic_keratosis', 'age_approximate', 'sex',
        'dataset', 'height', 'width', 'image_path'
    ]
    df_metadata = df_metadata[columns].copy(deep=True)

    df_metadata.to_parquet(output_directory / 'isic-2017-metadata.parquet')
    settings.logger.info(f'isic-2017-metadata.parquet is saved to {output_directory}')
