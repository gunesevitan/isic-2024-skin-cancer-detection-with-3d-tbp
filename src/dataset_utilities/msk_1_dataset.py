import sys
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'msk-1'
    image_directory = dataset_directory / 'ISIC-images'

    df_metadata = pd.read_csv(dataset_directory / 'metadata.csv')

    output_directory = settings.DATA / 'datasets'
    output_directory.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

        image = cv2.imread(image_directory / f'{row["isic_id"]}.jpg')

        height, width = image.shape[:2]
        df_metadata.loc[idx, 'height'] = height
        df_metadata.loc[idx, 'width'] = width

    df_metadata['height'] = df_metadata['height'].astype(int)
    df_metadata['width'] = df_metadata['width'].astype(int)
    df_metadata['image_path'] = df_metadata['isic_id'].apply(lambda x: str(image_directory / f'{str(x)}.jpg'))

    df_metadata = df_metadata.rename(columns={
        'image_name': 'isic_id',
        'anatom_site_general_challenge': 'anatom_site_general'
    }).drop(columns=[
        'attribution', 'copyright_license',
        'concomitant_biopsy', 'diagnosis',
        'diagnosis_confirm_type', 'image_type'
    ])
    df_metadata['target'] = df_metadata['benign_malignant'].map({'benign': 0, 'malignant': 1})
    df_metadata['id'] = pd.Series(df_metadata.index).apply(lambda x: f'msk-1_{x}')
    df_metadata['dataset'] = 'msk-1'

    columns = [
        'id', 'isic_id', 'target', 'sex', 'age_approx',
        'anatom_site_general', 'dataset', 'height', 'width', 'image_path'
    ]
    df_metadata = df_metadata[columns].copy(deep=True)

    df_metadata.to_parquet(output_directory / 'msk-1-metadata.parquet')
    settings.logger.info(f'msk-1-metadata is saved to {output_directory}')
