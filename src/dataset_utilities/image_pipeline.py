import sys
import shutil
from tqdm import tqdm
import pandas as pd
import cv2

sys.path.append('..')
import settings


def resize_with_aspect_ratio(image, longest_edge, interpolation=cv2.INTER_AREA):

    """
    Resize image while preserving its aspect ratio

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array

    longest_edge: int
        Desired number of pixels on the longest edge

    interpolation: int
        Interpolation method

    Returns
    -------
    image: numpy.ndarray of shape (resized_height, resized_width, 3)
        Resized image array
    """

    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    image = cv2.resize(image, dsize=(int(width * scale), int(height * scale)), interpolation=interpolation)

    return image


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    df_metadata = pd.read_parquet(dataset_directory / 'isic_metadata.parquet')

    output_directory = settings.DATA / 'isic_master_dataset'
    output_directory.mkdir(parents=True, exist_ok=True)

    output_image_directory = output_directory / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    longest_edge = 1024

    for idx, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

        height = row['height']
        width = row['width']
        
        output_image_path = output_image_directory / str(row['image_path']).split('/')[-1]
        df_metadata.loc[idx, 'image_path'] = output_image_path

        if height > longest_edge or width > longest_edge:
            image = cv2.imread(row['image_path'])
            image = resize_with_aspect_ratio(image=image, longest_edge=longest_edge, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            height = image.shape[0]
            width = image.shape[1]
            df_metadata.loc[idx, 'height'] = height
            df_metadata.loc[idx, 'width'] = width

        else:
            shutil.copy2(row['image_path'], output_image_path)

    df_metadata['image_path'] = df_metadata['image_path'].astype(str)
    df_metadata.to_parquet(output_directory / 'metadata.parquet')
    settings.logger.info(f'metadata.parquet is saved to {output_directory}')
