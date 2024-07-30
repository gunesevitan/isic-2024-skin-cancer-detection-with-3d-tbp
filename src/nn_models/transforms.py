import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_image_transforms(**transform_parameters):

    """
    Get image transforms for dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    transforms: dict
        Transforms for training and inference
    """

    training_transforms = A.Compose([
        A.Transpose(p=transform_parameters['transpose_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.RandomBrightnessContrast(
            brightness_limit=transform_parameters['brightness_limit'],
            contrast_limit=transform_parameters['contrast_limit'],
            brightness_by_max=True,
            p=transform_parameters['random_brightness_contrast_probability']
        ),

        #A.PadIfNeeded(
        #    min_height=transform_parameters['pad_min_height'],
        #    min_width=transform_parameters['pad_min_width'],
        #    border_mode=cv2.BORDER_CONSTANT,
        #    value=0
        #),
        A.Resize(
            height=224,
            width=224,
            interpolation=cv2.INTER_LINEAR
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    inference_transforms = A.Compose([
        #A.PadIfNeeded(
        #    min_height=transform_parameters['pad_min_height'],
        #    min_width=transform_parameters['pad_min_width'],
        #    border_mode=cv2.BORDER_CONSTANT,
        #    value=0
        #),
        A.Resize(
            height=224,
            width=224,
            interpolation=cv2.INTER_LINEAR
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    transforms = {'training': training_transforms, 'inference': inference_transforms}
    return transforms