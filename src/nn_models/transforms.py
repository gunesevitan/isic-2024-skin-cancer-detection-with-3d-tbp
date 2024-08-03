import numpy as np
import cv2
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class AdaptiveResize(ImageOnlyTransform):

    def __init__(
            self,
            resize_height, resize_width,
            upsample_interpolations, downsample_interpolations,
            always_apply=True, p=1.0
    ):

        super(AdaptiveResize, self).__init__(always_apply=always_apply, p=p)

        self.resize_height = resize_height
        self.resize_width = resize_width
        self.upsample_interpolations = upsample_interpolations
        self.downsample_interpolations = downsample_interpolations

    def apply(self, inputs, **kwargs):

        """
        Resize image based on its properties

        Parameters
        ----------
        inputs: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        inputs: numpy.ndarray of shape (resized_height, resized_width, channel)
             Resized image array
        """

        image_height, image_width = inputs.shape[:2]

        if self.resize_height > image_height or self.resize_width > image_width:
            if len(self.upsample_interpolations) > 1:
                interpolation = np.random.choice(self.upsample_interpolations)
            else:
                interpolation = self.upsample_interpolations[0]
        else:
            if len(self.downsample_interpolations) > 1:
                interpolation = np.random.choice(self.downsample_interpolations)
            else:
                interpolation = self.downsample_interpolations[0]

        inputs = cv2.resize(inputs, dsize=(self.resize_width, self.resize_height), interpolation=interpolation)

        return inputs


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
        A.HueSaturationValue(
            hue_shift_limit=transform_parameters['hue_shift_limit'],
            sat_shift_limit=transform_parameters['sat_shift_limit'],
            val_shift_limit=transform_parameters['val_shift_limit'],
            p=transform_parameters['hue_saturation_value_probability']
        ),
        AdaptiveResize(
            resize_height=transform_parameters['resize_height'],
            resize_width=transform_parameters['resize_width'],
            upsample_interpolations=(cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4),
            downsample_interpolations=(cv2.INTER_AREA,),
            always_apply=True
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    inference_transforms = A.Compose([
        AdaptiveResize(
            resize_height=transform_parameters['resize_height'],
            resize_width=transform_parameters['resize_width'],
            upsample_interpolations=(cv2.INTER_CUBIC,),
            downsample_interpolations=(cv2.INTER_AREA,),
            always_apply=True
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


def get_tta_inputs(inputs, tta_idx):

    """
    Get test-time augmented inputs

    Parameters
    ----------
    inputs: torch.Tensor of shape (batch_size, channel, height, width)
        Image tensor

    tta_idx: int (0 <= tta_idx <= 7)
        Index of the test-time augmentation operation

    Returns
    -------
    inputs: torch.Tensor of shape (batch_size, channel, height, width)
        Augmented image tensor
    """

    if tta_idx >= 4:
        inputs = inputs.transpose(2, 3)

    if tta_idx % 4 == 0:
        return inputs
    elif tta_idx % 4 == 1:
        return inputs.flip(2)
    elif tta_idx % 4 == 2:
        return inputs.flip(3)
    elif tta_idx % 4 == 3:
        return inputs.flip([2, 3])
