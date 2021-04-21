import albumentations as aug

"""
The get_transforms() function will be called when constructing datasets for training
(see DataLoader.py)

Data augmentation will be applied in the data pipeline to help both models do better
In the SSD paper, only cropping is mentioned. 
In the YOLO paper, tricks mentioned include crops, rotation, and hue, saturation and exposure shifts. 
We will apply cropping, hue, and saturation change. Rotation will not be done since segmentation polygons are not available.

We will also perform the following data augmentation techniques:
- Gaussian blur
- Random brightness

Augmentation is done using the "albumentations" library (https://github.com/albumentations-team/albumentations)
"""

def get_transforms(model = 'ssd'):
    assert model in ['ssd', 'yolo']
    if model == 'ssd':
        height, width = 300, 300
    else:
        height, width = 320, 320

    return aug.Compose([
        aug.RandomSizedBBoxSafeCrop(height=height, width=width, p=0.5),
        aug.HueSaturationValue(),
        aug.GaussianBlur(blur_limit=(1,3)),
        # aug.MotionBlur(blur_limit=5),
        aug.RandomBrightnessContrast(),
        # aug.RandomFog(),
        # aug.Rotate(limit=5)  # apply rotation with a small angle
    ], bbox_params=aug.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['class_labels']))