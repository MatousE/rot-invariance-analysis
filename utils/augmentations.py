import numpy as np
from PIL import Image
from torchvision import transforms


def rotate_crop(image, angle, crop_size):

    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    width, height = rotated_image.size
    crop_width, crop_height = crop_size

    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2

    cropped_image = rotated_image.crop((left, top, right, bottom))

    return cropped_image


def rotation(image, angle):
    """
    Rotates the image by a specified angle.

    Args:
    image (PIL.Image): Input image to be rotated.
    angle (int or float): The angle by which to rotate the image.

    Returns:
    PIL.Image: The rotated image.
    """
    return image.rotate(angle)


def v_flip(image):
    """
    Vertically flips the image.

    Args:
    image (PIL.Image): Input image to be vertically flipped.

    Returns:
    PIL.Image: The vertically flipped image.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def h_flip(image):
    """
    Horizontally flips the image.

    Args:
    image (PIL.Image): Input image to be horizontally flipped.

    Returns:
    PIL.Image: The horizontally flipped image.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


augmentation_registry = {
    'control': None,
    'rotation': rotation,
    'v_flip': v_flip,
    'h_flip': h_flip,
}