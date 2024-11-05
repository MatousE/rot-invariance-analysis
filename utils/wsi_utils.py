import os
import cv2
import time
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError


PIXEL_WHITE = 255
PIXEL_TH = 200
PIXEL_BLACK = 0
CHANNEL = 3


def read_wsi(tif_file_path, level):
    """Load a WSI at a specific level and return as a NumPy array."""
    try:
        wsi = OpenSlide(tif_file_path)
        num_levels = len(wsi.level_dimensions)
        if level >= num_levels:
            print(f"Warning: Requested level {level} is out of range. Using the highest available level {num_levels - 1}.")
            level = num_levels - 1

        slide_w, slide_h = wsi.level_dimensions[level]
        rgba_image = np.asarray(wsi.read_region((0, 0), level, (slide_w, slide_h)))
        return wsi, rgba_image, (slide_w, slide_h)
    except OpenSlideUnsupportedFormatError:
        print('Error: Unsupported slide format.')
        return None, None, None

'''
    Convert RGBA to RGB, HSV and GRAY.
'''
def convert_to_color_spaces(rgba_image):

    '''
        This function splits and merges R, G, B channels.
        HSV and GRAY images are also created for future segmentation procedure.

        Args:
            - rgba_: Image to be processed, NumPy array type.

    '''
    r, g, b, _ = cv2.split(rgba_image)
    rgb_image = cv2.merge((r, g, b))
    gray_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    return rgb_image, gray_image, hsv_image

'''
'''
def get_contours(contour_image, rgb_image_shape):
    """Extract contours and return bounding boxes, coordinates, and a binary mask."""
    contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contour_coords = [np.squeeze(c) for c in contours]

    mask = np.zeros(rgb_image_shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (PIXEL_WHITE,) * CHANNEL, thickness=-1)
    
    return bounding_boxes, contour_coords, contours, mask

'''
    Perform segmentation and get contours.
'''
def segment_hsv_image(hsv_image):
    """Perform HSV segmentation to remove background and extract contours."""
    lower, upper = np.array([20, 20, 20]), np.array([200, 200, 200])
    thresh = cv2.inRange(hsv_image, lower, upper)

    close_kernel = np.ones((15, 15), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    return get_contours(opened, hsv_image.shape)

def extract_patches(wsi_rgb, contours, mask, patch_size=256):
    """Extract valid patches from the segmented regions."""
    patches, patch_coords = [], []

    selected_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for i, contour in enumerate(selected_contours):
        x, y, w, h = cv2.boundingRect(np.squeeze(contour))

        for y_pos in range(y, y + h, patch_size):
            for x_pos in range(x, x + w, patch_size):
                patch = wsi_rgb[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]
                patch_mask = mask[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]

                if patch.shape == (patch_size, patch_size, CHANNEL):
                    bitwise = cv2.bitwise_and(patch, patch_mask)
                    white_pixel_count = cv2.countNonZero(cv2.cvtColor(bitwise, cv2.COLOR_RGB2GRAY))

                    if white_pixel_count >= (patch_size ** 2) * 0.75:
                        patches.append(patch)
                        patch_coords.append((x_pos, y_pos))

    return patches, patch_coords
