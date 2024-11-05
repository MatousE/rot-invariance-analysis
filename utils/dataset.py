
import os
from torch.utils.data import Dataset
from PIL import Image
from .wsi_utils import read_wsi, convert_to_color_spaces, segment_hsv_image, extract_patches


class KIRCDataset(Dataset):
    def __init__(self, root_dir, level=2, patch_size=256):
        
        self.level = level
        self.patch_size = patch_size
        self.case_image_paths = [[os.path.join(root_dir, case, file) for file in os.listdir(os.path.join(root_dir, case)) if file.endswith('.svs')][0] for case in os.listdir(root_dir)]

    def __len__(self):
        return len(self.case_image_paths)
    
    def __getitem__(self, i):

        _, rgba_image, _ = read_wsi(self.case_image_paths[i], level=self.level)

        wsi_rgb, _, wsi_hsv = convert_to_color_spaces(rgba_image)

        _, _, contours, mask = segment_hsv_image(wsi_hsv)

        patches, _ = extract_patches(wsi_rgb, contours, mask, self.patch_size) 

        patches = [Image.fromarray(patch.astype('uint8'), 'RGB') for patch in patches]


        return patches



