import os
import time
import yaml
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from utils import KIRCDataset, model_registry, rotate_crop


class ModelCache:
    """Cache models to avoid redundant loading."""

    def __init__(self, models, device):
        self.models = models
        self.device = device
        self.cache = {}

    def get_model(self, model_name, model_params):
        if model_name not in self.cache:
            model, transform, wrapper = model_registry[model_name](model_params)
            self.cache[model_name] = wrapper(model, transform, self.device)
        return self.cache[model_name]


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_embeddings(
        patches, wrapper, theta, crop_size, batch_size=16
):
    embeddings = []

    patches = [rotate_crop(patch, theta, crop_size) for patch in patches]

    for i in range(0, len(patches), batch_size):
        batch_images = patches[i:i + batch_size]
        batch_embds = wrapper.get_embeddings(batch_images)
        embeddings.append(batch_embds)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def main(params):
    image_dir = params['image_directory']
    image_level = params['image_level']
    image_patch_size = params['image_patch_size']
    image_patch_samples = params['image_patch_samples']
    image_patch_crop_size = params['image_patch_crop_size']
    theta_samples = params['theta_samples']
    output_dir = params['output_directory']
    models = params['models']
    model_params = params['model_parameters']

    set_seed()

    dataset = KIRCDataset(
        root_dir=image_dir,
        level=image_level,
        patch_size=image_patch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_slides = 0
    total_patches = 0
    model_cache = ModelCache(models, device)

    all_embeddings = {}  # {aug_name: {model_name: [embeddings]}}
    print('STARTING EXPERIMENT')

    for i in range(len(dataset)):  # Iterate over WSIs (or dataset)

        patches = dataset[i]
        print(len(patches))

        if len(patches) > image_patch_samples:
            indices = random.sample(range(0, len(patches)), image_patch_samples)
            patches = [patches[i] for i in indices]

        total_slides += 1
        total_patches += len(patches)
        thetas = np.linspace(start=0, stop=360, num=theta_samples, endpoint=False)

        st = time.time()

        for model_name, model_param in zip(models, model_params):
            model_wrapper = model_cache.get_model(model_name, model_param)
            if model_name not in all_embeddings:
                all_embeddings[model_name] = {}

            for theta in thetas:
                experiment_name = f"rotate_{int(theta)}"

                embeddings = extract_embeddings(patches=patches,
                                                wrapper=model_wrapper,
                                                theta=theta,
                                                crop_size=[image_patch_crop_size, image_patch_crop_size])

                if experiment_name not in all_embeddings[model_name]:
                    all_embeddings[model_name][experiment_name] = []

                print(f'WSI: {i} - Model: {model_name} - Theta: {int(theta)} - Embedding: {embeddings.shape}')

                all_embeddings[model_name][experiment_name].append(embeddings)

        print(f"Processing time: {time.time() - st:.2f} seconds")

    print('ENDING EXPERIMENT')
    print(f'TOTAL SLIDES: {total_slides}')
    print(f'TOTAL PATCHES: {total_patches}')

    current_date = datetime.now()

    # Format the date as YYYYMMDD
    formatted_date = current_date.strftime('%Y%m%d')

    # Concatenate all embeddings along the batch dimension
    final_embeddings = {}  # {aug_name: {model_name: concatenated_embeddings}}
    for model_name, aug_dict in all_embeddings.items():
        final_embeddings[model_name] = {}
        for aug_name, emb_list in aug_dict.items():
            final_embeddings[model_name][aug_name] = torch.cat(emb_list, dim=0)

    print(final_embeddings)

    output_path = os.path.join(output_dir, f"final_embeddings_{formatted_date}.pth")
    print(output_path)
    torch.save(final_embeddings, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--yaml', type=str, help='Path to experiment parameters YAML file.')
    args = parser.parse_args()

    with open(args.yaml, 'r') as stream:
        params = yaml.safe_load(stream)
        print(params)
        main(params)
