from pathlib import Path
from typing import Callable, List, Tuple, Union, Optional
import random

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


def augm1():
    
    transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5
        ),
        transforms.RandomRotation(360)
    ])

    path = Path(__file__).parents[1] / 'data' / 'satellite_small'

    dset = torchvision.datasets.ImageFolder(str(path), transforms)
    for i, (image, label) in enumerate(dset):
        cv2.imshow(f'Img {i + 1}', np.array(image))
        key = cv2.waitKey(20000)
        if key == 27:
            break


def display_image(
    img: Union[torch.Tensor, np.ndarray],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Display an image on a matplotlib figure.
    Parameters
    ----------
    img : Union[torch.Tensor, np.ndarray]
        An image to display. If got torch.Tensor then convert it
        to np.ndarray with axes permutation.
    ax : Optional[plt.Axes], optional
        Axes for image showing. If not given then a new Figure and Axes
        will be created.
    Returns
    -------
    plt.Axes
        Axes with showed image.
    """
    if isinstance(img, torch.Tensor):
        img = img.clone().detach().cpu().permute(1, 2, 0).numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img)
    return ax


def main():
    path = Path(__file__).parents[1] / 'data' / 'satellite_small' / 'train'
    dset = RegionsDataset(path)
    for _ in range(10):
        img = dset[random.randint(0, len(dset))]

        img_size = img.shape[1:]
        reg_size = (112, 112)

        reg_get = RegionGetting(img_size, reg_size)
        regions = reg_get(img)

        display_image(img)

        fig, axs = plt.subplots(1, 2)
        for i, region in enumerate(regions):
            axs[i] = display_image(region, axs[i])

        plt.show()

if __name__ == '__main__':
    main()
