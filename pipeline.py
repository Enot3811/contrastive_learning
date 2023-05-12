from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import RegionGetting, RegionsDataset, ContrastiveTransformations
from simclr import Encoder, info_nce_loss


def main():
    path = Path(__file__).parents[1] / 'data' / 'satellite_small' / 'train'
    mean = [0.40821529, 0.37909216, 0.28159137]
    std = [0.12716784, 0.10098667, 0.08900745]
    reg_size = (224, 224)
    image_size = (2248, 2248)
    stride = 56
    margin = reg_size[0] // stride
    batch_size = 4
    epochs = 2
    hidden_dim = 128
    lr = 5e-4
    temperature = 0.07
    weight_decay = 1e-4

    dset = RegionsDataset(path)
    normalize = transforms.Normalize(mean = mean, std = std)

    contrast_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, 
                                        contrast=0.5, 
                                        saturation=0.5, 
                                        hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9)
        ])
    transformer = ContrastiveTransformations(contrast_transforms)

    region_selector = RegionGetting(
        image_size, reg_size, stride, batch_size, margin)

    d_loader = DataLoader(dset, shuffle=True)

    model = Encoder(hidden_dim)

    for e in range(epochs):
        for image in d_loader:
            regions = region_selector(image)
            regions = torch.concat(regions)
            augmented_regions = transformer(regions)
            augmented_regions = torch.concat(augmented_regions)
            augmented_regions = augmented_regions / 255

            # Отобразить
            # import matplotlib.pyplot as plt
            # from image_tools import show_grid
            # from torch_tools import tensor_to_numpy
            # augmented_regions_arr = tensor_to_numpy(augmented_regions)
            # show_grid(augmented_regions_arr, 2, 4)
            # plt.show()

            augmented_regions = normalize(augmented_regions)

            feats = model(augmented_regions)
            info_nce_loss(feats, temperature)




if __name__ == '__main__':
    main()
