from pathlib import Path

import torch
import torch.optim as optim
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
    device = torch.device('cpu')

    dset = RegionsDataset(path)
    normalize = transforms.Normalize(mean=mean, std=std)

    # Аугментации
    contrast_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.6, 1.4),
                                                       contrast=(0.5, 1.5),
                                                       saturation=(0.7, 1.3),
                                                       hue=0.05)
                                ], p=0.9),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)],
                               p=0.2)
    ])

    transformer = ContrastiveTransformations(contrast_transforms)

    region_selector = RegionGetting(
        image_size, reg_size, stride, batch_size, margin)

    d_loader = DataLoader(dset, shuffle=True)

    model = Encoder(hidden_dim)
    model.to(device=device)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(epochs):
        # Проход по train датасету
        model.train()
        for image in d_loader:
            regions = region_selector(image)
            regions = torch.concat(regions)
            augmented_regions = transformer(regions)
            augmented_regions = torch.concat(augmented_regions)
            augmented_regions.to(device=device)
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
            loss = info_nce_loss(feats, temperature)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # TODO Валидация
        # with torch.no_grad():
        #     model.eval()
        #     for image in val_d_loader:
        #         pass


if __name__ == '__main__':
    main()
