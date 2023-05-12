from pathlib import Path
import random

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataset import RegionGetting, RegionsDataset, ContrastiveTransformations
from image_tools import display_image


def main():
    # Создаём датасет с разными спутниковыми снимками
    path = Path(__file__).parents[1] / 'data' / 'satellite_small' / 'train'
    dset = RegionsDataset(path)

    #tensor([104.0949,  96.6685,  71.8058]) tensor([32.4278, 25.7516, 22.6969])
    # mean, std = dset.calculate_mean_std()
    # [0.40821529 0.37909216 0.28159137] [0.12716784 0.10098667 0.08900745]
    # print(mean / 255, std / 255)
    # Параметры для нормализации изображения
    mean = [0.40821529, 0.37909216, 0.28159137]
    std = [0.12716784, 0.10098667, 0.08900745]
    normalize = transforms.Normalize(mean = mean, std = std)

    for _ in range(2):
        # Считываем какой-нибудь снимок
        img = dset[random.randint(0, len(dset))]
        img = img[:, 0:560, 0:560]

        img_size = img.shape[1:]
        reg_size = (112, 112)

        # Берём 2 случайных региона
        reg_get = RegionGetting(img_size, reg_size, stride=56)
        regions = reg_get(img)

        # Исходный снимок
        display_image(img)

        # 2 выбранных региона
        fig, axs = plt.subplots(1, 2)
        for i, region in enumerate(regions):
            axs[i] = display_image(region, axs[i])

        # Преобразователь для contrastive
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

        # Отображаем 2 региона аугментированные разным образом
        for region in regions:
            region = region
            fig, axs = plt.subplots(1, 2)
            augmented_regs = transformer(region)
            for i in range(2):
                augmented_regs[i] = augmented_regs[i] / 255.0
                axs[i] = display_image(augmented_regs[i], axs[i])
                # Нормализуем после отображения,
                # чтобы картинка нормально смотрелась
                augmented_regs[i] = normalize(augmented_regs[i])


        plt.show()

if __name__ == '__main__':
    main()
