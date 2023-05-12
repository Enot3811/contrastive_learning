from typing import Tuple, Callable, List
from pathlib import Path

import cv2
import torch


class RegionGetting:  
    """
    Селектор регионов.

    Позволяет брать регионы с одного изображения без повторов и с минимальным
    заданным расстоянием между друг другом.
    """    
    def __init__(
        self,
        image_size: Tuple[int, int],
        region_size: Tuple[int, int],
        stride: int = None,
        regions_per_image: int = 2,
        region_margin: int = 1
    ):
        """
        Инициализация селектора.

        Args:
            image_size (Tuple[int, int]): Размер изображений, из которых будут
            браться регионы.
            region_size (Tuple[int, int]): Размер регионов, которые будут
            извлекаться.
            stride (int, optional): Шаг окна региона.
            regions_per_image (int, optional): Сколько регионов необходимо
            выбрать с одного изображения.
            region_margin (int, optional): Минимальное расстояние в регионах
            между выбираемыми регионами.
        """        
        self.region_margin = region_margin
        self.regions_per_image = regions_per_image
        self.image_size = image_size

        h, w = image_size
        w_reg, h_reg = region_size

        if stride is None:
            stride = w_reg
        
        # На основе размеров изображений, окон и шага окна создаются
        # индексаторы, обратившись по индексам к которым можно получить
        # индексы для определённого окна
        self.x_indexer = (
            torch.arange(0, w_reg)[None, :] +
            torch.arange(0, w - w_reg + 1, stride)[None, :].T
        )
        self.y_indexer = (
            torch.arange(0, h_reg)[None, :] +
            torch.arange(0, h - h_reg + 1, stride)[None, :].T
        )
        # По сути сколько окон, столько и элементов в индексаторе
        self.x_windows = self.x_indexer.size(0)
        self.y_windows = self.y_indexer.size(0)

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Выбрать регионы из изображения.

        Args:
            img (torch.Tensor): Изображение.

        Returns:
            List[torch.Tensor]: Список тензоров с выбранными регионами.
        """        
        h, w = self.image_size
        
        # Регионы будут браться исходя из логической матрицы
        available_regions = torch.ones(
            self.x_windows, self.y_windows, dtype=torch.bool)
        
        gotten_regions = []
        while len(gotten_regions) != self.regions_per_image:
            x_idx = torch.randint(0, self.x_windows, ())
            y_idx = torch.randint(0, self.y_windows, ())

            # Если подобранный регион можно взять
            if available_regions[x_idx, y_idx]:
                x_slice = self.x_indexer[x_idx]
                y_slice = self.y_indexer[y_idx]
                # Окно берётся
                gotten_regions.append(img[:, y_slice][:, :, x_slice])
                # В логической матрице блокируется данный регион и соседи
                # на расстоянии margin регионов
                _start_bl = max(0, x_idx - self.region_margin)
                _end_bl = min(x_idx + self.region_margin + 1, w)
                x_blocking = slice(_start_bl, _end_bl)
                _start_bl = max(0, y_idx - self.region_margin)
                _end_bl = min(y_idx + self.region_margin + 1, h)
                y_blocking = slice(_start_bl, _end_bl)
                available_regions[y_blocking, x_blocking] = False
                # print(available_regions)
            # print(x_idx * 112, y_idx * 112)
        return gotten_regions


class RegionsDataset(torch.utils.data.Dataset):
    """
    Датасет возвращающий изображения регионов из указанной директории.
    """    

    def __init__(
        self,
        image_directory: Path,
        processing: Callable = None,
        **kwargs
    ):
        """
        Инициализация датасета.

        Args:
            image_directory (Path): Директория с изображениями.
            processing (Callable, optional): Функции предобработки данных.
        """        
        self.images = list(map(str, image_directory.rglob('*.jpg')))
        self.processing = processing
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Возвращает изображение из датасета по индексу.

        Args:
            idx (int): Индекс изображения.

        Returns:
            torch.Tensor: Тензор со считанным изображением.
        """        
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).permute(2, 0, 1)

        if self.processing is not None:
            img = self.processing(img)
        return img