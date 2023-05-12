from typing import Optional, Union, List, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def read_image(path: Union[Path, str], grayscale: bool = False) -> np.ndarray:
    """
    Read image to numpy array.
    Parameters
    ----------
    path : Union[Path, str]
        Path to image file
    grayscale : bool, optional
        Whether read image in grayscale, by default False
    Returns
    -------
    np.ndarray
        Array containing read image.
    Raises
    ------
    FileNotFoundError
        Did not find image.
    ValueError
        Image reading is not correct.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Did not find image {path}.')
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError('Image reading is not correct.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_scaled_shape(
    orig_h: int, orig_w: int, orig_scale: float,
    overlap_step: float, fov: float, net_size: Optional[int] = 112
) -> Tuple[int, int, int, float]:
    """
    Масштабирование размеров изображения, исходя из размеров поля зрения
    и входа сети.
    Конвертация поля зрения и шага перекрывающего окна из метров в пиксели
    при новом размере изображения.

    Args:
        orig_h (int): Исходная высота изображения в пикселях.
        orig_w (int): Исходная ширина изображения в пикселях.
        orig_scale (float): Исходный масштаб изображения (метров на пиксель).
        overlap_step (float): перекрывающий шаг в метрах.
        fov (float): Размер стороны поля зрения в метрах.
        net_size (Optional[int], optional): Размер входа сети.

    Returns:
        Tuple[int, int, int, int]: Новый размеры изображения,
        размер перекрывающего шага в пикселях и новый масштаб.
    """    
    fov_px = fov / orig_scale  # Поле зрения в пикселях
    # Коэффициент масштабирования для привидения размеров участка
    # к размеру входа сети
    resize_coef = fov_px / net_size
    # Новый масштаб
    new_scale = orig_scale * resize_coef

    # Отмасштабированный шаг перекрывающего окна в пикселях
    scaled_overlap_px = int(overlap_step / new_scale)  # 16 пикселей для 30 метров
    # Новые размеры изображения
    h = int(orig_h / resize_coef)  # 
    w = int(orig_w / resize_coef)
    return h, w, scaled_overlap_px, new_scale


def resize_image(
    image: np.ndarray,
    new_size: Tuple[int, int],
    interpolation: Optional[int] = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to given size.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    new_size : Tuple[int, int]
        Tuple containing new image size.

    Returns
    -------
    np.ndarray
        Resized image
    """
    return cv2.resize(
        image, new_size, None, None, None, interpolation=interpolation)


def get_sliding_windows(
    source_image: np.ndarray,
    h_win: int,
    w_win: int,
    stride: Optional[int] = None
) -> np.ndarray:
    """
    Cut a given image into windows with defined shapes and stride.

    Args:
        source_image (np.ndarray): The original image.
        h_win (int): Height of the windows.
        w_win (int): Width of the windows.
        stride (Optional[int]): The stride of the sliding windows.
        If not defined it will be set by w_win value.

    Returns:
        np.ndarray: The cut image with shape `[num_windows, h_win, w_win, c]`.
    """    
    w, h, c = source_image.shape

    if stride is None:
        stride = w_win

    x_indexer = (
        np.expand_dims(np.arange(w_win), 0) +
        np.expand_dims(np.arange(w - w_win - 1, step=stride), 0).T
    )
    y_indexer = (
        np.expand_dims(np.arange(h_win), 0) +
        np.expand_dims(np.arange(h - h_win - 1, step=stride), 0).T
    )
    windows = source_image[x_indexer][:, :, y_indexer].swapaxes(1, 2)
    windows = windows.reshape(-1, w_win, h_win, c)
    return windows


def load_images(image_paths: List[Path]) -> torch.Tensor:
    """
    Load and prepare images to pass into a network.

    Args:
        image_paths (List[Path]): Paths to the needed images.

    Returns:
        torch.Tensor: Loaded and processed images with shape
        `[num_images, c, h, w].`
    """    
    images = []
    for img in image_paths:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        images.append(img)
    return torch.cat(images)


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by given angle in degrees.

    Args:
        img (np.ndarray): The image to rotate with shape `[h, w, c]`.
        angle (float): The angle of rotating.

    Returns:
        np.ndarray: The rotated image with shape `[h, w, c]`.
    """    
    h, w, _ = img.shape

    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1)
    dst = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    return dst


def process_raw_real_image(
    image: np.ndarray,
    angle: Optional[float] = 32.0,
    white_space: Optional[float] = 0.15
) -> np.ndarray:
    """
    Given real images are rotated by angle about 32 degrees, and have white
    empty space around image.
    Rotate the given image and cut white space.

    Args:
        image (np.ndarray): The image to process.
        angle (Optional[float], optional): The angle of rotating.
        white_space (Optional[float], optional): A percent of white space.

    Returns:
        np.ndarray: The processed image.
    """
    h, w, _ = image.shape
    rotated_img = rotate_img(image, angle)
    cut_img = rotated_img[int(h * white_space):h - int(h * white_space),
                          int(w * white_space):w - int(w * white_space)]
    return cut_img


def show_grid(
    arr: np.ndarray,
    h: int,
    w: int,
    size: Tuple[float, float] = (20.0, 20.0)
) -> Tuple[Figure, plt.Axes]:
    """
    Show a batch of images on a grid.

    Args:
        arr (np.ndarray): The batch of the images with shape `[b, h, w, c]`.
        h (int): A number of images in one column of the grid.
        w (int): A number of images in one string of the grid.
        size (Tuple[float, float], optional): A size of plt figure.

    Returns:
        Tuple[Figure, plt.Axes]: The figure and axes with showed images.
    """    
    fig, axs = plt.subplots(h, w)
    fig.set_size_inches(*size, forward=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(arr.shape[0]):
        row = i // w
        column = i % w
        axs[row][column].get_yaxis().set_visible(False)
        axs[row][column].get_xaxis().set_visible(False)
        axs[row][column].imshow(arr[i])
    return axs