from typing import Optional, Union, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def read_image(path: Union[Path, str], grayscale: bool = False) -> np.ndarray:
    """Read image to numpy array.

    Parameters
    ----------
    path : Union[Path, str]
        Path to image file
    grayscale : bool, optional
        Whether read image in grayscale, by default False.

    Raises
    ------
    FileNotFoundError
        Did not find image.
    ValueError
        Image reading is not correct.

    Returns
    -------
    np.ndarray
        Array containing read image.
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


def save_image(img: np.ndarray, path: Path) -> None:
    """Сохранить переданное изображение по указанному пути.

    Parameters
    ----------
    img : np.ndarray
        Сохраняемое изображение.
    path : Path
        Путь для сохранения изображения.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('Не удалось сохранить изображение.')


def get_scaled_shape(
    orig_h: int, orig_w: int, orig_scale: float,
    overlap_step: float, fov: float, net_size: Optional[int] = 112
) -> Tuple[int, int, int, float]:
    """
    Масштабирование размеров изображения, исходя из размеров поля зрения
    и входа сети.
    Конвертация поля зрения и шага перекрывающего окна из метров в пиксели
    при новом размере изображения.

    Parameters
    ----------
    orig_h : int
        Исходная высота изображения в пикселях.
    orig_w : int
        Исходная ширина изображения в пикселях.
    orig_scale : float
        Исходный масштаб изображения (метров на пиксель).
    overlap_step : float
        перекрывающий шаг в метрах.
    fov : float
        Размер стороны поля зрения в метрах.
    net_size : int, optional
        Размер входа сети. По умолчанию равен 112.

    Returns
    -------
    Tuple[int, int, int, int]
        Новый размеры изображения, размер перекрывающего шага в пикселях
        и новый масштаб.
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
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize an image to a given size.

    Parameters
    ----------
    image : np.ndarray
        The image to resize.
    new_size : Tuple[int, int]
        A Tuple containing the new image size.
    interpolation : int, optional
        An interpolation type, by default cv2.INTER_LINEAR.

    Returns
    -------
    np.ndarray
        The resized image.
    """
    return cv2.resize(
        image, new_size, None, None, None, interpolation=interpolation)


def get_sliding_windows(
    source_image: np.ndarray,
    h_win: int,
    w_win: int,
    stride: int = None
) -> np.ndarray:
    """Cut a given image into windows with defined shapes and stride.

    Parameters
    ----------
    source_image : np.ndarray
        The original image.
    h_win : int
        Height of the windows.
    w_win : int
        Width of the windows.
    stride : int, optional
        The stride of the sliding windows.
        If not defined it will be set by `w_win` value.

    Returns
    -------
    np.ndarray
        The cut image with shape `[num_windows, h_win, w_win, c]`.
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


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by given angle in degrees.

    Parameters
    ----------
    img : np.ndarray
        The image to rotate with shape `[h, w, c]`.
    angle : float
        The angle of rotating.

    Returns
    -------
    np.ndarray
        The rotated image with shape `[h, w, c]`.
    """
    h, w, _ = img.shape

    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1)
    dst = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    return dst


def show_grid(
    arr: np.ndarray,
    h: int,
    w: int,
    size: Tuple[float, float] = (20.0, 20.0)
) -> Tuple[Figure, plt.Axes]:
    """Show a batch of images on a grid.

    Parameters
    ----------
    arr : np.ndarray
        The batch of the images with shape `[b, h, w, c]`.
    h : int
        A number of images in one column of the grid.
    w : int
        A number of images in one string of the grid.
    size : Tuple[float, float], optional
        A size of plt figure.

    Returns
    -------
    Tuple[Figure, plt.Axes]
        The figure and axes with showed images.
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


def display_image(
    img: Union[torch.Tensor, np.ndarray],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Display an image on a matplotlib figure.

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


def normalize_image(
    img: Union[np.ndarray, torch.Tensor],
    max_values: Tuple[Union[int, float]] = None,
    min_values: Tuple[Union[int, float]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Нормализовать изображение в диапазон от 0 до 1.

    Parameters
    ----------
        img : Union[np.ndarray, torch.Tensor]
            Массив или тензор с изображением.
        max_values : Tuple[Union[int, float]], optional)
            Максимальные значения каналов изображения. Если не заданы,
            берутся максимальные значения из переданного изображения.
        min_values : Tuple[Union[int, float]], optional
            Минимальные значения каналов изображения. Если не заданы,
            берутся минимальные значения из переданного изображения.

    Raises
    ------
    TypeError
        Given image must be np.ndarray or torch.Tensor.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Нормализованное изображение в том же типе данных, в котором было дано.
    """
    if isinstance(img, torch.Tensor):
        if max_values is None:
            max_ch = img.amax(axis=(1, 2))
            min_ch = img.amin(axis=(1, 2))
        else:
            max_ch = torch.tensor(max_values, dtype=img.dtype)
            min_ch = torch.tensor(min_values, dtype=img.dtype)
        normalized = ((img - min_ch[:, None, None]) /
                      (max_ch - min_ch)[:, None, None])
        return torch.clip(normalized, 0.0, 1.0)
    elif isinstance(img, np.ndarray):
        if max_values is None:
            max_ch = img.max(axis=(0, 1))
            min_ch = img.min(axis=(0, 1))
        else:
            max_ch = np.array(max_values, dtype=img.dtype)
            min_ch = np.array(min_values)
        normalized = ((img - min_ch[None, None, :]) /
                      (max_ch - min_ch)[None, None, :])
        return np.clip(normalized, 0.0, 1.0)
    else:
        raise TypeError(
            'Given image must be np.ndarray or torch.Tensor but it has '
            f'{type(img)}')
