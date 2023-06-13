from typing import Union, Tuple
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


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


def save_image(img: np.ndarray, path: Union[Path, str]) -> None:
    """Сохранить переданное изображение по указанному пути.

    Parameters
    ----------
    img : np.ndarray
        Сохраняемое изображение.
    path : Union[Path, str]
        Путь для сохранения изображения.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('Не удалось сохранить изображение.')


def get_scaled_shape(
    orig_h: int, orig_w: int, orig_scale: float,
    overlap_step: float, fov: float, net_size: int = 112
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
    scaled_overlap_px = int(overlap_step / new_scale)
    # Новые размеры изображения
    h = int(orig_h / resize_coef)
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
        image, new_size[::-1], None, None, None, interpolation=interpolation)


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
        The cut image with shape `[n_h_win, n_w_win, h_win, w_win, c]`.
    """
    h, w, c = source_image.shape

    if stride is None:
        stride = w_win

    x_indexer = (
        np.expand_dims(np.arange(w_win), 0) +
        np.expand_dims(np.arange(w - w_win, step=stride), 0).T
    )
    y_indexer = (
        np.expand_dims(np.arange(h_win), 0) +
        np.expand_dims(np.arange(h - h_win, step=stride), 0).T
    )
    windows = source_image[y_indexer][:, :, x_indexer].swapaxes(1, 2)
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
    img = img.copy()
    h, w, _ = img.shape

    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1)
    dst = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    return dst


def process_raw_real_image(
    image: np.ndarray,
    angle: float = 32.0,
    white_space: float = 0.15
) -> np.ndarray:
    """Повернуть изображение и обрезать белые край.

    Исходное изображение повёрнуто на угол около 32 градусов и имеет белое
    пустое пространство по краям. Данная функция преобразует изображение к
    нормальному виду.

    Parameters
    ----------
    image : np.ndarray:
        Исходное изображение для обработки.
    angle : float, optional
        Угол поворота. По умолчанию 32.
    white_space : float, optional
        Доля белого пространства в размерах после поворота. По умолчанию 0.15.

    Returns
    -------
        np.ndarray: Обработанное изображение.
    """
    image = image.copy()
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
    """Show a batch of images on a grid.

    Parameters
    ----------
    arr : np.ndarray
        The batch of the images with shape `[b, h_img, w_img, c]`.
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
        if h == 1 or w == 1:
            axs[i].get_yaxis().set_visible(False)
            axs[i].get_xaxis().set_visible(False)
            axs[i].imshow(arr[i])
        else:
            row = i // w
            column = i % w
            axs[row][column].get_yaxis().set_visible(False)
            axs[row][column].get_xaxis().set_visible(False)
            axs[row][column].imshow(arr[i])
    return axs


def figure_to_ndarray(fig: Figure) -> np.ndarray:
    """Конвертирует `figure` в изображение в виде `ndarray`.

    Parameters
    ----------
    fig : Figure
        Фигура, которую необходимо конвертировать.

    Returns
    -------
    np.ndarray
        Изображение из фигуры.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # draw the canvas, cache the renderer
    image_flat = np.frombuffer(
        canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*fig.canvas.get_width_height(), 3)  # (H, W, 3)
    return image


def overlay_images(
    img1: np.ndarray,
    img2: np.ndarray,
    img_size: Tuple[int, int] = None
) -> np.ndarray:
    """Наложить одно изображение на другое.

    Parameters
    ----------
    img1 : np.ndarray
        Первое изображение.
    img2 : np.ndarray
        Второе изображение.
    img_size : Tuple[int, int], optional
        Размеры итогового изображения.
        Если не заданы, то берутся размеры первого изображения `img1`.

    Returns
    -------
    np.ndarray
        Наложенное изображение.
    """
    if img_size is not None:
        # Переворачиваем размеры для cv2
        img_size = img_size[::-1]
        img1 = cv2.resize(img1, dsize=img_size)
    else:
        # Переворачиваем размеры для cv2
        img_size = img1.shape[1::-1]
    img2 = cv2.resize(img2, dsize=img_size)

    overlay_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
    return overlay_img


def draw_windows_grid(
    image: np.ndarray, win_size: int, stride: int,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1
) -> np.ndarray:
    """Нарисовать сетку из перекрывающих окон на изображении.

    Parameters
    ----------
    image : np.ndarray
        Исходное изображение.
    win_size : int
        Размер окна.
    stride : int
        Шаг окна.
    color : Tuple[int, int, int], optional
        Цвет рисуемых рамок. По умолчанию - красный.
    thickness : int, optional
        Толщина рисуемых рамок. По умолчанию - 1.

    Returns
    -------
    np.ndarray
        Отредактированная картинка.
    """
    image = image.copy()  # Копировать изображение, чтобы не испортить исходник
    h, w = image.shape[:2]
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            image = cv2.rectangle(
                image, (j, i), (j + win_size, i + win_size),
                color, thickness)
    return image
