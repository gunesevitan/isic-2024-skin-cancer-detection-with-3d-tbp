import numpy as np
import cv2


def lab_to_xyz(l, a, b):

    """
    Convert LAB color space to XYZ

    Parameters
    ----------
    l: numpy.ndarray of shape (n_samples)
        L values

    a: numpy.ndarray of shape (n_samples)
        A values

    b: numpy.ndarray of shape (n_samples)
        B values

    Returns
    -------
    x: numpy.ndarray of shape (n_samples)
        X values

    y: numpy.ndarray of shape (n_samples)
        Y values

    z: numpy.ndarray of shape (n_samples)
        Z values
    """

    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    y = np.where(y ** 3 > 0.008856, y ** 3, (y - 16 / 116) / 7.787)
    x = np.where(x ** 3 > 0.008856, x ** 3, (x - 16 / 116) / 7.787)
    z = np.where(z ** 3 > 0.008856, z ** 3, (z - 16 / 116) / 7.787)

    x *= 95.047
    y *= 100.000
    z *= 108.883

    return x, y, z


def xyz_to_rgb(x, y, z):

    """
    Convert XYZ color space to RGB

    Parameters
    ----------
    x: numpy.ndarray of shape (n_samples)
        X values

    y: numpy.ndarray of shape (n_samples)
        Y values

    z: numpy.ndarray of shape (n_samples)
        Z values

    Returns
    -------
    r: numpy.ndarray of shape (n_samples)
        R values

    g: numpy.ndarray of shape (n_samples)
        G values

    b: numpy.ndarray of shape (n_samples)
        B values
    """

    x = x / 100.0
    y = y / 100.0
    z = z / 100.0

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    r = np.where(r > 0.0031308, 1.055 * (r ** (1 / 2.4)) - 0.055, 12.92 * r)
    g = np.where(g > 0.0031308, 1.055 * (g ** (1 / 2.4)) - 0.055, 12.92 * g)
    b = np.where(b > 0.0031308, 1.055 * (b ** (1 / 2.4)) - 0.055, 12.92 * b)

    r = (np.clip(r, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(b, 0, 1) * 255).astype(np.uint8)

    return r, g, b


def rgb_to_hsv(r, g, b):

    """
    Convert RGB color space to HSV

    Parameters
    ----------
    r: numpy.ndarray of shape (n_samples)
        R values

    g: numpy.ndarray of shape (n_samples)
        G values

    b: numpy.ndarray of shape (n_samples)
        B values

    Returns
    -------
    h: numpy.ndarray of shape (n_samples)
        H values

    s: numpy.ndarray of shape (n_samples)
        S values

    v: numpy.ndarray of shape (n_samples)
        V values
    """

    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 1, 3)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    return h, s, v
