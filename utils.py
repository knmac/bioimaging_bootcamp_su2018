"""Utility functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, restoration, transform, color, data, feature
from scipy import signal, ndimage
from matplotlib.colors import LinearSegmentedColormap


def load_image(fname):
    """Load one of the samples images and show different color channels

    Args:
        fname: filename (or path) of the image. Will load a default image from
            skimage immunohistochemistry if empty.

    Returns:
        img: loaded image
    """
    if fname is not None:
        assert os.path.exists(fname), '{} not found'.format(fname)
        img = io.imread(fname)
    else:
        img = data.immunohistochemistry()
    return img


def show_custom_channels(img, color_space, title=''):
    """Show image with color space of choice

    Args:
        img: the input image
        color_space: color space to show images, either `rgb` or `hed`
        title: title for the plot (optional)
    """
    assert color_space in ['rgb', 'hed'], \
            'Unsupported color space. Must be either `rgb` or `hed`'

    if color_space == 'rgb':
        # allocate memory for 3 channels
        r_chn = np.zeros(img.shape, dtype=np.uint8)
        g_chn = np.zeros(img.shape, dtype=np.uint8)
        b_chn = np.zeros(img.shape, dtype=np.uint8)

        # extract data from each channel
        r_chn[..., 0] = img[..., 0]
        g_chn[..., 1] = img[..., 1]
        b_chn[..., 2] = img[..., 2]

        img_lst = [img, r_chn, g_chn, b_chn]
        cmap_lst = [None, None, None, None]
        title_lst = ['Original image', 'R channel', 'G channel', 'B channel']
    elif color_space == 'hed':
        # Create an artificial color close to the orginal one
        cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
        cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                     'saddlebrown'])
        cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                                       'white'])

        hed = color.rgb2hed(img)

        img_lst = [img, hed[..., 0], hed[..., 1], hed[..., 2]]
        cmap_lst = [None, cmap_hema, cmap_eosin, cmap_dab]
        title_lst = ['Original image', 'Hematoxylin', 'Eosin', 'DAB']

    # plot the original image and its 3 channels
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    for i in range(4):
        axes[i].imshow(img_lst[i], cmap_lst[i])
        axes[i].set_title(title_lst[i])
        axes[i].axis('off')

    plt.suptitle(title)
    pass


def zoom_in(img, top, left, height=64, width=64):
    """Zoom in and show a small window to see triplet of color values

    Args:
        img: orginal image to zoom in
        top, left: top and left corners of the zoomed in window
        height, width: height and width of the zoomed in window

    Returns:
        zoomed in patch
    """
    H, W, C = img.shape
    assert top >= 0 and top+height <= H, 'invalid top: {}'.format(top)
    assert left >= 0 and left+width <= W, 'invalid left: {}'.format(left)
    return img[top:top+height, left:left+width, :]


def add_noise(img, sigma=5):
    """Add Gaussian noise to image

    Args:
        img: image to add noise to
        sigma: standard deviation for gaussian noise

    Returns:
        noised image. Out-of-range values are clipped
    """
    noised = np.copy(img).astype(float)
    gauss = np.random.normal(0, sigma, np.shape(img))

    # Additive Noise
    noised = np.clip(np.round(gauss + noised), 0, 255)
    return np.uint8(noised)


def simple_denoise(noised, kernel_size=3):
    """Simple method to denoise a noised image, using Median filter

    Args:
        noised: noised image
        kernel_size: size of Median kernel

    Returns:
        denoised image
    """
    denoised = []
    for i in range(3):
        denoised += [signal.medfilt2d(noised[..., i], kernel_size)]
    return np.dstack(denoised)


def blur(img, mode='box', block_size=3):
    """Blur an image with Gaussian filter

    Args:
        img: input image
        mode: blurring mode, either `box`, `gaussian`, or `motion`
        block_size: size of kernel

    Returns:
        Blurred image
    """
    assert mode in ['box', 'gaussian', 'motion']

    dummy = np.copy(img)
    if mode == 'box':
        h = np.ones((block_size, block_size)) / block_size ** 2
    elif mode == 'gaussian':
        h = signal.gaussian(block_size, block_size / 3).reshape(block_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
    elif mode == 'motion':
        h = np.eye(block_size) / block_size

    blurred = []
    for i in range(3):
        blurred += [signal.convolve2d(dummy[..., i], h, mode='valid')]
    return np.uint8(np.dstack(blurred))


def simple_deblur(blurred_noised):
    """Deblur a blurred image with Wiener filter

    Args:
        blurred_noised: blurred image with noise

    Returns:
        Deblurred image
    """
    img = color.rgb2gray(blurred_noised)
    psf = np.ones((5, 5)) / 25
    img = signal.convolve2d(img, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deblurred = restoration.wiener(img, psf, 1100)
    return np.dstack([deblurred]*3)


def detect_cell_boundary(img, sigma=3):
    """Detect cell boundary and overlay the results on images, using Canny

    Args:
        img: input image
        sigma: standard deviation for the Gausian kernel

    Returns:
        edges: binary image of the edge, shape of (H, W)
    """
    gray = color.rgb2gray(img)
    edges = feature.canny(gray, sigma=sigma).astype(np.uint8)
    return edges


def compute_cell_size():
    """compute the average size of a cell
    """
    return
