"""Utility functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, restoration, transform, color, data, feature, util
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

    Returns:
        list of corresponding channels
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
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for i in range(4):
        axes[i].imshow(img_lst[i], cmap_lst[i])
        axes[i].set_title(title_lst[i])
        axes[i].axis('off')

    plt.suptitle(title)

    # return color channels and color maps if using hed
    if color_space == 'hed':
        return img_lst[1:], cmap_lst[1:]
    return None


def show_with_cmap(img_lst, cmap_lst, title_lst):
    """Show a list of image with given color map

    Args:
        img_lst: list of images
        cmap_lst: list of color map
        title_lst: list of title for each image
    """
    assert len(img_lst) == len(cmap_lst)
    assert len(img_lst) == len(title_lst)

    N = len(img_lst)
    fig, axes = plt.subplots(1, N, figsize=(4*N, 4), sharex=True, sharey=True)
    for i in range(N):
        axes[i].imshow(img_lst[i], cmap_lst[i])
        axes[i].set_title(title_lst[i])
        axes[i].axis('off')
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


def add_noise(img, sigma=None):
    """Add Gaussian noise to image

    Args:
        img: image to add noise to
        sigma: standard deviation for gaussian noise. If None, will be set as
            standard deviation of img

    Returns:
        noised image. Out-of-range values are clipped
    """
    if sigma is None:
        sigma = img.std()
    gauss = np.random.normal(0, sigma, np.shape(img))
    noised = img + gauss
    return noised


def simple_denoise(noised, kernel_size=3):
    """Simple method to denoise a noised image, using Median filter

    Args:
        noised: noised image
        kernel_size: size of Median kernel

    Returns:
        denoised image
    """
    return signal.medfilt2d(noised, kernel_size)


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
    return signal.convolve2d(dummy, h, mode='valid').astype(np.uint8)


def simple_deblur(blurred_noised):
    """Deblur a blurred image with Wiener filter

    Args:
        blurred_noised: blurred image with noise

    Returns:
        Deblurred image
    """
    img = np.copy(blurred_noised)
    psf = np.ones((5, 5)) / 25
    img = signal.convolve2d(img, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deblurred = restoration.wiener(img, psf, 1100)
    return deblurred


def detect_cell_boundary(img, sel_cmap):
    """Detect cell boundary and overlay the results on images, using blob
    detection algorithms: Laplacian of Gaussian, Difference of Gaussian,
    and Determinant of Hessian

    Args:
        img: input image

    Returns:
        blobs_list: list of detected blobs using different methods, following
            this order: log, dog, and doh. Each item is a sublist of blobs,
            where each sub-list item is x, y, and radius of a blob
    """
    # dummy = np.copy(img)
    dummy = util.invert(img)

    # Laplacian of Gaussian
    blobs_log = feature.blob_log(dummy, max_sigma=30, num_sigma=10,
                                 threshold=0.01)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

    # Difference of Gaussian
    blobs_dog = feature.blob_dog(dummy, max_sigma=30, threshold=0.01)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

    # Determinant of Hessian
    blobs_doh = feature.blob_doh(dummy, max_sigma=30, threshold=0.00065)

    # concatenate all results
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['red', 'green', 'blue']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']

    # plot results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(3):
        ax[i].set_title(titles[i])
        ax[i].imshow(img, sel_cmap)
        for blob in blobs_list[i]:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=colors[i], linewidth=2, fill=False)
            ax[i].add_patch(c)
        ax[i].set_axis_off()
    plt.suptitle('Detect cell boundary using blob detection')

    # compute cell size
    print('Average cell size (in pixel^2):')
    for i in range(3):
        cell_size = 0.0
        for blob in blobs_list[i]:
            cell_size += np.pi * blob[2]**2
        if cell_size != 0.0:
            cell_size /= len(blobs_list[i])
        print('  - {}: {:.3f}'.format(titles[i], cell_size))
    pass
