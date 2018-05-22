"""Class demo on Wed, May 30
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import skimage.io
import scipy.signal
import scipy.ndimage
import skimage.restoration
import skimage.transform
import skimage.color
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def load_image(fname):
    """Load one of the samples images and show different color channels

    Args:
        fname: filename (or path) of the image

    Returns:
        img: loaded image
    """
    assert os.path.exists(fname), '{} not found'.format(fname)
    img = skimage.io.imread(fname)
    return img


def show_rgb_channels(img, title=''):
    """Show and image with its three RGB channels

    Args:
        img: the input image
        title: title for the plot (optional)
    """
    # allocate memory for 3 channels
    r_chn = np.zeros(img.shape, dtype=np.uint8)
    g_chn = np.zeros(img.shape, dtype=np.uint8)
    b_chn = np.zeros(img.shape, dtype=np.uint8)

    # extract data from each channel
    r_chn[..., 0] = img[..., 0]
    g_chn[..., 1] = img[..., 1]
    b_chn[..., 2] = img[..., 2]

    # plot the original image and its 3 channels
    _, axes = plt.subplots(1, 4)
    axes[0].imshow(img)
    axes[1].imshow(r_chn)
    axes[2].imshow(g_chn)
    axes[3].imshow(b_chn)

    # set xlabel
    axes[0].set_xlabel('original')
    axes[1].set_xlabel('R channel')
    axes[2].set_xlabel('G channel')
    axes[3].set_xlabel('B channel')

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


def show_hed_channels(img, title=''):
    """Separate H&E color stain channels from the image

    Args:
        img: input image
        title: title for the plot (optional)

    Returns:
        hed: image converted from RGB to HED channels
    """
    # Create an artificial color close to the orginal one
    cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
    cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                 'saddlebrown'])
    cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                                   'white'])

    hed = skimage.color.rgb2hed(img)

    # plot the original image and its 3 channels
    _, axes = plt.subplots(1, 4)
    axes[0].imshow(img)
    axes[1].imshow(hed[..., 0], cmap=cmap_hema)
    axes[2].imshow(hed[..., 1], cmap=cmap_eosin)
    axes[3].imshow(hed[..., 2], cmap=cmap_dab)

    # set xlabel
    axes[0].set_xlabel('original')
    axes[1].set_xlabel('Hematoxylin')
    axes[2].set_xlabel('Eosin')
    axes[3].set_xlabel('DAB')

    plt.suptitle(title)
    return hed


def add_noise(img):
    """Add Gaussian noise to image

    Args:
        img: image to add noise to

    Returns:
        noised image. Out-of-range values are clipped
    """
    noise = np.random.normal(0, 10, img.shape)
    return (img + noise).astype(np.uint8)


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
        denoised += [scipy.signal.medfilt2d(noised[..., i], kernel_size)]
    return np.dstack(denoised)


def blur(img, sigma=1):
    """Blur an image with Gaussian filter

    Args:
        img: input image
        sigma: standard deviation for Gaussian kernel

    Returns:
        Blurred image
    """
    return scipy.ndimage.gaussian_filter(img, sigma=sigma)


def simple_deblur(blurred_noised):
    """Deblur a blurred image with Wiener filter

    Args:
        blurred_noised: blurred image with noise

    Returns:
        Deblurred image
    """
    psf = np.ones((5, 5)) / 25
    deblurred = []
    for i in range(3):
        img = blurred_noised[..., i]
        img = scipy.signal.convolve2d(img, psf, 'same')
        img += 0.1 * img.std() * np.random.standard_normal(img.shape)
        deblurred += [skimage.restoration.wiener(img, psf, 1100)]
    return np.dstack(deblurred)


def detect_cell_boundary(img):
    """Detect cell boundary and overlay the results on images
    """
    return


def compute_cell_size():
    """compute the average size of a cell
    """
    return


def main():
    """Main function
    """
    # Load one of these sample image, show different color channels.
    img = load_image(os.path.join('samples', 'IMG_6562.JPG'))
    # img = (skimage.transform.rescale(img, 0.5) * 255).astype(np.uint8)
    show_rgb_channels(img, title='Input image')

    # Zoom in and show a small window to see triplet of color values for a
    # 64x64 (or so) window
    zoomed = zoom_in(img, 1000, 1000)
    show_rgb_channels(zoomed, title='Zoomed-in window')

    # Separate H&E color stain channels from the image
    img_hed = show_hed_channels(img, title='Immunohistochemical staining '
                                           'colors separation')

    # Add noise and do a simple denoising task
    noised = add_noise(img)
    denoised = simple_denoise(noised, kernel_size=3)
    show_rgb_channels(noised, 'Image with Gaussian noise')
    show_rgb_channels(denoised, 'Image denoised with Median filter')

    # Apply blurring and add noise and do a simple deblurring task, using the
    # Wiener filter
    blurred_noised = add_noise(blur(img, sigma=3))
    deblurred = simple_deblur(blurred_noised)
    show_rgb_channels(blurred_noised, 'Blurred image with noise')
    show_rgb_channels(deblurred, 'Deblurred image')

    # Detect cell boundary and overlay the results on images

    # Compute the average size of a cell

    plt.show()
    pass


if __name__ == '__main__':
    main()
