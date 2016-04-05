import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import random
import scipy
from matplotlib import colors


def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def main():
    img = np.loadtxt('image.txt')
    length = img.shape[0]
    width = img.shape[1]
    x = range(length)
    y = range(width)
    max = img.max()
    min = img.min()
    plt.figure()
    plt.contourf(y, x, img,np.arange(min,max/5,(max/5-min)/10))

    plt.show()
    # Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
    img = ndimage.gaussian_filter(img, sigma=(0.7, 0.7), order=0)
    np.savetxt('img.txt', img)
    plt.contourf(y, x, img,np.arange(min,max/5,(max/5-min)/10))
    plt.show()

def generate_rand_arr(x, y, prob=0.5):
    rand = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if random.random() < prob:
                rand[i,j] = 1
    return rand


def plot():
    path = 'D:\\Documents\\MyDocuments\\Zare\\NAIMS\\Imaging\\01072016\\'
    img = np.loadtxt(path + 'image2.txt')
    length = img.shape[0]
    width = img.shape[1]
    x = range(length)
    y = range(width)
    rand = generate_rand_arr(length, width, 0.9)
    max = img.max()
    min = img.min()
    #img = scipy.ndimage.zoom(img, 3)
    #img *= rand
    #dft = np.fft.fft2(img)
    plt.figure()
    cmap = plt.cm.get_cmap('rainbow')
    fig = plt.contourf(img ,np.arange(50000,max/5,(max/5-50000)/10),cmap=cmap ,extend='both')
    cbar = plt.colorbar()
    plt.show()


plot()