import numpy as np
import PIL.ImageOps
import PIL.Image
import torch

# TO DO: define transform to apply to data when loading --> each transform = one class
# mainly the data augmentation

class HistEqualization:
    """

    """
    def __init__(self):
        """

        """

    def __call__(self, image):
        """

        """
        return PIL.ImageOps.equalize(image)

class AutoContrast:
    """

    """
    def __init__(self, cutoff=0):
        """

        """
        self.cutoff = cutoff

    def __call__(self, image):
        """

        """
        return PIL.ImageOps.autocontrast(image, cutoff=self.cutoff)

class ResizeMax:
    """

    """
    def __init__(self, max_len=512):
        """

        """
        self.max_len = max_len

    def __call__(self, image):
        """

        """
        s = image.size
        max_dim, min_dim = np.argmax(s), np.argmin(s)
        aspect_ratio = s[max_dim]/s[min_dim]
        new_s = list(s)
        new_s[max_dim], new_s[min_dim] = self.max_len, int(self.max_len/aspect_ratio)
        return image.resize(new_s, PIL.Image.ANTIALIAS)

class PadToSquare:
    """

    """
    def __init__(self):
        """

        """

    def __call__(self, image):
        """

        """
        s = list(image.size)
        max_len = max(s)
        pad_w = max_len - s[0]
        pad_h = max_len - s[1]
        padding = (pad_w//2, pad_h//2, pad_w-(pad_w//2), pad_h-(pad_h//2))
        return PIL.ImageOps.expand(image, padding, fill=0)

class MinMaxNormalization:
    """

    """
    def __init__(self, vmin=0, vmax=1):
        """

        """
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, image):
        """

        """
        arr = np.array(image).astype('float')
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (self.vmax - self.vmin) * arr + self.vmin
        return arr
