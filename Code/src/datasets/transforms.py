import numpy as np
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.Image
import torch
import torchvision.transforms as TF

class Grayscale:
    """
    Convert image to a Grayscale one. If a mask is provided, it's only passed.
    """
    def __init__(self):
        """
        Constructor of the grayscale transform.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """

    def __call__(self, image, mask=None):
        """
        Convert image to grayscale.
        ----------
        INPUT
            |---- image (PIL.Image) the image to convert.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (PIL.Image) the converted image.
            |---- mask (PIL.Image) the mask.
        """
        return PIL.ImageOps.grayscale(image), mask

    def __str__(self):
        """
        Transform printing format
        """
        return "Grayscale()"

class HistEqualization:
    """
    Apply an histogram equalization algorithm to the image, and passes the mask.
    """
    def __init__(self):
        """
        Constructor of the histogram equalization transform.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """

    def __call__(self, image, mask=None):
        """
        Apply the histogram equalization.
        ----------
        INPUT
            |---- image (PIL.Image) the image to transform.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (PIL.Image) the tranformed image.
            |---- mask (PIL.Image) the mask.
        """
        return PIL.ImageOps.equalize(image), mask

    def __str__(self):
        """
        Transform printing format
        """
        return "HistEqualization()"

class AutoContrast:
    """
    Apply an auto-contrast algorithm to the image, and passes the mask.
    """
    def __init__(self, cutoff=0):
        """
        Constructor of the autocontrast transform.
        ----------
        INPUT
            |---- cutoff (int) the upper and lower precentile to which the
            |           histogram will be saturated.
        OUTPUT
            |---- None
        """
        self.cutoff = cutoff

    def __call__(self, image, mask=None):
        """
        Apply the autocontrast algorithm.
        ----------
        INPUT
            |---- image (PIL.Image) the image to convert.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (PIL.Image) the converted image.
            |---- mask (PIL.Image) the mask.
        """
        return PIL.ImageOps.autocontrast(image, cutoff=self.cutoff), mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"AutoContrast(cutoff={self.cutoff})"

class ResizeMax:
    """
    Resize the image (and mask)'s larger dimension to the given max length by
    keeping the aspect ratio.
    """
    def __init__(self, max_len=512):
        """
        Constructor of the Resize to max transform.
        ----------
        INPUT
            |---- max_len (int) the major axis length of the resized image.
        OUTPUT
            |---- None
        """
        self.max_len = max_len

    def __call__(self, image, mask=None):
        """
        Resize the image (and mask)'s larger dimension to the max-len.
        ----------
        INPUT
            |---- image (PIL.Image) the image to resize.
            |---- mask (PIL.Image) the mask to resize.
        OUTPUT
            |---- image (PIL.Image) the resized image.
            |---- mask (PIL.Image) the resized mask.
        """
        s = image.size
        if s[0] != s[1]:
            max_dim, min_dim = np.argmax(s), np.argmin(s)
        else:
            max_dim, min_dim = 0, 1
        aspect_ratio = s[max_dim]/s[min_dim]
        new_s = list(s)
        new_s[max_dim], new_s[min_dim] = self.max_len, int(self.max_len/aspect_ratio)
        image = image.resize(new_s, PIL.Image.ANTIALIAS)
        if mask:
            mask = mask.resize(new_s, PIL.Image.ANTIALIAS)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"ResizeMax(max_len={self.max_len})"

class PadToSquare:
    """
    Pad the image (and mask) to a square, with value zero
    """
    def __init__(self):
        """
        Constructor of the padding to square transform.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """

    def __call__(self, image, mask=None):
        """
        Pad the image and mask to a square.
        ----------
        INPUT
            |---- image (PIL.Image) the image to pad.
            |---- mask (PIL.Image) the mask to pad.
        OUTPUT
            |---- image (PIL.Image) the padded image.
            |---- mask (PIL.Image) the padded mask.
        """
        s = list(image.size)
        max_len = max(s)
        pad_w = max_len - s[0]
        pad_h = max_len - s[1]
        padding = (pad_w//2, pad_h//2, pad_w-(pad_w//2), pad_h-(pad_h//2))
        image = PIL.ImageOps.expand(image, padding, fill=0)
        if mask:
            mask = PIL.ImageOps.expand(mask, padding, fill=0)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return "PadToSquare()"

class MinMaxNormalization:
    """
    Normalized (Min-Max) the image.
    """
    def __init__(self, vmin=0, vmax=1):
        """
        Constructor of the grayscale transform.
        ----------
        INPUT
            |---- vmin (float / int) the desired minimum value.
            |---- vmax (float / int) the desired maximum value.
        OUTPUT
            |---- None
        """
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, image, mask=None):
        """
        Apply a Min-Max Normalization to the image, and passes the a¨mask.
        ----------
        INPUT
            |---- image (PIL.Image) the image to normalize.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (np.array) the normalized image.
            |---- mask (np.array) the passed mask.
        """
        arr = np.array(image).astype('float')
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (self.vmax - self.vmin) * arr + self.vmin
        if mask:
            mask = np.array(mask)
        return arr, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"MinMaxNormalization(vmin={self.vmin}, vmax={self.vmax})"

class RandomHorizontalFlip:
    """
    Randomly flip horizontally the image (and mask).
    """
    def __init__(self, p=0.5):
        """
        Constructor of the random horizontal flip transform.
        ----------
        INPUT
            |---- p (float) between 0 and 1, the probability of flipping.
        OUTPUT
            |---- None
        """
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randmly flip horizontally the image (and mask in the same fashion).
        ----------
        INPUT
            |---- image (PIL.Image) the image to flip.
            |---- mask (PIL.Image) the mask to flip.
        OUTPUT
            |---- image (PIL.Image) the flipped image.
            |---- mask (PIL.Image) the flipped mask.
        """
        if np.random.random() > self.p:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            if mask:
                mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomHorizontalFlip(p={self.p})"

class RandomVerticalFlip:
    """
    Randomly flip vertically the image (and mask).
    """
    def __init__(self, p=0.5):
        """
        Constructor of the random vertical flip transform.
        ----------
        INPUT
            |---- p (float) between 0 and 1, the probability of flipping.
        OUTPUT
            |---- None
        """
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randmly flip vertically the image (and mask in the same fashion).
        ----------
        INPUT
            |---- image (PIL.Image) the image to flip.
            |---- mask (PIL.Image) the mask to flip.
        OUTPUT
            |---- image (PIL.Image) the flipped image.
            |---- mask (PIL.Image) the flipped mask.
        """
        if np.random.random() > self.p:
            image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            if mask:
                mask = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomVerticalFlip(p={self.p})"

class RandomRotation:
    """
    Randomly rotate the image (and mask).
    """
    def __init__(self, degree_range=(-20,20), resample=PIL.Image.BILINEAR):
        """
        Constructor of the random rotation transform.
        ----------
        INPUT
            |---- degree_range (tuple (min, max)) the degree range of possible
            |           rotations angles.
            |---- resample () the interpolation strategy to use.
        OUTPUT
            |---- None
        """
        self.degree_range = degree_range
        self.resample = resample

    def __call__(self, image, mask=None):
        """
        Randmly rotate the image (and mask in the same fashion).
        ----------
        INPUT
            |---- image (PIL.Image) the image to rotate.
            |---- mask (PIL.Image) the mask to rotate.
        OUTPUT
            |---- image (PIL.Image) the rotated image.
            |---- mask (PIL.Image) the rotated mask.
        """
        angle = np.random.randint(self.degree_range[0], self.degree_range[1])
        image = image.rotate(angle, resample=self.resample)
        if mask:
            mask = mask.rotate(angle, resample=self.resample)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomRotation(degree_range={self.degree_range})"

class RandomScaling:
    """
    Randomly scale the image (and mask).
    """
    def __init__(self, scale_range=(0.8,1.2), resample=PIL.Image.BILINEAR):
        """
        Constructor of the random scalling transform.
        ----------
        INPUT
            |---- scale_range (tuple (min, max)) the range of possible scalling factors.
            |---- resample () the interpolation strategy to use.
        OUTPUT
            |---- None
        """
        self.scale_range = scale_range
        self.resample = resample

    def __call__(self, image, mask=None):
        """
        Randmly scale the image (and mask in the same fashion).
        ----------
        INPUT
            |---- image (PIL.Image) the image to scale.
            |---- mask (PIL.Image) the mask to scale.
        OUTPUT
            |---- image (PIL.Image) the scalled image.
            |---- mask (PIL.Image) the scalled mask.
        """
        scale = self.scale_range[0] + np.random.random() * (self.scale_range[1] - self.scale_range[0])
        image = image.transform(image.size, method=PIL.Image.AFFINE, \
                                data=(scale, 0, 0, 0, scale, 0), resample=self.resample)
        if mask:
            mask = mask.transform(mask.size, method=PIL.Image.AFFINE, \
                                  data=(scale, 0, 0, 0, scale, 0), resample=self.resample)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomScaling(scale_range={self.scale_range})"

class RandomBrightness:
    """
    Randomly chang the image brightness (and passes the mask).
    """
    def __init__(self, upper=1.2, lower=0.8):
        """
        Constructor of the random rotation transform.
        ----------
        INPUT
            |---- upper (float) the upper brightness change limit.
            |---- lower (float) the lower brightness change limit.
        OUTPUT
            |---- None
        """
        self.upper = upper
        self.lower = lower

    def __call__(self, image, mask=None):
        """
        Randmly rotate the image (and mask in the same fashion).
        ----------
        INPUT
            |---- image (PIL.Image) the image to adjust.
            |---- mask (PIL.Image) the mask to pass.
        OUTPUT
            |---- image (PIL.Image) the adjusted image.
            |---- mask (PIL.Image) the passed mask.
        """
        factor = self.lower + np.random.random() * (self.upper - self.lower)
        enhancer = PIL.ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomBrightness(lower={self.lower}, upper={self.upper})"


class ToTorchTensor:
    """
    Convert the image (and mask) to a torch.Tensor.
    """
    def __init__(self):
        """
        Constructor of the grayscale transform.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """

    def __call__(self, image, mask=None):
        """
        Convert the image (PIL or np.array) to a torch.Tensor
        ----------
        INPUT
            |---- image (PIL.Image or np.array) the image to convert.
            |---- mask (PIL.Image or np.array) the mask to convert.
        OUTPUT
            |---- image (torch.Tensor) the converted image.
            |---- mask (torch.Tensor) the converted mask.
        """
        image = TF.ToTensor()(image)
        if mask is not None:
            mask = TF.ToTensor()(mask)
        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return "ToTorchTensor()"

class Compose:
    """
    Compose, in a sequential fashion, multiple transforms than can handle an
    image and a mask in their __call__.
    """
    def __init__(self, *transformations):
        """
        Constructor of the composition of transforms.
        ----------
        INPUT
            |---- *transformations (args) list of transforms to compose.
        OUTPUT
            |---- None
        """
        self.transformations = transformations

    def __call__(self, image, mask=None):
        """
        Passes the image (and mask) through all the transforms.
        ----------
        INPUT
            |---- image () the input image.
            |---- mask () the input mask.
        OUTPUT
            |---- image () the transformed image.
            |---- mask () the transformed mask.
        """
        for f in self.transformations:
            image, mask = f(image, mask)

        return image, mask

    def __str__(self):
        """
        Transform printing format
        """
        return '    ' + '\n -> '.join([str(t) for t in self.transformations])
