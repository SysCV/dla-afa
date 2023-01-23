"""Standard Transform for AFA.

# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/transforms.py # pylint: disable=line-too-long

Source License
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all # pylint: disable=line-too-long
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
import random

import numpy as np
import torch
import torchvision.transforms as torch_tr
from PIL import Image, ImageEnhance
from skimage.filters import gaussian  # pylint: disable=no-name-in-module

from afa.utils.structures import AugmentType


class MaskToTensor:
    """Convert mask into tensor."""

    def __call__(
        self, img: Image.Image, blockout_predefined_area: bool = False
    ) -> torch.Tensor:
        """Call."""
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class RandomGaussianBlur:
    """Apply Gaussian Blur."""

    def __call__(self, img: Image.Image) -> Image.Image:
        """Call."""
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, channel_axis=-1)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


def _is_pil_image(img: Image.Image) -> bool:
    """Check whether input is PIL image."""
    return isinstance(img, Image.Image)


def adjust_brightness(
    img: Image.Image, brightness_factor: float
) -> Image.Image:
    """Adjust brightness of an Image.

    Args:
        img: PIL Image to be adjusted.
        brightness_factor:  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.

    Raises:
        TypeError: img should be PIL Image.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img: Image.Image, contrast_factor: float) -> Image.Image:
    """Adjust contrast of an Image.

    Args:
        img: PIL Image to be adjusted.
        contrast_factor: How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.

    Raises:
        TypeError: img should be PIL Image.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(
    img: Image.Image, saturation_factor: float
) -> Image.Image:
    """Adjust color saturation of an image.

    Args:
        img: PIL Image to be adjusted.
        saturation_factor:  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.

    Raises:
        TypeError: img should be PIL Image.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img: PIL Image to be adjusted.
        hue_factor:  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.

    Raises:
        ValueError: hue_factor is not in [-0.5, 0.5].
        TypeError: img should be PIL Image.
    """
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(f"hue_factor {hue_factor} is not in [-0.5, 0.5].")

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()

    np_h = np.array(h, dtype=np.uint8)  # type: ignore
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")

    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


class ColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        """Init."""
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(
        brightness: float, contrast: float, saturation: float, hue: float
    ) -> AugmentType:
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(
                max(0, 1 - brightness), 1 + brightness
            )
            transforms.append(
                torch_tr.Lambda(
                    lambda img: adjust_brightness(img, brightness_factor)
                )
            )

        if contrast > 0:
            contrast_factor = np.random.uniform(
                max(0, 1 - contrast), 1 + contrast
            )
            transforms.append(
                torch_tr.Lambda(
                    lambda img: adjust_contrast(img, contrast_factor)
                )
            )

        if saturation > 0:
            saturation_factor = np.random.uniform(
                max(0, 1 - saturation), 1 + saturation
            )
            transforms.append(
                torch_tr.Lambda(
                    lambda img: adjust_saturation(img, saturation_factor)
                )
            )

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_hue(img, hue_factor))
            )

        np.random.shuffle(transforms)
        transform = torch_tr.Compose(transforms)

        return transform  # type:ignore

    def __call__(self, img: Image.Image) -> Image.Image:
        """Call.

        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        return transform(img)
