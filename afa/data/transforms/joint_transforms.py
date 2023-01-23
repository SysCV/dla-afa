"""Joint Transform for AFA.

# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py # pylint: pylint: disable=line-too-long

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
# The above copyright notice and this permission notice shall be included in all
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
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


class RandomCrop:
    """Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.

    if image < crop_size:
        # slide crop within image, random offset
    else:
        # slide image within crop
    """

    def __init__(
        self,
        crop_size: List[int],
        ignore_index: int,
        nopad: bool = True,
    ):
        """Init."""
        self.size = crop_size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    @staticmethod
    def crop_in_image(
        centroid: Optional[List[int]],
        target_w: int,
        target_h: int,
        input_w: int,
        input_h: int,
        img: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """Cropping inside the image."""
        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = input_w - target_w
            max_y = input_h - target_h
            x1 = random.randint(c_x - target_w, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - target_h, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if input_w == target_w:
                x1 = 0
            else:
                x1 = random.randint(0, input_w - target_w)
            if input_h == target_h:
                y1 = 0
            else:
                y1 = random.randint(0, input_h - target_h)

        return img.crop((x1, y1, x1 + target_w, y1 + target_h)), mask.crop(
            (x1, y1, x1 + target_w, y1 + target_h)
        )

    def __call__(
        self,
        img: Image.Image,
        mask: Image.Image,
        centroid: Optional[List[int]] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """Call."""
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W
        target_h, target_w = self.size

        if w == target_w and h == target_h:
            return img, mask

        if self.nopad:
            # Shrink crop size if image < crop
            if target_h > h or target_w > w:
                shorter_side = min(w, h)
                target_h, target_w = shorter_side, shorter_side
        else:
            # Pad image if image < crop
            if target_h > h:
                pad_h = (target_h - h) // 2 + 1
            else:
                pad_h = 0
            if target_w > w:
                pad_w = (target_w - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(
                    mask, border=border, fill=self.ignore_index
                )
                w, h = img.size

        return self.crop_in_image(
            centroid, target_w, target_h, w, h, img, mask
        )


class RandomHorizontallyFlip:
    """Random horizontally flip class."""

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Call."""
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                Image.FLIP_LEFT_RIGHT
            )
        return img, mask


class Scale:
    """Scale image such that longer side is equal to required size."""

    def __init__(self, size: int) -> None:
        """Init."""
        self.size = size

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Call."""
        assert img.size == mask.size
        w, h = img.size

        if w > h:
            long_edge = w
        else:
            long_edge = h

        if long_edge == self.size:
            return img, mask

        scale = self.size / long_edge
        target_w = int(w * scale)
        target_h = int(h * scale)
        target_size = (target_w, target_h)

        return img.resize(target_size, Image.BILINEAR), mask.resize(
            target_size, Image.NEAREST
        )


class RandomRotate:
    """Random rotate class."""

    def __init__(self, angle: int) -> None:
        """Init."""
        self.angle = angle

    def __call__(
        self, image: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Call."""
        angle = random.randint(0, self.angle * 2) - self.angle
        w, h = image.size

        angle = random.randint(0, self.angle * 2) - self.angle

        if label is not None:
            label = self.pad_image("constant", label, h, h, w, w, value=255)
            label = label.rotate(angle, resample=Image.NEAREST)
            label = label.crop((w, h, w + w, h + h))

        image = self.pad_image("reflection", image, h, h, w, w)
        image = image.rotate(angle, resample=Image.BILINEAR)
        image = image.crop((w, h, w + w, h + h))
        return image, label

    def pad_reflection(
        self, image: Image.Image, top: int, bottom: int, left: int, right: int
    ) -> Image.Image:
        """Padding reflection."""
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return image
        h, w = image.shape[:2]
        next_top = next_bottom = next_left = next_right = 0
        if top > h - 1:
            next_top = top - h + 1
            top = h - 1
        if bottom > h - 1:
            next_bottom = bottom - h + 1
            bottom = h - 1
        if left > w - 1:
            next_left = left - w + 1
            left = w - 1
        if right > w - 1:
            next_right = right - w + 1
            right = w - 1
        new_shape = list(image.shape)
        new_shape[0] += top + bottom
        new_shape[1] += left + right
        new_image = np.empty(new_shape, dtype=image.dtype)
        new_image[top : top + h, left : left + w] = image
        new_image[:top, left : left + w] = image[top:0:-1, :]
        new_image[top + h :, left : left + w] = image[-1 : -bottom - 1 : -1, :]
        new_image[:, :left] = new_image[:, left * 2 : left : -1]
        new_image[:, left + w :] = new_image[
            :, -right - 1 : -right * 2 - 1 : -1
        ]
        return self.pad_reflection(
            new_image, next_top, next_bottom, next_left, next_right
        )

    @staticmethod
    def pad_constant(
        image: Image.Image,
        top: int,
        bottom: int,
        left: int,
        right: int,
        value: int,
    ) -> Image.Image:
        """Padding constant."""
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return image
        h, w = image.shape[:2]
        new_shape = list(image.shape)
        new_shape[0] += top + bottom
        new_shape[1] += left + right
        new_image = np.empty(new_shape, dtype=image.dtype)
        new_image.fill(value)
        new_image[top : top + h, left : left + w] = image
        return new_image

    def pad_image(
        self,
        mode: str,
        image: Image.Image,
        top: int,
        bottom: int,
        left: int,
        right: int,
        value: int = 0,
    ) -> Image.Image:
        """Padding rotated image."""
        if mode == "reflection":
            img = Image.fromarray(
                self.pad_reflection(
                    np.asarray(image), top, bottom, left, right
                )
            )
        elif mode == "constant":
            img = Image.fromarray(
                self.pad_constant(
                    np.asarray(image), top, bottom, left, right, value
                )
            )
        else:
            raise ValueError(f"Unknown mode {mode}")
        return img


class RandomSizeAndCrop:
    """Random resize and crop the image."""

    def __init__(
        self,
        crop_size: List[int],
        ignore_index: int,
        nopad: bool = True,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        pre_size: Optional[int] = None,
    ) -> None:
        """Init."""
        self.crop = RandomCrop(
            crop_size=crop_size, ignore_index=ignore_index, nopad=nopad
        )
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size

    def __call__(
        self,
        img: Image.Image,
        mask: Image.Image,
        centroid: Optional[List[int]] = None,
    ) -> Tuple[Image.Image, Image.Image, float]:
        """Call."""
        assert img.size == mask.size

        scale_amt = random.uniform(self.scale_min, self.scale_max)

        if self.pre_size is not None:
            in_w, in_h = img.size
            # find long edge
            if in_w > in_h:
                # long is width
                pre_scale = self.pre_size / in_w
            else:
                pre_scale = self.pre_size / in_h
            scale_amt *= pre_scale

        w, h = [int(i * scale_amt) for i in img.size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        resized_img, resized_mask = (
            img.resize((w, h), Image.BICUBIC),
            mask.resize((w, h), Image.NEAREST),
        )

        img, mask = self.crop(resized_img, resized_mask, centroid)
        return img, mask, scale_amt
