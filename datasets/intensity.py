import random

import PIL
import PIL.Image
import torch
import numpy as np
import torchvision.transforms.functional as TF



class Rescale:

    def __init__(self, scale=1 / 255.0):
        """
        Args:
            min_val ():
            max_val ():
        """
        self.scale = scale

    def __call__(self, frame):
        """
        Args:
            frame ():
        Returns:
        """
        if isinstance(frame, PIL.Image.Image):
            return np.asarray(frame) * self.scale
        elif isinstance(frame, np.ndarray):

            return frame * self.scale


class Brightness:
    """Change brightness of frame."""

    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_brightness(frame, brightness_factor=self.brightness)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_brightness(frame, brightness_factor=self.brightness)
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomBrightness:
    """Change randomly brightness of frame."""

    def __init__(self, abs_brightness=0.01):
        self.brightness = 1 + random.uniform(-abs(abs_brightness), abs(abs_brightness))

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_brightness(frame, brightness_factor=self.brightness)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_brightness(frame, brightness_factor=self.brightness)
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class Hue:
    """
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`."""

    def __init__(self, hue):
        assert abs(hue) <= 0.5, f'hue value is {hue}, it should be <=0.5'
        self.hue = hue

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_hue(frame, hue_factor=self.hue)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_hue(frame, hue_factor=self.hue)
        else:

            raise TypeError('Expected  PIL.Image or torch.Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomHue:
    """
    Change randomly hue value
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`."""

    def __init__(self, hue_range=0.5):
        assert abs(hue_range) <= 0.5, f'hue value is {hue_range}, it should be <=0.5'
        self.hue = random.uniform(0, 1) * hue_range

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_hue(frame, hue_factor=self.hue)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_hue(frame, hue_factor=self.hue)
        else:

            raise TypeError('Expected  PIL.Image or torch.Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class Contrast:
    """Change contrast value  of frame.
      contrast (float): 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    """

    def __init__(self, contrast=1):
        """
        Args:
            contrast ():
        """
        assert 0 <= contrast <= 2, f'contrast should be in the range of [0,2] , given value was {contrast}'
        self.contrast = contrast

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_contrast(frame, contrast_factor=self.contrast)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_contrast(frame, contrast_factor=self.contrast)
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomContrast:
    """Change randomly brightness of frame."""

    def __init__(self, contrast=1):
        """
        Args:
            contrast ():
        """
        # assert 0 <= contrast <= 2, f'contrast should be in the range of [0,2] , given value was {contrast}'
        self.contrast = random.uniform(0, 2)

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_contrast(frame, contrast_factor=self.contrast)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_contrast(frame, contrast_factor=self.contrast)
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class Saturation:
    """Change saturation value  of frame.
    """

    def __init__(self, saturation=1):
        """
        Args:
            saturation (float): 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        """
        assert 0 <= saturation <= 2, f'saturation should be in the range of [0,2] , given value was {saturation}'
        self.saturation = saturation

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_saturation(frame, saturation_factor=self.saturation)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_saturation(frame, saturation_factor=self.saturation)
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomSaturation:
    """Change  randomly saturation value  of frame.
    """

    def __init__(self, saturation=1):
        """
        Args:
            saturation (float): 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        """
        assert 0 <= saturation <= 2, f'saturation should be in the range of [0,2] , given value was {saturation}'
        self.saturation = random.uniform(0, 2)

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_saturation(frame, saturation_factor=self.saturation)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return TF.adjust_saturation(frame, saturation_factor=self.saturation)
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomColorAugment(object):
    def __init__(self, brightness=0.2, contrast=0.2, hue=0, saturation=0):
        """
        Args:
            brightness ():
            contrast ():
            hue ():
            saturation ():
        """

        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = 1

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = 1

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = 1

        if 0 <= hue <= 0.5:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = 0

        self.saturation = saturation_factor
        self.contrast = contrast_factor
        self.brightness = brightness_factor
        self.hue = hue_factor

    def __call__(self, frame):
        """
        Args:
            frame (PIL.Image or np.array or torch.Tensor):
        Returns:  frame (PIL.Image or np.array or torch.Tensor)
        """
        color_transforms = [Saturation(saturation=self.saturation), Hue(hue=self.hue),
                            Brightness(brightness=self.brightness), Contrast(contrast=self.contrast)]
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):

            for ct in color_transforms:
                frame = ct(frame)
            return frame
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            for ct in color_transforms:
                frame = ct(frame)
            return frame
        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))