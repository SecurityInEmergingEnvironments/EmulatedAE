from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.preprocessor.jpeg_compression import JpegCompression
from art.defences.preprocessor.variance_minimization import TotalVarMin
from art.config import ART_NUMPY_DTYPE

from tqdm.auto import tqdm
from typing import Optional, Tuple, TYPE_CHECKING, Any
from io import BytesIO
import numpy as np
import tensorflow as tf

class EAE(Preprocessor):
  def __init__(self, params):
    super().__init__(is_fitted=True, apply_fit=False, apply_predict=True)
    self.intermediary_size = params['intermediary_size']
    self.resizeMethod = params['resizeMethod']
    self.resolution = params['resolution']
    self._check_params()
  
  def _check_params(self):
    if self.intermediary_size < 0 or not type(self.intermediary_size) == int:
      raise ValueError("intermediary_size must be a positive integer")
  
  # override __call__ method
  def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    denoised_x = (tf.image.resize(tf.image.resize(x, (self.intermediary_size, self.intermediary_size), method = self.resizeMethod), (self.resolution[0], self.resolution[1]), method = self.resizeMethod))
    return denoised_x, y

class MyJpegCompression(JpegCompression):
  # TODO
  def __init__(
        self,
        clip_values: "CLIP_VALUES_TYPE",
        quality: int = 50,
        channels_first: bool = False,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
    ):
    super().__init__(clip_values=clip_values, channels_first = channels_first, apply_fit=apply_fit, apply_predict=apply_predict, verbose = verbose)
  
  def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply JPEG compression to sample `x`.
    For input images or videos with 3 color channels the compression is applied in mode `RGB`
    (3x8-bit pixels, true color), for all other numbers of channels the compression is applied for each channel with
    mode `L` (8-bit pixels, black and white).
    :param x: Sample to compress with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`. `x` values are expected to be in
              the data range [0, 1] or [0, 255].
    :param y: Labels of the sample `x`. This function does not affect them in any way.
    :return: compressed sample.
    """
    x_ndim = x.ndim
    if x_ndim not in [4, 5]:
        raise ValueError(
            "Unrecognized input dimension. JPEG compression can only be applied to image and video data."
        )

    # Swap channel index
    if self.channels_first and x_ndim == 4:
        # image shape NCHW to NHWC
        x = np.transpose(x, (0, 2, 3, 1))
    elif self.channels_first and x_ndim == 5:
        # video shape NCFHW to NFHWC
        x = np.transpose(x, (0, 2, 3, 4, 1))

    # insert temporal dimension to image data
    if x_ndim == 4:
        x = np.expand_dims(x, axis=1)

    # Convert into uint8
    if self.clip_values[1] == 1.0:
        x = x * 255
    x = x.astype("uint8")

    # Compress one image at a time
    x_jpeg = x.copy()
    for idx in tqdm(np.ndindex(x.shape[:2]), desc="JPEG compression", disable=not self.verbose):
        if x.shape[-1] == 3:
            x_jpeg[idx] = self._compress(x[idx], mode="RGB")
        else:
            for i_channel in range(x.shape[-1]):
                x_channel = x[idx[0], idx[1], ..., i_channel]
                x_channel = self._compress(x_channel, mode="L")
                x_jpeg[idx[0], idx[1], :, :, i_channel] = x_channel

    # Convert to ART dtype
    if self.clip_values[1] == 1.0:
        x_jpeg = x_jpeg / 255.0
    x_jpeg = x_jpeg.astype(ART_NUMPY_DTYPE)

    # remove temporal dimension for image data
    if x_ndim == 4:
        x_jpeg = np.squeeze(x_jpeg, axis=1)

    # Swap channel index
    if self.channels_first and x_jpeg.ndim == 4:
        # image shape NHWC to NCHW
        x_jpeg = np.transpose(x_jpeg, (0, 3, 1, 2))
    elif self.channels_first and x_ndim == 5:
        # video shape NFHWC to NCFHW
        x_jpeg = np.transpose(x_jpeg, (0, 4, 1, 2, 3))
    return x_jpeg, y