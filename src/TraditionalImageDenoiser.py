
"""
This component is intended to be the non-neural networks-based image denoisers
"""
import os
from pathlib import Path
import tensorflow as tf
import json
import gc
import numpy as np

from art.defences.preprocessor.jpeg_compression import JpegCompression
from art.defences.preprocessor.variance_minimization import TotalVarMin
from skimage.restoration import denoise_nl_means, calibrate_denoiser

from tqdm.auto import tqdm
from art.config import ART_NUMPY_DTYPE
from typing import Optional, Tuple, TYPE_CHECKING, Any
class MyJpegCompression(JpegCompression):
  def __init__(
        self,
        quality,
        clip_values: "CLIP_VALUES_TYPE",
        channels_first: bool = False,
        apply_fit: bool = True,
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

class MyNonLocalDenoiser():
  # this denoiser is not scalable on standard resolution such as 224 x 224 x 3 for dataset such as ImageNet
  def __init__(self, resolution, patch_size=7, patch_distance=11, h=0.1,
    fast_mode=True, channel_axis=None):
    self.patch_size = patch_size
    self.patch_distance = patch_distance
    self.h = h
    self.fast_mode = fast_mode
    self.channel_axis = channel_axis
    self.resolution = resolution
  def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # this is impossible to run as it would take 2.857 HOURS to denoise 1000 images of size 224 x 224 x 3 as original func
    # this reduces to ~ 2.25 hours for 10,000 images or 810 seconds or 13.5 minutes per 1000 images
    small_x = tf.image.resize(x, (32, 32), method = 'bilinear')
    denoised_small_img = denoise_nl_means(
      image = small_x, 
      fast_mode = self.fast_mode, 
      patch_distance = self.patch_distance, 
      h = self.h,
      channel_axis = self.channel_axis)
    denoised_x = (tf.image.resize(denoised_small_img, (self.resolution[0], self.resolution[1]), method = 'bilinear'))
    return denoised_x, y

class MyTVDenoiser():
  # this denoiser is not scalable on standard resolution such as 224 x 224 x 3 for dataset such as ImageNet
  def __init__(self, resolution, prob = 0.3, solver = "L-BFGS-B"):
    self.tv = TotalVarMin(
      prob = prob,
      solver = solver
    )
    self.resolution = resolution
  def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # this is impossible to run as it would take ~ 3 HOURS to denoise 1000 images of size 224 x 224 x 3 as original func
    # this reduces to ~ 2 hours for 10,000 images or 720 seconds or 12 mins per 1000 images
    small_x = tf.image.resize(x, (32, 32), method = 'bilinear')
    denoised_small_img = self.tv(small_x)
    denoised_x = (tf.image.resize(denoised_small_img, (self.resolution[0], self.resolution[1]), method = 'bilinear'))
    return denoised_x, y

class MyNoise2Self():
  # this denoiser is not scalable on standard resolution such as 224 x 224 x 3 for dataset such as ImageNet
  def __init__(self, denoise_function_name, resolution, denoise_parameters,
  stride=4, approximate_loss=True, extra_output=False):
    self.denoise_function_name = denoise_function_name
    self.denoise_parameters = denoise_parameters
    self.stride = stride
    self.approximate_loss = approximate_loss
    self.extra_output = extra_output
    self.resolution = resolution
  
  def _e_ae(self, image, intermediary_size, method):
    return (tf.image.resize(tf.image.resize(image * 255, (intermediary_size, intermediary_size), method = method), (self.resolution[0],self.resolution[1]), method = method))
  
  def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # this is impossible to run as it would take 99.4 seconds to denoise 1000 images of size 224 x 224 x 3 as original func
    calibrated_denoiser = calibrate_denoiser(
      image = x,
      denoise_function = self._e_ae,
      denoise_parameters = self.denoise_parameters,
      stride = self.stride,
      approximate_loss = self.approximate_loss,
      extra_output = self.extra_output
    )
    return calibrated_denoiser(x), y

class TraditionalImageDenoiser:
  def __init__(self, name, kwargs, resolution, dataset_name, all_eval_paths, y_true_for_benign, y_true_for_adv, evaluateDenoiserFunc):
    self.kwargs = kwargs
    self.name = name
    self.dataset_name = dataset_name
    self.all_eval_paths = all_eval_paths
    self.y_true_for_benign = y_true_for_benign
    self.y_true_for_adv = y_true_for_adv
    self.evaluateDenoiserFunc = evaluateDenoiserFunc

  def train (self, ds_train, ds_test, resolution):
    # TODO: rename all "train" func to "trainAndEval"
    count = 0
    best_denoiser_avg_eval_score = 0
    best_denoiser = None
    best_eval_result, bestParam, bestModelSavedPath= {}, {}, ""
    
    if self.name == "emulated-ae":
      bestIntermediarySize = None
      total = len(self.kwargs['denoisers']['emulated-ae']['EAE_intermediary_sizes'])
      for intermediary_size in self.kwargs['denoisers']['emulated-ae']['EAE_intermediary_sizes']:
        denoiser = (lambda img : (tf.image.resize(tf.image.resize(img * 255, (intermediary_size,intermediary_size), method = "bilinear"), (resolution[0],resolution[1]), method = "bilinear")))
        denoiserParams = {"intermediary_size": intermediary_size, "method": "bilinear"}
        evaluationReportPath = self.kwargs['denoisers']['emulated-ae']['evaluationReportPath'] + "/intermediary_size_{}".format(intermediary_size)
        current_denoiser_avg_eval_score, evalResults = self._evaluate(
          evaluationReportPath = evaluationReportPath,
          denoiser = denoiser,
          denoiserParams = denoiserParams,
          ds_test = ds_test,
          _denoise_image_func = self._e_ae_denoise_image_func
        )
        if current_denoiser_avg_eval_score > best_denoiser_avg_eval_score:
          best_denoiser_avg_eval_score = current_denoiser_avg_eval_score
          best_eval_result = evalResults
          bestIntermediarySize = intermediary_size
          best_denoiser = denoiser
          # best_denoiser = (lambda img : (tf.image.resize(tf.image.resize(img * 255, (bestIntermediarySize,bestIntermediarySize), method = "bilinear"), (resolution[0],resolution[1]), method = "bilinear")))
          print("bestIntermediarySize thus far: {}, best_denoiser_avg_eval_score thus far: {}".format(bestIntermediarySize, best_denoiser_avg_eval_score))
        count += 1
        print("[TraditionalImageDenoiser] {} / {} Done".format(count, total))
      # save best param
      best_eval_result['param'] = {
        "intermediary_size": bestIntermediarySize,
        "method": "bilinear"
      }
      current_denoiser_eval_score_given_noiseLevel = best_denoiser_avg_eval_score
      return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser 
    
    elif self.name == "jpegCompression":
      bestQuality = None
      total = len(self.kwargs['denoisers']['jpegCompression']['quality'])
      for quality in self.kwargs['denoisers']['jpegCompression']['quality']:
        if quality <= 0 or not type(quality) == int:
          raise Exception("quality of jpeg compressor must be postive integer")
        jpegCompressor = MyJpegCompression(quality=quality, clip_values=(0,255))
        denoiserParams = {"quality": quality}
        evaluationReportPath = self.kwargs['denoisers']['jpegCompression']['evaluationReportPath'] + "/quality_{}".format(quality)
        current_denoiser_avg_eval_score, evalResults = self._evaluate(
          evaluationReportPath = evaluationReportPath,
          denoiser = jpegCompressor,
          denoiserParams = denoiserParams,
          ds_test = ds_test,
          _denoise_image_func = self._jpeg_denoise_image_func
        )
        if current_denoiser_avg_eval_score > best_denoiser_avg_eval_score:
          best_denoiser_avg_eval_score = current_denoiser_avg_eval_score
          best_eval_result = evalResults
          bestQuality = quality
          best_denoiser = jpegCompressor
          print("bestQuality thus far: {}, best_denoiser_avg_eval_score thus far: {}".format(bestQuality, best_denoiser_avg_eval_score))
        
        count += 1
        print("[TraditionalImageDenoiser] {} / {} Done".format(count, total))
      # save best param
      best_eval_result['param'] = {
        "quality": bestQuality
      }
      current_denoiser_eval_score_given_noiseLevel = best_denoiser_avg_eval_score
      return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser 

    elif self.name == "tv":
      # this denoiser is not scalable on standard resolution such as 224 x 224 x 3 for dataset such as ImageNet
      prob, solver = self.kwargs['denoisers']['tv']['probs'], self.kwargs['denoisers']['tv']['solvers']
      best_denoiser = TotalVarMin(
        prob = prob,
        solver = solver
      )
      # save best param
      best_eval_result['param'] = {
        "prob": prob,
        "solver": solver
      }
      current_denoiser_eval_score_given_noiseLevel = 0.01 # to satisfy (current_denoiser_eval_score_given_noiseLevel > best_denoiser_eval_score_all_noiseLevel) in Denoiser
      return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser 

    elif self.name == "non-local":
      best_fast_mode, best_patch_distance, best_cut_off_distance = None, None, None
      channel_axis = 3 # for RGB
      fast_mode = self.kwargs['denoisers']['non-local']['fast_mode']
      patch_distance =  self.kwargs['denoisers']['non-local']['patch_distance']
      h = self.kwargs['denoisers']['non-local']['cut_off_distance']
      best_denoiser = MyNonLocalDenoiser(
        fast_mode = fast_mode, patch_distance = patch_distance, h = h, channel_axis = channel_axis,
        resolution = resolution
      )
      # save best param
      best_eval_result['param'] = {
        "fast_mode": fast_mode,
        "patch_distance": patch_distance,
        "h" : h,
        "channel_axis": channel_axis
      }
      current_denoiser_eval_score_given_noiseLevel = 0.01 # to satisfy (current_denoiser_eval_score_given_noiseLevel > best_denoiser_eval_score_all_noiseLevel) in Denoiser
      return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser 
    
    elif self.name == "noise2self":
      # source: https://scikit-image.org/docs/stable/api/skimage.restoration.html#calibrate-denoiser
      denoise_function_name = self.kwargs['denoisers']['noise2self']['denoise_function_name']
      best_denoiser = MyNoise2Self(
        denoise_function_name = denoise_function_name,
        resolution = resolution,
        denoise_parameters = {
          "intermediary_size": self.kwargs['denoisers']['noise2self']['e_ae_intermediary_sizes'],
          "method": self.kwargs['denoisers']['noise2self']['methods']
        },
        stride=4, approximate_loss=True, extra_output=False
      )
      # save best param
      best_eval_result['param'] = {
        "denoise_function_name": denoise_function_name,
        "stride": 4, # default value
        "approximate_loss": True, # default value
        "extra_output": False # default value
      }
      current_denoiser_eval_score_given_noiseLevel = 0.01 # to satisfy (current_denoiser_eval_score_given_noiseLevel > best_denoiser_eval_score_all_noiseLevel) in Denoiser
      return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser 
    
    else:
      raise Exception("{} is not supported yet!".format(self.name))
  
  def _evaluate(self, 
    evaluationReportPath, denoiser, denoiserParams, ds_test,
    _denoise_image_func):
    Path(evaluationReportPath).mkdir(parents=True, exist_ok=True)
    current_denoiser_avg_eval_score, evalResults = None, None
    evaluationReportPath += "/denoisingEval.json"
    if not os.path.exists(evaluationReportPath):
      print("[TraditionalImageDenoiser] working on {}".format(evaluationReportPath))
      evalResults, current_denoiser_avg_eval_score = self.evaluateDenoiserFunc(
        curr_denoiser = denoiser,
        curr_denoiser_name = self.name, # "emulated-ae",
        curr_denoiser_params = denoiserParams,
        current_ds_test = ds_test,
        all_eval_paths = self.all_eval_paths,
        y_true_for_benign = self.y_true_for_benign,
        y_true_for_adv = self.y_true_for_adv,
        # denoise_image_func = self._denoise_image_func
        denoise_image_func = _denoise_image_func
      )
      print("current_denoiser_avg_eval_score: {}".format(current_denoiser_avg_eval_score))
      evalResults['current_denoiser_avg_eval_score'] = current_denoiser_avg_eval_score
      with open(evaluationReportPath, 'w') as fp:
        json.dump(evalResults, fp,  indent=4)
      print("[SUCCESS] saved to {}".format(evaluationReportPath))
    else:
      print("evaluationReportPath {} exists !".format(evaluationReportPath))
      with open(evaluationReportPath, 'r') as fp:
        saved_result = json.load(fp)
        current_denoiser_avg_eval_score = saved_result['current_denoiser_avg_eval_score']
        evalResults = saved_result
    
    return current_denoiser_avg_eval_score, evalResults

  def _denoise_image_func(self, img, curr_denoiser):
    denoise_func_dict = {
      "emulated-ae": self._e_ae_denoise_image_func,
      "jpegCompression": self._jpeg_denoise_image_func,
      "tv": self._tv_denoise_image_func,
      "non-local": self._nl_denoise_image_func,
      "noise2self": self._noise2self_denoise_image_func
    }
    return denoise_func_dict[self.name](img = img, curr_denoiser = curr_denoiser)

  def _e_ae_denoise_image_func(self, img, curr_denoiser):
    # clip values of img is [0,1]
    return curr_denoiser(img)
  
  def _noise2self_denoise_image_func(self, img, curr_denoiser):
    denoised_img,_ = curr_denoiser(img)
    return denoised_img

  def _jpeg_denoise_image_func(self, img, curr_denoiser):
    # clip values of img is [0,1]
    denoised_img,_ = curr_denoiser(img * 255)
    return denoised_img
  
  def _nl_denoise_image_func(self, img, curr_denoiser):
    # clip values of img is [0,1]
    denoised_img,_ = curr_denoiser(img * 255)
    return denoised_img
    
  def _tv_denoise_image_func(self, img, curr_denoiser):
    # clip values of img is [0,1]
    denoised_img,_ = curr_denoiser(img.numpy() * 255)
    return denoised_img
