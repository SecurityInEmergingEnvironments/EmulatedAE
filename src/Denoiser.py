import os
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
import json
import gc # import garbage collector interface
from queue import Queue

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Lambda, Reshape, Conv2D, Input, Dense, Dropout, MaxPool2D, UpSampling2D,Flatten, AveragePooling2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, add, LeakyReLU, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error


from DataLoader import DataLoader
from Utils import substituteString
from Constants import BATCH_SIZE, ALL_NN_DENOISER_NAMES, ALL_REGULAR_DENOISER_NAMES, VAE_DEFAULT_INPUT_SHAPE, VAE_REGULAR_TRAINING_EPOCHS
from ParamsRange import ParamsRangeForNNDenoisers
from TraditionalImageDenoiser import TraditionalImageDenoiser

def getModelFilePath(parentPath, dataset_name, currentModelNum, nameOfModel):
    model = None
    modelFilePath = parentPath + \
      "/" + dataset_name + "/" + dataset_name + "_models" + "/" + nameOfModel + "_num_" + str(currentModelNum)
    return modelFilePath

class Denoiser:
  def __init__(self, kwargs):
    self.kwargs = kwargs
    self.denoiserKwargs = kwargs['denoisers']
    self.listOfDenoiserNames = self._getDenoiserNames(self.denoiserKwargs['denoiserNames'], self.denoiserKwargs['mode'])
    self.dataset_name = kwargs['train']['trainingSet']
    self.rawSavedDenoiserPath = self.denoiserKwargs['savedDenoiserPath']
    # self.rawSavedDenoisedAdvPath = self.denoiserKwargs['savedDenoisedAdvPath']
    self.y_true_raw = None

  def _getAdvPredSavedPathAndAdvExamplesNoiseDSPath(self, my_dict, preprocessorObj):
    advPredSavedPath = substituteString(
      my_dict = my_dict,
      formatStr = preprocessorObj['advPredPath']
    )
    advExamplesNoiseDSPath = substituteString(
      my_dict = my_dict,
      formatStr = preprocessorObj['advExamplesNoiseDSPath']
    )
    return advPredSavedPath, advExamplesNoiseDSPath

  def _getTargetPaths(self, numberOfTargetModels, ds_test, load_model = True):
    
    all_paths = []
    attackers = self.kwargs['attack']['attackers']
    modelNames = self.kwargs['train']['namesOfModels']
    print("[_getTargetPaths] numberOfTargetModels: {}".format(numberOfTargetModels))
    progress_bar_test = tf.keras.utils.Progbar(len(attackers) * len(modelNames) * len(numberOfTargetModels)) 
    preprocessorObj = self.kwargs['attack']['preprocessors'][-1] # there should be only 1 here
    for attack in attackers:
      attackerName = attack['attackerName']
      attackParams = attack['params']
      savedFilePath = attack['savedFilePath']
      dsSavedFilePath = attack['dsSavedFilePath']
      noiseDsSavedFilePath = attack['noiseDsSavedFilePath']
      for nameOfModel in modelNames:
        for numOfModel in numberOfTargetModels:
          my_dict = {
            "nameOfModel": nameOfModel,
            "numberOfModel": numOfModel,
            "attackerName": attackerName,
            "savedFilePath":savedFilePath,
            "dsSavedFilePath": dsSavedFilePath,
            "noiseDsSavedFilePath": noiseDsSavedFilePath
          }
          my_dict['pathToCreate'] = substituteString(my_dict = my_dict, formatStr = preprocessorObj['pathToCreate'])
          # advPredPath = substituteString(
          #   my_dict = my_dict,
          #   formatStr = self.kwargs['attack']['advPredPath']
          # )
          # advExamplesDSPath = substituteString(
          #   my_dict = my_dict,
          #   formatStr = self.kwargs['attack']['advExamplesDSPath']
          # )
          # advExamplesNoiseDSPath = substituteString(
          #   my_dict = my_dict,
          #   formatStr = self.kwargs['attack']['advExamplesNoiseDSPath']
          # )
          _, advExamplesNoiseDSPath = self._getAdvPredSavedPathAndAdvExamplesNoiseDSPath(
              my_dict = my_dict,
              preprocessorObj = preprocessorObj
            )
          # load adv. examples / adv. noise
          # ds_adv, noise_ds_adv = None, None
          noise_ds_adv = None
          # if os.path.isdir(advExamplesDSPath) and len(os.listdir(advExamplesDSPath)) > 0:
          #   ds_adv = tf.data.experimental.load(advExamplesDSPath, element_spec=(tf.TensorSpec(shape=(self.kwargs['img_size'], self.kwargs['img_size'], 3), dtype=tf.float32, name=None)))
          
          if os.path.isdir(advExamplesNoiseDSPath) and len(os.listdir(advExamplesNoiseDSPath)) > 0:
            noise_ds_adv = tf.data.experimental.load(advExamplesNoiseDSPath, element_spec=(tf.TensorSpec(shape=(self.kwargs['img_size'], self.kwargs['img_size'], 3), dtype=tf.float32, name=None)), compression='GZIP')

          targetModelFilePath = getModelFilePath(
            dataset_name = self.dataset_name,
            currentModelNum = numOfModel,
            nameOfModel = nameOfModel,
            parentPath = self.kwargs['parentPath']
          )
          targetModel = None
          if load_model:
            targetModel = tf.keras.models.load_model(targetModelFilePath)

          all_paths.append({
            # "advPredPath": advPredPath,
            "nameOfModel":nameOfModel,
            "numOfModel": numOfModel,

            "attackerName": attackerName,
            "attackParams": attackParams,

            "advExamplesNoiseDSPath": advExamplesNoiseDSPath,
            "targetModelFilePath": targetModelFilePath,
            "targetModel": targetModel,
            "noise_ds_adv": noise_ds_adv
          })
          progress_bar_test.add(1)
    return all_paths

  def denoise(self):
    # step 1: create instance of a denoiser for every denoiserName
    numberOfTargetModels = self.kwargs['denoisers']['numberOfTargetModels'] # to get the best version of each denoiser
    os.environ["CUDA_VISIBLE_DEVICES"] = self.kwargs['gpu']
    # step 2.1: load training data
    print("[DEBUG] self.dataset_name: {}".format(self.dataset_name))
    (ds_train, _, ds_test), _ = DataLoader().load_tfds_data(
        img_size=self.kwargs['img_size'],
        dataset_name = self.dataset_name, range1 = None, range2 = None, range3 = None,
        train_idx_range_1 = [0,50], train_idx_range_2 = [50,100], val_idx_range = None)
    y_true_raw = np.concatenate([y for x, y in ds_test], axis=0)
    y_true_for_benign = np.argmax(y_true_raw,axis=1)
    y_true_for_adv = np.argmax(y_true_raw,axis=1) [0:self.kwargs['attack']['numberOfAdvExamples']]
    # step 2.2: preprocess training-validation data
    ds_train, ds_test = self._preprocess(ds_train = ds_train, ds_test = ds_test)

    all_eval_paths_for_picking_best_version = self._getTargetPaths(
      numberOfTargetModels = numberOfTargetModels,
      ds_test = ds_test,
      load_model=False
      )

    best_denoisers = {}

    for denoiserName in self.listOfDenoiserNames:
      best_denoisers[denoiserName] = {}
      best_denoiser_eval_score_all_noiseLevel = 0
      best_eval_result = None
      best_training_noiseLevel = None
      for noiseLevel in self.denoiserKwargs['noiseLevels']:
        denoiser = self._buildDenoiser(
          name = denoiserName,
          noiseLevel = noiseLevel,
          all_eval_paths = all_eval_paths_for_picking_best_version,
          y_true_for_benign = y_true_for_benign,
          y_true_for_adv = y_true_for_adv
        )
        # step 2: train the denoiser
        
        current_denoiser_eval_score_given_noiseLevel, current_eval_result, curr_denoiser = denoiser.train(ds_train = ds_train, ds_test = ds_test, resolution = self.denoiserKwargs['resolution'])
        if current_denoiser_eval_score_given_noiseLevel > best_denoiser_eval_score_all_noiseLevel:
          best_denoiser_eval_score_all_noiseLevel = current_denoiser_eval_score_given_noiseLevel
          best_eval_result = current_eval_result
          best_eval_result['best_training_noiseLevel'] = best_training_noiseLevel
          param = current_eval_result['param']
          if denoiserName in ALL_NN_DENOISER_NAMES:
            param['noise_level'] = noiseLevel
          best_eval_result['best_denoiser_eval_score_all_noiseLevel'] = best_denoiser_eval_score_all_noiseLevel
          best_denoisers[denoiserName] = {
            "denoiser": curr_denoiser,
            "param": param,
            "denoise_image_func": denoiser._denoise_image_func,
            "best_eval_result":best_eval_result
          }
        print("best_denoiser_eval_score_all_noiseLevel thus far: {} at noiseLevel: {}".format(best_denoiser_eval_score_all_noiseLevel, noiseLevel))
        if denoiserName not in ALL_NN_DENOISER_NAMES:
          break # non-nn denoisers do not depend on Gaussian noise
    # """
    ## step 3: evaluate the best denoiser of all noise levels

    # clean up 'all_eval_paths_for_picking_best_version'
    for item in all_eval_paths_for_picking_best_version:
      del item['targetModel']
      gc.collect()
      del item['noise_ds_adv']
      gc.collect()
    del all_eval_paths_for_picking_best_version
    gc.collect()
    # print("best_denoisers: {}".format(best_denoisers))
    all_eval_paths_for_evaluate_best_version = self._getTargetPaths(
      numberOfTargetModels = self.kwargs['train']['numberOfModels'],
      ds_test = ds_test,
      load_model=False
    )
    
    for denoiserName, denoiserDict in best_denoisers.items():
      best_denoiser = denoiserDict['denoiser']
      curr_denoiser_params = denoiserDict['param']
      denoise_image_func = denoiserDict['denoise_image_func']
      bestDenoiserEvalReportPath = self.denoiserKwargs['bestDenoiserGivenAllNoiseLevels'] + "/best_{}_Denoiser_report.json".format(denoiserName)
      if not os.path.exists(bestDenoiserEvalReportPath):
        print("{} not exists! working on it".format(bestDenoiserEvalReportPath))

        evalResults, current_denoiser_avg_eval_score = evaluateDenoiser(
          curr_denoiser = best_denoiser,
          curr_denoiser_name = denoiserName,
          curr_denoiser_params = curr_denoiser_params,
          current_ds_test = ds_test,
          all_eval_paths = all_eval_paths_for_evaluate_best_version,
          y_true_for_benign = y_true_for_benign,
          y_true_for_adv = y_true_for_adv,
          denoise_image_func = denoise_image_func,
          deleteTargetModel=True,
          deleteDsAdv = True
        )
        evalResults['current_denoiser_avg_eval_score'] = current_denoiser_avg_eval_score
        if denoiserName in ALL_NN_DENOISER_NAMES and denoiserDict['param']['noise_level']:
          evalResults['best_noise_level'] = denoiserDict['param']['noise_level']

        del best_denoiser
        gc.collect()
        del denoiserDict['denoiser']
        gc.collect()
        
        with open(bestDenoiserEvalReportPath, 'w') as fp:
          json.dump(evalResults, fp,  indent = 4)
        print("[SUCCESS] saved to {}".format(bestDenoiserEvalReportPath))
        # print(json.dumps(evalResults, indent=4))
      else:
        print("{} exists !".format(bestDenoiserEvalReportPath))
        with open(bestDenoiserEvalReportPath, 'r') as fp:
          saved_eval_result_best_denoiser = json.load(fp)
          print(saved_eval_result_best_denoiser)
    # """

  def _getDenoiserNames(self, names = [], mode = None):
    result = []
    if mode == 'nn' or mode == 'both':
      result += ALL_NN_DENOISER_NAMES
    if mode == 'regular' or mode == 'both':
      result += ALL_REGULAR_DENOISER_NAMES
    return result if result != [] else names

  def _preprocess(self, ds_train = None, ds_test = None, noise = 1e-1):
    if ds_train is None:
      raise Exception("ds_train is None")
    if ds_test is None:
      raise Exception("ds_test is None")
    ds_train = ds_train.map(lambda image, label: ((image / 255) + noise * np.random.normal(0, noise, size=(self.kwargs['img_size'], self.kwargs['img_size'], 3)), image / 255))
    ds_test = ds_test.map(lambda image, label: ((image / 255) + noise * np.random.normal(0, noise, size=(self.kwargs['img_size'], self.kwargs['img_size'], 3)), image / 255))
    return ds_train, ds_test

  def _buildDenoiser(self, name = '', noiseLevel = None, all_eval_paths = [],
          y_true_for_benign = None,
          y_true_for_adv = None):
    savedDenoiserPath = substituteString(
        my_dict={
          "noiseLevel": noiseLevel,
          "denoiserName": name
        },
        formatStr=self.rawSavedDenoiserPath
      )
    print("name is {}".format(name))
    if name == "ae":
      return AutoEncoder(
        img_size = self.kwargs['img_size'],
        savedDenoiserPath = savedDenoiserPath,
        resolution = self.denoiserKwargs['resolution'],
        dataset_name=self.dataset_name,
        all_eval_paths = all_eval_paths,
        y_true_for_benign = y_true_for_benign,
        y_true_for_adv = y_true_for_adv
      )
    elif name == "vae":
      return Vae(
        img_size = self.kwargs['img_size'],
        savedDenoiserPath = savedDenoiserPath,
        resolution = self.denoiserKwargs['resolution'],
        dataset_name=self.dataset_name,
        all_eval_paths = all_eval_paths,
        y_true_for_benign = y_true_for_benign,
        y_true_for_adv = y_true_for_adv
      )
    elif name == "unet":
      return Unet(
        img_size = self.kwargs['img_size'],
        savedDenoiserPath = savedDenoiserPath,
        resolution = self.denoiserKwargs['resolution'],
        dataset_name=self.dataset_name,
        all_eval_paths = all_eval_paths,
        y_true_for_benign = y_true_for_benign,
        y_true_for_adv = y_true_for_adv
      )
    # elif name == "emulated-ae":
    elif name in ALL_REGULAR_DENOISER_NAMES:
      print("[DEBUG] TraditionalImageDenoiser is {}".format(name))
      return TraditionalImageDenoiser(
        name = name,
        kwargs=self.kwargs,
        resolution = self.denoiserKwargs['resolution'],
        dataset_name=self.dataset_name,
        all_eval_paths = all_eval_paths,
        y_true_for_benign = y_true_for_benign,
        y_true_for_adv = y_true_for_adv,
        evaluateDenoiserFunc = evaluateDenoiser
      )
    else:
      return None

def evaluateDenoiser(all_eval_paths, current_ds_test, y_true_for_benign, y_true_for_adv, denoise_image_func, 
  curr_denoiser, curr_denoiser_name, curr_denoiser_params,
  deleteTargetModel = False, deleteDsAdv = False, averageMode = 'macro'):
    total_individual_denoiser_score = 0
    avg_individual_denoiser_score = 0
    results = {"GPU": "rtx6k"}
    total = len(all_eval_paths)
    count = 0
    for all_eval_paths_obj in all_eval_paths:
      print("all_eval_paths_obj: {}".format(all_eval_paths_obj))
      # advPredPath = all_eval_paths_obj['advPredPath']
      advExamplesNoiseDSPath = all_eval_paths_obj['advExamplesNoiseDSPath']
      targetModelFilePath = all_eval_paths_obj['targetModelFilePath']
      targetModel = all_eval_paths_obj['targetModel']
      noise_ds_adv = all_eval_paths_obj['noise_ds_adv']
      attackParams = all_eval_paths_obj['attackParams']

      key = all_eval_paths_obj['nameOfModel'] + "_" + str(all_eval_paths_obj['numOfModel'])
      attackerName = all_eval_paths_obj['attackerName']

      if targetModel is None:
        targetModel = tf.keras.models.load_model(targetModelFilePath)

      # assess the denoising strength against benign examples
      denoised_benign_y_preds =  np.array([])
      denoised_adv_y_preds =  np.array([])
      benign_y_preds =  np.array([])
      progress_bar_test = tf.keras.utils.Progbar(y_true_for_benign.shape[0]) 
      #TODO: putting denoising benign and adv. examples into its own function
      noise_ds_adv = noise_ds_adv.batch(batch_size=BATCH_SIZE, drop_remainder=True)
      iterator = iter(noise_ds_adv)
      totalDenoisedTotalElapsed = 0
      totalTargetPredict = 0
      adv_y_preds = np.array([])
      denoised_y_preds =  np.array([])

      for _, benign_img in current_ds_test:
        # no denoiser, benign examples
        start_target_predict= datetime.datetime.now()
        benign_y_pred = targetModel.predict(benign_img * 255)
        end_target_predict= datetime.datetime.now()
        totalTargetPredict += (((end_target_predict - start_target_predict).total_seconds()))
        benign_y_pred = np.argmax(benign_y_pred, axis=1)
        benign_y_preds = np.concatenate((benign_y_preds, benign_y_pred), axis = 0)

        # no denoiser, adv. examples
        noise = (iterator.get_next()).numpy()
        adv = ((benign_img * 255) + noise) / 255
        adv_y_pred = targetModel.predict(adv * 255)
        adv_y_pred = np.argmax(adv_y_pred, axis=1)
        adv_y_preds = np.concatenate((adv_y_preds, adv_y_pred), axis = 0)

        # with denoiser, benign examples
        denoised_benign_img = denoise_image_func(img = benign_img, curr_denoiser = curr_denoiser)

        denoised_y_pred = targetModel.predict(denoised_benign_img)
        denoised_y_pred = np.argmax(denoised_y_pred, axis=1)
        denoised_benign_y_preds = np.concatenate((denoised_benign_y_preds, denoised_y_pred), axis = 0)

        # with denoiser, adv. examples
        start_denoised = datetime.datetime.now()
        denoised_adv = denoise_image_func(img = adv, curr_denoiser = curr_denoiser)
        end_denoised = datetime.datetime.now()
        denoisedElapsed = ((end_denoised - start_denoised).total_seconds())
        totalDenoisedTotalElapsed += denoisedElapsed

        denoised_y_pred = targetModel.predict(denoised_adv)
        denoised_y_pred = np.argmax(denoised_y_pred, axis=1)
        denoised_adv_y_preds = np.concatenate((denoised_adv_y_preds, denoised_y_pred), axis = 0)

        progress_bar_test.add(denoised_y_pred.shape[0])

      denoised_benign_acc = 100 * accuracy_score(y_true_for_benign, denoised_benign_y_preds)
      denoised_benign_precision, denoised_benign_recall, denoised_benign_f1_score, _ = precision_recall_fscore_support(y_true_for_benign, denoised_benign_y_preds, average = averageMode)
      benign_acc = 100 * accuracy_score(y_true_for_benign, benign_y_preds)
      benign_precision, benign_recall, benign_f1_score, _ = precision_recall_fscore_support(y_true_for_benign, benign_y_preds, average = averageMode)
      
      del denoised_benign_y_preds
      gc.collect()
      del benign_y_preds
      gc.collect()
      print("[benign] benign_acc: {}, denoised_benign_acc: {}".format(benign_acc, denoised_benign_acc))
        
      denoised_adv_acc = 100 * accuracy_score(y_true_for_adv, denoised_adv_y_preds)
      denoised_adv_precision, denoised_adv_recall, denoised_adv_f1_score, _ = precision_recall_fscore_support(y_true_for_adv, denoised_adv_y_preds, average = averageMode)
      del denoised_adv_y_preds
      gc.collect()
      adv_acc = 100 * accuracy_score(y_true_for_adv, adv_y_preds)
      adv_precision, adv_recall, adv_f1_score, _ = precision_recall_fscore_support(y_true_for_adv, adv_y_preds, average = averageMode)
      print("[adv] adv_acc: {}, denoised_adv_acc: {}".format(adv_acc, denoised_adv_acc))
      print("denoisers took {} seconds".format((totalDenoisedTotalElapsed / y_true_for_adv.shape[0]) * 1000))
      
      if key not in results:
        results[key] = {
          "baseline_performance":{
          "natural_accuracy": benign_acc,
          "natural_precision": benign_precision * 100,
          "natural_recall": benign_recall * 100,
          "natural_f1-score": benign_f1_score * 100,
          "inference_elapsed_time_per_1000_in_s": (totalTargetPredict / y_true_for_adv.shape[0]) * 1000
          }
        }
      if 'grey-box_setting' not in results[key]:
        results[key]['grey-box_setting'] = {}
      if attackerName not in results[key]['grey-box_setting']:
        results[key]['grey-box_setting'][attackerName] = {
            "type_of_attack": "evasion",
            "attackParams": attackParams,
            "attacker_performance": {
              "robust_accuracy": adv_acc,
              "robust_precision": adv_precision * 100,
              "robust_recall": adv_recall * 100,
              "robust_f1-score": adv_f1_score * 100
          }
        }
      if 'defenders' not in results[key]['grey-box_setting'][attackerName]:
        results[key]['grey-box_setting'][attackerName]['defenders'] = []
      
      results[key]['grey-box_setting'][attackerName]['defenders'].append(
        {
          "nameOfDefenders": curr_denoiser_name,
          "type": "PREPROCESSOR",
          "defense_params": curr_denoiser_params,
          "defender_performance":{
            "natural_accuracy": denoised_benign_acc,
            "natural_precision": denoised_benign_precision * 100,
            "natural_recall": denoised_benign_recall * 100,
            "natural_f1-score": denoised_benign_f1_score * 100,

            "robust_accuracy": denoised_adv_acc,
            "robust_precision": denoised_adv_precision * 100,
            "robust_recall": denoised_adv_recall  * 100,
            "robust_f1-score": denoised_adv_f1_score  * 100,
            "inference_elapsed_time_per_1000_in_s": (totalDenoisedTotalElapsed / y_true_for_adv.shape[0]) * 1000
          }
        }
      )

      total_individual_denoiser_score += (denoised_benign_acc + denoised_adv_acc) # equal weight
      # print(json.dumps(results, indent=4))
      count += 1
      print("\n{} / {}".format(count, total))
      del targetModel
      gc.collect()
      if deleteTargetModel:
        del all_eval_paths_obj['targetModel']
        gc.collect()
      if deleteDsAdv:
        del all_eval_paths_obj['noise_ds_adv']
        gc.collect()
      del noise_ds_adv
      gc.collect() # https://stackoverflow.com/a/65893798
      
    print("total_individual_denoiser_score: {}".format(total_individual_denoiser_score))
    print("total: {}".format(total))
    return results, (total_individual_denoiser_score / total) # TODO: maybe there is a better way to rank them

def train(ds_train, ds_test, resolution, modelsPipeline, dataset_name, all_eval_paths, y_true_for_benign, y_true_for_adv,
  fittingFunc, denoise_image_func):
    total = len(modelsPipeline)
    count = 0
    best_denoiser_avg_eval_score = 0
    best_denoiser = None
    best_eval_result, bestParam, bestModelSavedPath= {}, {}, ""
    for modelDict in modelsPipeline:
      denoiser_name = modelDict['denoiser_name']
      denoiser = modelDict['model']
      param = modelDict['param']
      save_path = modelDict['save_path']
      evaluationReportPath = modelDict['evaluationReportPath']
      if len(os.listdir(save_path)) == 0:
        print("[DEBUG] working on {}".format(save_path))
        denoiser, timeTaken = fittingFunc(
          ds_train = ds_train,
          ds_test = ds_test,
          denoiser = denoiser
        )
        
        denoiser.save(save_path)
        history_save_path = save_path + "/history.npz"
        print("[DEBUG] saving to {}".format(history_save_path))
        
        np.savez_compressed(history_save_path, trainingTimeInSeconds = timeTaken)
      else:
        print("{} exits !".format(save_path))
        denoiser = tf.keras.models.load_model(save_path)
        print("[DEBUG] successfully loaded the denoiser")
      count += 1
      print("{} / {}".format(count, total))

      # <refactor this block
      current_denoiser_avg_eval_score = None
      if not os.path.exists(evaluationReportPath):
        evalResults, current_denoiser_avg_eval_score = evaluateDenoiser(
          curr_denoiser = denoiser,
          curr_denoiser_name = denoiser_name, 
          curr_denoiser_params = param,
          current_ds_test = ds_test,
          all_eval_paths = all_eval_paths,
          y_true_for_benign = y_true_for_benign,
          y_true_for_adv = y_true_for_adv,
          denoise_image_func = denoise_image_func
        )
        print("current_denoiser_avg_eval_score: {}".format(current_denoiser_avg_eval_score))
        evalResults['current_denoiser_avg_eval_score'] = current_denoiser_avg_eval_score
        with open(evaluationReportPath, 'w') as fp:
          json.dump(evalResults, fp,  indent=4)
        print("[SUCCESS] saved to {}".format(evaluationReportPath))
        print(json.dumps(evalResults, indent=4))
      else:
        print("evaluationReportPath {} exists !".format(evaluationReportPath))
        with open(evaluationReportPath, 'r') as fp:
          saved_result = json.load(fp)
        current_denoiser_avg_eval_score = saved_result['current_denoiser_avg_eval_score']
        evalResults = saved_result
      
      if current_denoiser_avg_eval_score > best_denoiser_avg_eval_score:
        best_denoiser_avg_eval_score = current_denoiser_avg_eval_score
        best_eval_result = evalResults
        bestParam = param
        bestModelSavedPath = save_path
        # del best_denoiser
        # gc.collect()
        best_denoiser = denoiser
        print("best_denoiser_avg_eval_score thus far: {}".format(best_denoiser_avg_eval_score))
      # else:
      #   del denoiser
      #   gc.collect()
      del modelDict['model']
      # refactor this block>
    
    # save best param
    best_eval_result['param'] = bestParam
    best_eval_result['bestModelSavedPath'] = bestModelSavedPath
    current_denoiser_eval_score_given_noiseLevel = best_denoiser_avg_eval_score
    return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser

class Vae:
  # source: https://github.com/Roy-YL/VAE-Adversarial-Defense/blob/master/MNIST_CIFAR10/train_vae.py
  # paper: https://arxiv.org/pdf/1812.02891.pdf
  def __init__(self, img_size, savedDenoiserPath = "", dataset_name = "", resolution = [None, None, None], all_eval_paths = [],
  y_true_for_benign = None, y_true_for_adv = None):
    self.img_size = img_size
    self.resolution = resolution
    self.savedDenoiserPath = savedDenoiserPath
    self.dataset_name = dataset_name
    self.all_eval_paths = all_eval_paths
    self.y_true_for_benign = y_true_for_benign
    self.y_true_for_adv = y_true_for_adv
    self.modelsPipeline = self._buildAllmodels()
    self.bestDenoiserPath = None
  
  def _denoise_image_func(self, img, curr_denoiser):
    img = tf.image.resize(img, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear")
    return tf.image.resize(curr_denoiser.predict(img), (self.img_size, self.img_size), method = "bilinear") * 255

  def _build_model(self, latent_dim):
    def sampling(args):
      z_mean, z_log_sigma = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean=0 and std=1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_sigma) * epsilon
    
    inputs = Input(shape=(VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE, 3))
    latent_dim = latent_dim

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    conv_shape = x.get_shape().as_list()[1:]
    conv_dim = int(conv_shape[0]) * int(conv_shape[1]) * int(conv_shape[2])

    # print(conv_shape, conv_dim)

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
    # print(encoder.summary())

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(conv_dim)(latent_inputs)
    x = Reshape(conv_shape)(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, decoded, name='decoder')
    # print(decoder.summary())
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='VAE')

    reconstruction_loss = mean_squared_error(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= (32 * 32 * 3)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss * 0.01)
    vae.add_loss(vae_loss)

    return vae
  
  def _buildModelGivenParam(self, param = None):
    vae = self._build_model(
      latent_dim = param['latent_dim']
    )
    #Initializing and compiling model
    if param['optimizerLR'] == 'default':
      vae.compile(optimizer=param['optimizer'])
    else:
      optimizer = None
      if param['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=param['optimizerLR'])
      vae.compile(optimizer=optimizer)
    return vae

  def _buildAllmodels(self):
    modelsPipeline = []
    params = ParamsRangeForNNDenoisers(name = "vae", resolution=self.resolution).getParams()
    for param in params:
      # complete save path
      modelPath = substituteString(
        my_dict={
          "latent_dim": param['latent_dim'],
          "optimizer": param['optimizer'],
          "optimizerLR": param['optimizerLR']
        },
        formatStr = "latent_dim-{latent_dim}_optimizer-{optimizer}_optimizerLR-{optimizerLR}" 
      )
      save_path = self.savedDenoiserPath + "/" + modelPath
      Path(save_path).mkdir(parents=True, exist_ok=True)
      evaluationReportPath = save_path + "/denoisingEval.json"
      self.bestDenoiserPath = self.savedDenoiserPath + "/bestDenoiser.json"
      
      modelDict = {
        "denoiser_name": "vae",
        "model": self._buildModelGivenParam(param = param),
        "param": param,
        "save_path": save_path,
        "evaluationReportPath": evaluationReportPath
      }
      modelsPipeline.append(modelDict)
    return modelsPipeline # containing {"model":model, "param":param} for each item

  def _fittingFunc(self, ds_train, ds_test, denoiser):
    ds_train = ds_train.map(lambda noise, benign:(tf.image.resize(noise, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear"), tf.image.resize(benign, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear")))
    ds_test = ds_test.map(lambda noise, benign:(tf.image.resize(noise, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear"), tf.image.resize(benign, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear")))
    start = time.time()
    history = denoiser.fit(
        ds_train,
        epochs = VAE_REGULAR_TRAINING_EPOCHS,
        batch_size = 1,
        shuffle = True,
        validation_data = (ds_test),
        verbose = 1
    )
    end = time.time()
    timeTaken = (end - start)
    return denoiser, timeTaken

  def train(self, ds_train = None, ds_test = None, resolution = [None, None, None]):
    return train(
      ds_train = ds_train,
      ds_test = ds_test,
      resolution = resolution,
      modelsPipeline = self.modelsPipeline,
      dataset_name = self.dataset_name,
      all_eval_paths = self.all_eval_paths,
      y_true_for_benign = self.y_true_for_benign,
      y_true_for_adv = self.y_true_for_adv,
      fittingFunc = self._fittingFunc,
      denoise_image_func = self._denoise_image_func
    )

  def old_train(self, ds_train = None, ds_test = None, resolution = [None, None, None]):
    total = len(self.modelsPipeline)
    count = 0
    best_denoiser_avg_eval_score = 0
    best_eval_result, bestParam, bestModelSavedPath= {}, {}, ""
    best_denoiser = None
    for modelDict in self.modelsPipeline:
      test_denoised = None
      vae = modelDict['model']
      param = modelDict['param']
      save_path = modelDict['save_path']
      evaluationReportPath = modelDict['evaluationReportPath']
      if len(os.listdir(save_path)) == 0:
        print("[DEBUG] working on {}".format(save_path))
        # Need to resize image before sending it to Vae for training
        ds_train = ds_train.map(lambda noise, benign:(tf.image.resize(noise, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear"), tf.image.resize(benign, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear")))
        ds_test = ds_test.map(lambda noise, benign:(tf.image.resize(noise, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear"), tf.image.resize(benign, (VAE_DEFAULT_INPUT_SHAPE, VAE_DEFAULT_INPUT_SHAPE), method = "bilinear")))
        start = time.time()
        history = vae.fit(
            ds_train,
            epochs = VAE_REGULAR_TRAINING_EPOCHS,
            shuffle = True,
            validation_data = (ds_test),
            verbose = 1
        )
        end = time.time()
        vae.save(save_path)
        print("[DEBUG] saved to {}".format(save_path))
        history_save_path = save_path + "/history.npz"
        timeTaken = (end - start)
        np.savez_compressed(history_save_path, trainingTimeInSeconds = timeTaken)
        print("[DEBUG] saved to {}".format(history_save_path))
        ds_test_images = ds_test.take(3)
        test_images = np.concatenate([x for x,y in ds_test_images], axis=0)
        
        progress_bar_test = tf.keras.utils.Progbar(60) # 3 * 20
        for curr_test_images,_ in ds_test_images:
            # curr_test_images_denoised = vae.predict(curr_test_images)
            curr_test_images_denoised = tf.image.resize(vae.predict(curr_test_images), (self.img_size, self.img_size), method = "bilinear")
            if test_denoised is None:
                test_denoised = np.array(curr_test_images_denoised)
            else:
                test_denoised = np.concatenate((test_denoised, curr_test_images_denoised), axis = 0)
            progress_bar_test.add(curr_test_images.shape[0])
        
        # TODO:refactoring
        rows = 4 # defining no. of rows in figure
        cols = 12 # defining no. of colums in figure
        cell_size = 1.5
        f = plt.figure(figsize=(cell_size*cols,cell_size*rows*2)) # defining a figure 
        f.tight_layout()
        for i in range(rows):
            
            for j in range(cols): 
                f.add_subplot(rows*2,cols, (2*i*cols)+(j+1)) # adding sub plot to figure on each iteration
                plt.imshow((test_images[i*cols + j] * 255).astype("uint8")) 
                plt.axis("off")
            
            for j in range(cols): 
                f.add_subplot(rows*2,cols,((2*i+1)*cols)+(j+1)) # adding sub plot to figure on each iteration
                plt.imshow((test_denoised[i*cols + j] * 255).astype("uint8"))
                plt.axis("off")

        f.suptitle("Vae Results - {}".format(self.dataset_name),fontsize=18)
        plt.savefig(save_path+"/Vae_denoised_{}.png".format(self.dataset_name))

        plt.show()

        modelDict['model'] = None # to reduce memory usage, u can load model thru modelDict['save_path']
      else:
        print("{} exits !".format(save_path))
        vae = tf.keras.models.load_model(save_path)
      count += 1
      print("[Vae_train] {} / {}".format(count, total))

      if not os.path.exists(evaluationReportPath):
        evalResults, current_denoiser_avg_eval_score = self._evaluateDenoiser(curr_denoiser = vae, current_ds_test = ds_test) # TODO
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
      
      if current_denoiser_avg_eval_score > best_denoiser_avg_eval_score:
        best_denoiser_avg_eval_score = current_denoiser_avg_eval_score
        best_eval_result = evalResults
        bestParam = param
        bestModelSavedPath = save_path
        best_denoiser = vae
        print("best_denoiser_avg_eval_score thus far: {}".format(best_denoiser_avg_eval_score))
      
      del vae
      del modelDict['model']
    
    # save best param
    best_eval_result['param'] = bestParam
    best_eval_result['bestModelSavedPath'] = bestModelSavedPath
    current_denoiser_eval_score_given_noiseLevel = best_denoiser_avg_eval_score
    return current_denoiser_eval_score_given_noiseLevel, best_eval_result, best_denoiser

class Unet:
  def __init__(self, img_size, savedDenoiserPath = "", dataset_name = "", resolution = [None, None, None], all_eval_paths = [],
  y_true_for_benign = None, y_true_for_adv = None):
    self.img_size = img_size
    self.resolution = resolution
    self.savedDenoiserPath = savedDenoiserPath
    self.dataset_name = dataset_name
    self.all_eval_paths = all_eval_paths
    self.y_true_for_benign = y_true_for_benign
    self.y_true_for_adv = y_true_for_adv
    self.modelsPipeline = self._buildAllmodels()
  
  def _build_model(self, input_layer, start_neurons, dropout_rate):
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same',dilation_rate=2)(input_layer)
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same',dilation_rate=2)(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    half_dropout_rate = (dropout_rate / 2)
    pool1 = Dropout(half_dropout_rate)(pool1)
    
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same',dilation_rate=2)(pool1)
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same',dilation_rate=2)(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same',dilation_rate=2)(pool2)
    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same',dilation_rate=2)(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)
    
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same',dilation_rate=2)(pool3)
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same',dilation_rate=2)(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    #Middle
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same',dilation_rate=2)(pool4)
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same',dilation_rate=2)(convm)
    
    #upconv part
    deconv4 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(dropout_rate)(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout_rate)(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same',dilation_rate=2)(uconv1)
    
    output_layer = Conv2D(3, (1,1), padding='same', activation='sigmoid')(uconv1)
    return output_layer

  def _buildModelGivenParam(self, param = None):
    # source: https://www.kaggle.com/tarunkr/model_unet-denoising-image-mnist-cifar10
    # source: https://www.kaggle.com/milan400/cifar10-denoising-unet-salt-pepper

    input_layer = Input((self.resolution[0], self.resolution[1], self.resolution[2]))
    output_layer = self._build_model(
      input_layer = input_layer,
      start_neurons = param['start_neurons'],
      dropout_rate = param['dropout_rate']
    )
    #Initializing and compiling model
    model_unet = Model(input_layer, output_layer)
    model_unet.compile(optimizer=param['optimizer'], loss=param['loss_func'])
    return model_unet

  def _buildAllmodels(self):
    modelsPipeline = []
    params = ParamsRangeForNNDenoisers(name = "unet", resolution=self.resolution).getParams()
    for param in params:
      # complete save path
      path = substituteString(
        my_dict={
          "loss_func": param['loss_func'],
          "start_neurons": param['start_neurons'],
          "optimizer": param['optimizer'],
          "dropout_rate": param['dropout_rate']
        },
        formatStr = "start_neurons-{start_neurons}_optimizer-{optimizer}_loss_func-{loss_func}_dropout_rate-{dropout_rate}" 
      )
      save_path = self.savedDenoiserPath + "/" + path
      Path(save_path).mkdir(parents=True, exist_ok=True)
      evaluationReportPath = save_path + "/denoisingEval.json"
      self.bestDenoiserPath = self.savedDenoiserPath + "/bestDenoiser.json"
      
      modelDict = {
        "denoiser_name": "unet",
        "model": self._buildModelGivenParam(param = param),
        "param": param,
        "save_path": save_path,
        "evaluationReportPath": evaluationReportPath
      }
      modelsPipeline.append(modelDict)
    return modelsPipeline # containing {"model":model, "param":param} for each item

  def _fittingFunc(self, ds_train, ds_test, denoiser):
    start = time.time()
    history = denoiser.fit(
        ds_train,
        epochs = 1,
        batch_size = 1,
        shuffle = True,
        validation_data = (ds_test),
        verbose = 1
    )
    end = time.time()
    timeTaken = (end - start)
    return denoiser, timeTaken
  
  def _denoise_image_func(self, img, curr_denoiser):
    return curr_denoiser.predict(img) * 255 # un-normalizing

  def train(self, ds_train = None, ds_test = None, resolution = [None, None, None]):
    return train(
      ds_train = ds_train,
      ds_test = ds_test,
      resolution = resolution,
      modelsPipeline = self.modelsPipeline,
      dataset_name = self.dataset_name,
      all_eval_paths = self.all_eval_paths,
      y_true_for_benign = self.y_true_for_benign,
      y_true_for_adv = self.y_true_for_adv,
      fittingFunc = self._fittingFunc,
      denoise_image_func = self._denoise_image_func
    )

class AutoEncoder:
  def __init__(self, img_size, savedDenoiserPath = "", dataset_name = "", resolution = [None, None, None], all_eval_paths = [],
  y_true_for_benign = None, y_true_for_adv = None):
    self.resolution = resolution
    self.savedDenoiserPath = savedDenoiserPath
    self.dataset_name = dataset_name
    self.all_eval_paths = all_eval_paths
    self.y_true_for_benign = y_true_for_benign
    self.y_true_for_adv = y_true_for_adv
    self.img_size = img_size
    self.modelsPipeline = self._buildAllmodels()

  def _buildAllmodels(self):
    modelsPipeline = []
    params = ParamsRangeForNNDenoisers(name = "ae", resolution=self.resolution).getParams()
    for param in params:
      # complete save path
      path = substituteString(
        my_dict={
          "numOfFilters": param['numOfFilters'],
          "optimizer": param['optimizer'],
          "loss_func": param['loss_func'],
          "optimizerLR": param['optimizerLR'],
          "dropout_rate": param['dropout_rate'],
        },
        formatStr = "numOfFilters-{numOfFilters}_optimizer-{optimizer}_loss_func-{loss_func}_optimizerLR-{optimizerLR}_dropout_rate-{dropout_rate}" 
      )
      save_path = self.savedDenoiserPath + "/" + path
      Path(save_path).mkdir(parents=True, exist_ok=True)
      evaluationReportPath = save_path + "/denoisingEval.json"
      self.bestDenoiserPath = self.savedDenoiserPath + "/bestDenoiser.json"
      
      modelDict = {
        "denoiser_name": "ae",
        "model": self._buildModelGivenParam(param = param),
        "param": param,
        "save_path": save_path,
        "evaluationReportPath": evaluationReportPath
      }
      modelsPipeline.append(modelDict)
    return modelsPipeline # containing {"model":model, "param":param} for each item

  def _buildModelGivenParam(self, param = None):
    size = self.img_size
    channel = 3

    # Encoder 
    inputs = Input(shape=(size, size, channel))
    numOfFilters = param['numOfFilters']
    dropout_rate = param['dropout_rate']

    x = Conv2D(numOfFilters, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)

    skip = Conv2D(64, 3, padding='same')(x) # skip connection for decoder
    x = LeakyReLU()(skip)
    x = BatchNormalization()(x)
    x = AveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(numOfFilters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = AveragePooling2D()(x)
    x = Dropout(dropout_rate)(x) # new
    x = Conv2D(numOfFilters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = AveragePooling2D()(x)

    # Decoder
    x = Conv2DTranspose(numOfFilters, 3,activation='relu',strides=(2,2), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(numOfFilters, 3, activation='relu',strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(numOfFilters, 3, activation='relu',strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2DTranspose(64, 3, padding='same')(x)
    x = add([x,skip]) # adding skip connection
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    decoded = Conv2DTranspose(3, 3, activation='sigmoid',strides=(2,2), padding='same')(x) # original: activation='sigmoid'

    autoencoder = Model(inputs, decoded)

    optimizer = None

    if param['optimizer'] == 'sgd':
      optimizer = SGD(learning_rate=param['optimizerLR'])
    elif param['optimizer']:
      optimizer = Adam(learning_rate=param['optimizerLR'])
    
    if optimizer is None:
      raise Exception("Optimizer is None")

    autoencoder.compile(optimizer = optimizer, loss=param['loss_func']) # original lr=0.00001
    return autoencoder
  
  def _fittingFunc(self, ds_train, ds_test, denoiser):
    start = time.time()
    history = denoiser.fit(
        ds_train,
        epochs = 5,
        batch_size = 1,
        shuffle = True,
        validation_data = (ds_test),
        verbose = 1
    )
    end = time.time()
    timeTaken = (end - start)
    return denoiser, timeTaken

  def _denoise_image_func(self, img, curr_denoiser):
    return curr_denoiser.predict(img) * 255 # un-normalizing

  def train(self, ds_train = None, ds_test = None, resolution = [None, None, None]):
    return train(
      ds_train = ds_train,
      ds_test = ds_test,
      resolution = resolution,
      modelsPipeline = self.modelsPipeline,
      dataset_name = self.dataset_name,
      all_eval_paths = self.all_eval_paths,
      y_true_for_benign = self.y_true_for_benign,
      y_true_for_adv = self.y_true_for_adv,
      fittingFunc = self._fittingFunc,
      denoise_image_func = self._denoise_image_func
    )
      
      
      

