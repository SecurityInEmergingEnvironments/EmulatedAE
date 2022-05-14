import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import statistics
from pathlib import Path
import gc # import garbage collector interface
import shutil

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging.config
logging.config.dictConfig({
  'version':1,
  'disable_existing_loggers': True
})
from DataLoader import DataLoader
from Utils import substituteString, generateSamples
from ReportGenerator import ClassifierBaseAttackReport
from Constants import BATCH_SIZE, TMP_NAME
from ParamsRange import ParamsRangeRetriever

from PreprocessingDenoiser import EAE, MyJpegCompression

"""
This componenet is to attack trained baseline classifiers
"""
class Attacker:
  def __init__(self, kwargs):
    self.kwargs = kwargs
    # self.advPredPath = kwargs['attack']['advPredPath']
    # self.advExamplesDSPath = kwargs['attack']['advExamplesDSPath']
    # self.advExamplesNoiseDSPath = kwargs['attack']['advExamplesNoiseDSPath']
    self.numberOfAdvExamples = kwargs['attack']['numberOfAdvExamples']
    self.averageMode = kwargs['attack']['averageMode']
    self.attackers = kwargs['attack']['attackers']
    self.dataset_name = kwargs['train']['trainingSet']
    self.modelNames = kwargs['train']['namesOfModels']
  
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

  def _getPreprocessorPipeline(self, preprocessorObj):
    preprocessorPipeline = []
    for preprocessorName in preprocessorObj['preprocessor_pipeline']:
      if preprocessorName == 'E-AE':
        preprocessorPipeline.append(EAE(preprocessorObj['params']))
      else:
        raise Exception("{} preprocessor is not supported yet".format(preprocessorName))
    
    return preprocessorPipeline

  def runAttack(self): # main public method
    # load the test data
    (_, _, ds_test), _ = DataLoader().load_tfds_data( 
      img_size=self.kwargs['img_size'],
      dataset_name = self.dataset_name, range1 = None, range2 = None, range3 = None,
      train_idx_range_1 = None, train_idx_range_2 = None, val_idx_range = None,
      mode = ("test-only" if self.dataset_name == "GTSRB" else ""))

    for attack in self.attackers:
      attackerName = attack['attackerName']
      params = attack['params']
      savedFilePath = attack['savedFilePath']
      dsSavedFilePath = attack['dsSavedFilePath']
      noiseDsSavedFilePath = attack['noiseDsSavedFilePath']
      for nameOfModel in self.modelNames:
        for numOfModel in (self.kwargs['train']['numberOfModels']):
          for preprocessorObj in self.kwargs['attack']['preprocessors']:
            preprocessing_defences = None
            if preprocessorObj['name'] != "NO_PREPROCESSOR":
              preprocessing_defences = self._getPreprocessorPipeline(preprocessorObj)
            my_dict = {
              "nameOfModel": nameOfModel,
              "numberOfModel": numOfModel,
              "attackerName": attackerName,
              "savedFilePath": savedFilePath,
              "dsSavedFilePath": dsSavedFilePath,
              "noiseDsSavedFilePath": noiseDsSavedFilePath
            }
            # make empty path
            pathToCreate = substituteString(my_dict = my_dict, formatStr = preprocessorObj['pathToCreate'])
            Path(pathToCreate).mkdir(parents=True, exist_ok=True)
            my_dict['pathToCreate'] = pathToCreate
            advPredSavedPath, advExamplesNoiseDSPath = self._getAdvPredSavedPathAndAdvExamplesNoiseDSPath(
              my_dict = my_dict,
              preprocessorObj = preprocessorObj
            )

            benign_acc, adv_acc, mean_l2_distances, mean_l_inf_distances = None, None, None, None
            # construct the full path name
            all_x_test_adv, adv_noises_list = None, None
            benign_y_preds = np.array([])
            adv_y_preds = np.array([])
            # get the models
            Path(advExamplesNoiseDSPath).mkdir(parents=True, exist_ok=True)
            targetModel, targetClassifier = None, None
            # if len(os.listdir(advExamplesDSPath)) == 0 or len(os.listdir(advExamplesNoiseDSPath)) == 0 or not os.path.exists(advPredSavedPath):
            if len(os.listdir(advExamplesNoiseDSPath)) == 0 or not os.path.exists(advPredSavedPath):
              targetModel = self._getModel(currentModelNum = numOfModel, nameOfModel = nameOfModel, parentPath = self.kwargs['parentPath'])
              _, classes, _ = ParamsRangeRetriever(self.dataset_name).getParams()
              targetClassifier = self._getClassifier(model = targetModel, classes = classes, preprocessing_defences = preprocessing_defences)
            # extract images and labels from test data
            
            # print("[DEBUG] finish loading data")
            y_true = np.concatenate([y for x, y in ds_test], axis=0)
            y_true = np.argmax(y_true,axis=1)[0:self.numberOfAdvExamples]# TODO: DEBUG

            l2_distances, l_inf_distances, fooling_rates = [],[],[]
            if len(os.listdir(advExamplesNoiseDSPath)) == 0:
              print(advExamplesNoiseDSPath+" not exist yet. Creating one...")
              # get the attacker
              attacker = self._getAttack(
                attackerName = attackerName,
                params = params,
                classifier = targetClassifier
              )

              
              
              # Generating adv. examples
              print("=== Generating adv. examples ===")
              # create the tmp path for checkpoints
              tmpPath = pathToCreate + TMP_NAME + attackerName # E.g: data_224_224/cifar100/attacks/VGG19/model_0/tmp_PGD
              Path(tmpPath).mkdir(parents=True, exist_ok=True)
              print("[DEBUG] tmpPath: {}".format(tmpPath))
              progress_bar_test = tf.keras.utils.Progbar(y_true.shape[0]) #TODO: DEBUG
              # get L2 and L-inf
              
              count, skip = 0, 0
              # load checkpoint if possible
              adv_noises_list, adv_y_preds, benign_y_preds = self._loadCkpt(
                loadPath = tmpPath,
                adv_noises_list = adv_noises_list,
                adv_y_preds = adv_y_preds,
                benign_y_preds = benign_y_preds,
                noiseDsSavedFilePath = noiseDsSavedFilePath
              )
              if adv_noises_list is not None:
                skip = int((adv_noises_list.shape[0]) / BATCH_SIZE)
                count = skip
                progress_bar_test.add(adv_noises_list.shape[0])

              for x, y in ds_test.skip(skip):
                # ART
                benign_y_pred = targetClassifier.predict(x)
                benign_y_pred = np.argmax(benign_y_pred,axis=1)
                benign_y_preds = np.concatenate((benign_y_preds, benign_y_pred), axis = 0)
                x_copy = np.copy(x.numpy()) # to prevent incidental modification accoding to https://stackoverflow.com/a/56554855
                x_test_adv = attacker.generate(x = x_copy, y = y.numpy())
                if all_x_test_adv is None:
                  all_x_test_adv = np.array(x_test_adv)
                elif all_x_test_adv.shape[0] < 16:
                  all_x_test_adv = np.concatenate((all_x_test_adv, x_test_adv), axis = 0)
                adv_y_pred = targetClassifier.predict(x_test_adv)
                adv_y_pred = np.argmax(adv_y_pred,axis=1)
                adv_y_preds = np.concatenate((adv_y_preds, adv_y_pred), axis = 0)
                
                adv_noises = x_test_adv - x_copy
                l2_distances.append(np.linalg.norm(adv_noises))
                l_inf_distances.append(np.linalg.norm(adv_noises.ravel(),ord = np.inf))

                if adv_noises_list is None:
                  adv_noises_list = np.array(adv_noises)
                else:
                  adv_noises_list = np.concatenate((adv_noises_list, adv_noises), axis = 0)
                
                # save checkpoints
                self._saveCkpt(
                  pathToSaveWhenExit = tmpPath + "/{}_ckpt_{}.npz".format(noiseDsSavedFilePath, count),
                  noiseToSaveWhenExit = adv_noises,
                  predToSaveWhenExit = adv_y_pred,
                  benign_y_pred = benign_y_pred
                )
                progress_bar_test.add(x.shape[0])
                count += 1
              
              # get classification performance metrics
              print("adv_y_preds.shape: "+str(adv_y_preds.shape))
              print("benign_y_preds.shape: "+str(benign_y_preds.shape))
              print("y_true.shape: "+str(y_true.shape))
              benign_acc = 100 * accuracy_score(y_true, benign_y_preds) 
              adv_acc = 100 * accuracy_score(y_true, adv_y_preds)
              adv_precision, adv_recall, adv_f1_score, _ = precision_recall_fscore_support(y_true, adv_y_preds, average = self.averageMode)
              print("benign_acc: " + str(benign_acc) + ", adv_acc: "+str(adv_acc))
              print("adv_precision: " + str(adv_precision) + ", adv_recall: "+str(adv_recall) + ", adv_f1_score: "+str(adv_f1_score))
              
              # get a sample of adv. examples
              if all_x_test_adv is not None and len(all_x_test_adv) > 0:
                benign_examples_path = "images/"+self.dataset_name+"_benign_examples.png"
                generateSamples(data = all_x_test_adv[:16], path = benign_examples_path)
                adv_examples_path = "images/"+self.dataset_name + "_attacker_" + attackerName + "_adv_examples.png"
                generateSamples(data = all_x_test_adv[:16], path = adv_examples_path)
              
              # save adv. examples
              # TODO: DEBUG
              np.savez_compressed(advPredSavedPath,
                benign_y_preds = benign_y_preds,
                benign_acc = benign_acc,
                adv_y_preds = adv_y_preds,
                adv_acc = adv_acc,
                adv_precision = adv_precision,
                adv_recall = adv_recall,
                adv_f1_score = adv_f1_score,
                l2_distances = l2_distances,
                l_inf_distances = l_inf_distances,
                averageMode = self.averageMode,
                numberOfAdvExamples = self.numberOfAdvExamples
              )
              if len(l2_distances) > 0 and len(l_inf_distances) > 0:
                mean_l2_distances = statistics.mean(l2_distances)
                mean_l_inf_distances = statistics.mean(l_inf_distances)
              # save adv_noises_list
              noise_ds_adv = tf.data.Dataset.from_tensor_slices((adv_noises_list))
              tf.data.experimental.save(noise_ds_adv, advExamplesNoiseDSPath, compression='GZIP')
              print("[SUCCESS] saved to "+advExamplesNoiseDSPath)
              
              # clean up ckpt
              if "tmp" in tmpPath.split('/')[-1]: # if 'tmp' in the deepest folder
                try:
                    shutil.rmtree(tmpPath)
                    print("[LOGGING] remove {}".format(tmpPath))
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))
              # clean up allocated mem.
              del noise_ds_adv
              del adv_noises_list
              gc.collect()
            # elif len(os.listdir(advExamplesNoiseDSPath)) == 0:
            #   print("working on {}".format(advExamplesNoiseDSPath))
            #   l2_distances, l_inf_distances = self._saveAdvNoise(
            #     # advExamplesDSPath = advExamplesDSPath,
            #     advExamplesNoiseDSPath = advExamplesNoiseDSPath,
            #     ds_test = ds_test,
            #     length = y_true.shape[0]
            #     )
            elif len(os.listdir(advExamplesNoiseDSPath)) > 0:
              print("{} already exists".format(advExamplesNoiseDSPath))

            self._saveClassificationMetricValues(
              # advExamplesDSPath = advExamplesDSPath,
              advExamplesNoiseDSPath = advExamplesNoiseDSPath,
              advPredSavedPath = advPredSavedPath,
              targetModel = targetModel,
              y_true = y_true,
              ds_test = ds_test,
              l2_distances = l2_distances,
              l_inf_distances = l_inf_distances
              )

            # # re-create ds-adv
            # if len(os.listdir(advExamplesDSPath)) == 0:
            #   noise_ds_adv = tf.data.experimental.load(advExamplesNoiseDSPath, element_spec=(tf.TensorSpec(shape=(self.kwargs['img_size'], self.kwargs['img_size'], 3), dtype=tf.float32, name=None)), compression='GZIP')
            #   noise_ds_adv = noise_ds_adv.batch(batch_size=BATCH_SIZE, drop_remainder=True)
            #   print("re-creating {}".format(advExamplesDSPath))
            #   all_x_test_adv = None
            #   i = 0
            #   progress_bar_test = tf.keras.utils.Progbar(self.numberOfAdvExamples)
            #   for noise in noise_ds_adv: # each batch are 20 images (default)
            #     # re-create adv. examples
            #     skip = i
            #     benign_examples = np.concatenate([np.copy(x.numpy()) for x,y in ds_test.skip(skip).take(1)], axis=0)
            #     x_test_adv = (benign_examples + noise)
            #     if all_x_test_adv is None:
            #       all_x_test_adv = np.array(x_test_adv)
            #     else:
            #       all_x_test_adv = np.concatenate((all_x_test_adv, x_test_adv), axis = 0)
            #     i+=1
            #     progress_bar_test.add(noise.shape[0])
            #   ds_adv = tf.data.Dataset.from_tensor_slices((all_x_test_adv))
            #   tf.data.experimental.save(ds_adv, advExamplesDSPath)
            #   print("[success] saved to {}".format(advExamplesDSPath))
            #   del ds_adv
            #   gc.collect()
            #   del all_x_test_adv
            #   gc.collect()
            # else:
            #   print("{} already exist!".format(advExamplesDSPath))

            if targetModel is not None:
              del targetModel
              gc.collect()
            if targetClassifier is not None:
              del targetClassifier
              gc.collect()
            if len(l2_distances) > 0 and len(l_inf_distances) > 0:
              mean_l2_distances = statistics.mean(l2_distances)
              mean_l_inf_distances = statistics.mean(l_inf_distances)
              print("l2_distances: "+str(mean_l2_distances))
              print("l_inf_distances: "+str(mean_l_inf_distances))
    del ds_test
    gc.collect()
  def _saveAdvNoise(self, advExamplesDSPath, advExamplesNoiseDSPath, length, ds_test):
    ds_adv = tf.data.experimental.load(advExamplesDSPath, element_spec=(tf.TensorSpec(shape=(self.kwargs['img_size'], self.kwargs['img_size'], 3), dtype=tf.float32, name=None)))
    ds_adv = ds_adv.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    count = 0
    adv_noises_list = None
    l2_distances, l_inf_distances = [],[]
    progress_bar_test = tf.keras.utils.Progbar(length) #TODO: DEBUG
    for adv in ds_adv: # each batch are 20 images (default)
      adv_copy = np.copy(adv.numpy())
      skip = count
      benign_examples = np.concatenate([x for x, y in ds_test.skip(skip).take(1)], axis=0)
      if not (adv_copy.shape == benign_examples.shape):
        raise Exception("adv_copy.shape: {}, benign_examples.shape: {}".format(adv_copy.shape, benign_examples.shape))
      adv_noises = adv_copy - benign_examples
      l2_distances.append(np.linalg.norm(adv_noises))
      l_inf_distances.append(np.linalg.norm(adv_noises.ravel(),ord = np.inf))
      if adv_noises_list is None:
        adv_noises_list = np.array(adv_noises)
      else:
        adv_noises_list = np.concatenate((adv_noises_list, adv_noises), axis = 0)
      progress_bar_test.add(adv.shape[0])
      count += 1
      if count == (self.numberOfAdvExamples / BATCH_SIZE):
        break
    noise_ds_adv = tf.data.Dataset.from_tensor_slices((adv_noises_list))
    tf.data.experimental.save(noise_ds_adv, advExamplesNoiseDSPath, compression='GZIP')
    print("[Success] saved to {}".format(advExamplesNoiseDSPath))
    del ds_adv
    gc.collect()
    del noise_ds_adv
    gc.collect()
    del adv_noises_list
    gc.collect()
    return l2_distances, l_inf_distances

  def _saveClassificationMetricValues(self,
    advExamplesNoiseDSPath, advPredSavedPath, targetModel, y_true, l2_distances, l_inf_distances, ds_test, advExamplesDSPath = None):
    if not os.path.exists(advPredSavedPath) and advExamplesDSPath is not None:
      # Natural classification performance
      print("working on {}".format(advPredSavedPath))
      benign_y_preds = targetModel.predict(ds_test)
      benign_y_preds = np.argmax(benign_y_preds,axis=1)[:self.numberOfAdvExamples]
      benign_acc = 100 * accuracy_score(y_true, benign_y_preds) 
      # Robustness
      ds_adv = tf.data.experimental.load(advExamplesDSPath, element_spec=(tf.TensorSpec(shape=(self.kwargs['img_size'], self.kwargs['img_size'], 3), dtype=tf.float32, name=None)))
      ds_adv = ds_adv.batch(batch_size=BATCH_SIZE, drop_remainder=True)
      adv_y_preds = targetModel.predict(ds_adv)
      adv_y_preds = np.argmax(adv_y_preds,axis=1)
      adv_acc = 100 * accuracy_score(y_true, adv_y_preds)
      adv_precision, adv_recall, adv_f1_score, _ = precision_recall_fscore_support(y_true, adv_y_preds, average = self.averageMode)
      # save results
      np.savez_compressed(advPredSavedPath,
        benign_y_preds = benign_y_preds,
        benign_acc = benign_acc,
        adv_y_preds = adv_y_preds,
        adv_acc = adv_acc,
        adv_precision = adv_precision * 100,
        adv_recall = adv_recall * 100,
        adv_f1_score = adv_f1_score * 100,
        l2_distances = l2_distances,
        l_inf_distances = l_inf_distances,
        averageMode = self.averageMode,
        numberOfAdvExamples = self.numberOfAdvExamples
      )
      print("saving to {}".format(advPredSavedPath))
      print("benign_acc: " + str(benign_acc) + ", adv_acc: "+str(adv_acc))
      print("adv_precision: " + str(adv_precision) + ", adv_recall: "+str(adv_recall) + ", adv_f1_score: "+str(adv_f1_score))
      del ds_adv
      gc.collect()
  
  def _saveCkpt(self, pathToSaveWhenExit, noiseToSaveWhenExit, predToSaveWhenExit, benign_y_pred):
    np.savez_compressed(pathToSaveWhenExit, 
        tmp_adv_noises = noiseToSaveWhenExit,
        tmp_adv_y_pred = predToSaveWhenExit,
        benign_y_pred = benign_y_pred
      )

  def _loadCkpt(self, loadPath, noiseDsSavedFilePath, benign_y_preds, adv_noises_list = None, adv_y_preds = np.array([])):
    if len(os.listdir(loadPath)) > 0:
      for count in range (len(os.listdir(loadPath))):
        # currentFilePath = loadPath + "/" + nameOfModel+"_model_0_attack_{}.npz".format(count)
        currentFilePath = loadPath + "/{}_ckpt_{}.npz".format(noiseDsSavedFilePath, count)
        try:
          ckpt = np.load(currentFilePath)
          adv_noises = ckpt['tmp_adv_noises']
          adv_y_pred = ckpt['tmp_adv_y_pred']
          adv_y_preds = np.concatenate((adv_y_preds, adv_y_pred), axis = 0)
          benign_y_pred = ckpt['benign_y_pred']
          benign_y_preds = np.concatenate((benign_y_preds, benign_y_pred), axis = 0)
          if adv_noises_list is None:
            adv_noises_list = np.array(adv_noises)
          else:
            adv_noises_list = np.concatenate((adv_noises_list, adv_noises), axis = 0)
          ckpt.close()
        except Exception as e:
          print(e)
          if 'tmp' in currentFilePath:
            os.remove(currentFilePath)
            print("removed {}".format(currentFilePath))
          else:
            print("{} file cannot be removed".format(currentFilePath))  
    return adv_noises_list, adv_y_preds, benign_y_preds
    # # noise ds should be available by this line
    # if (False): # sanity check
    #   noise_ds = tf.data.experimental.load(advExamplesNoiseDSPath, element_spec=(tf.TensorSpec(shape=(self.kwargs['img_size'], self.kwargs['img_size'], 3), dtype=tf.float32, name=None)), compression='GZIP')
    #   noise_ds = noise_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    #   i = 0
    #   progress_bar_test = tf.keras.utils.Progbar(y_true.shape[0]) #TODO: DEBUG
    #   adv_y_preds = np.array([])
    #   for noise in noise_ds:
    #     skip = i
    #     benign_examples = np.concatenate([x for x, y in ds_test.skip(skip).take(1)], axis=0)
    #     adv_examples = benign_examples + noise

    #     adv_y_preds = np.concatenate((adv_y_preds, np.argmax(targetModel.predict(adv_examples),axis=1)), axis = 0)
    #     i+=1
    #     progress_bar_test.add(benign_examples.shape[0])

    #   new_denoised_adv_acc = 100 * accuracy_score(y_true, adv_y_preds)
    #   print("[Noise reconstruction] adv_acc: " + str(new_denoised_adv_acc))


  def _getModel(self, parentPath, currentModelNum = None, nameOfModel = None):
    model = None
    modelFilePath = parentPath + \
      "/" + self.dataset_name + "/" + self.dataset_name + "_models" + "/" + nameOfModel + "_num_" + str(currentModelNum)
    model = tf.keras.models.load_model(modelFilePath)
    return model

  def _getAttack(self, attackerName, classifier, params):
    if attackerName == "PGD":
      return ProjectedGradientDescent(
        estimator = classifier,
        batch_size = params['batch_size'],
        eps = params['eps'],
        max_iter = params['max_iter'],
        num_random_init = params['num_random_init'],
        verbose=False
      )
    
    elif attackerName == "deepFool":
      return DeepFool(
        classifier = classifier,
        batch_size = params['batch_size'],
        epsilon = params['epsilon'],
        max_iter = params['max_iter'],
        verbose=False
      )

    elif attackerName == "FGSM":
      return FastGradientMethod(
        estimator = classifier,
        batch_size  = params['batch_size'],
        eps = params['eps'],
        num_random_init = params['num_random_init'],
        minimal=params['minimal']
      )
    else:
      return None
  
  def _getClassifier(self, model = None, classes = None, preprocessing_defences = None):
    ##############################################
    # boilerplate code for current wrapper. Might be removed in the next ART update.
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    # ##############################################

    tf2_classifier = TensorFlowV2Classifier(
      model = model,
      loss_object = loss_object,
      train_step = train_step,
      nb_classes = classes,
      preprocessing_defences = preprocessing_defences,
      input_shape = (self.kwargs['img_size'], self.kwargs['img_size'], 3),
    )
    return tf2_classifier