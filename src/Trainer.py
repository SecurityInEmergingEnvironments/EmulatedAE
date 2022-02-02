# this class trains models
from ModelLoader import ModelLoader
from ParamsRange import ParamsRangeRetriever
from Constants import MODEL_SAVE_PARENT_PATH, MIN_VAL_LOSS_CONSTANT
from DataLoader import DataLoader
from KFoldIndicesGenerator import KFoldIndicesGenerator
from tensorflow.keras import layers

import statistics
import numpy as np
import json
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report
import time

"""
This component is to train baseline classifiers.
Training involves fine-tuning hyperparameters and retraining with best
hyperparameters for multiple models
"""
class ClassifierTrainer:
  __kwargsTrain = None
  def __init__(self, kwargs):
    self.__gpu = kwargs['gpu']
    self.kwargs = kwargs
    self.__kwargsTrain = kwargs['train']
  
  def getBestParamPath(self, modelName:str = None) -> str:
    return self.__kwargsTrain['bestParamsJSONPath'] + "/" + modelName

  def isExist(self, path:str = None) -> bool:
    return os.path.exists(path)

  def unfreeze_model(self, model, params):
    # We unfreeze the top 'params['layersToUnfreeze'])' layers while leaving BatchNorm layers frozen
    # if params['layersToUnfreeze']) == 0, it means training all layers
    for layer in model.layers[(params['layersToUnfreeze']):]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

  """
  Retrain with best hyperparameter set and return the best estimator
  """
  def retrainWithBestParams(self,
      nameOfModel:str = None,
      modelSavedPath:str = None, x_train:np.ndarray = None, y_train:np.ndarray = None,
      bestParams:dict = {},
      dataset_name: str = None,
      classes:int = 0,
      freezeEpochs:int = -1,
      unfreezeEpochs:int = -1
      ):
    # retraining with best hyperparameters
    # start timer here
    start = time.time()
    all_loss_scores = [MIN_VAL_LOSS_CONSTANT]
    all_acc_scores = []
    best_estimator = None

    if len(os.listdir(modelSavedPath)) > 0:
      return tf.keras.models.load_model(modelSavedPath)

    best_estimator_history = None

    # load an untrained model
    # modelLoader = ModelLoader(classes)
    
    # get cross-validation indices. Read more about cross-validation here: https://scikit-learn.org/stable/modules/cross_validation.html
    kFoldIndicesGenerator = KFoldIndicesGenerator()
    all_indices = kFoldIndicesGenerator.generateIndices(self.__kwargsTrain['kFold_n_splits'])

    for curr_dictionary in all_indices:
      (ds_train, ds_val, ds_test), ds_info = DataLoader().load_tfds_data(
      img_size=self.kwargs['img_size'],
      dataset_name = dataset_name,
      range1 = curr_dictionary["train_1_string_range"],
      range2 = curr_dictionary["train_2_string_range"],
      range3 = curr_dictionary["val_string_range"],
      train_idx_range_1 = curr_dictionary["train_idx_range_1"],
      train_idx_range_2 = curr_dictionary["train_idx_range_2"],
      val_idx_range = curr_dictionary["val_idx_range"]
      )
      # training
      
      currentModel, history = self.frozenAndUnfrozenTraining(
        nameOfModel = nameOfModel,
        params = bestParams,
        dataset_name = dataset_name,
        curr_dictionary = curr_dictionary,
        freezeEpochs = self.__kwargsTrain['freezeEpochs'],
        unfreezeEpochs = self.__kwargsTrain['unfreezeEpochs'],
        ds_train = ds_train,
        ds_val = ds_val,
        ds_test = ds_test,
        classes = classes
      )
      
      # evaluate current model
      self.runTest(current_ds_test = ds_test, model = currentModel)
      curr_loss = history.history['val_loss'][-1]
      curr_acc = history.history['val_accuracy'][-1]
      
      # get the minimum loss score and save the best thus far model
      if len(all_loss_scores) > 0 and curr_loss < min(all_loss_scores):
        best_estimator = currentModel
        currentModel.save(modelSavedPath)
        print("saved thus far best currentModel to ",modelSavedPath)
        # save best_estimator_history
        tr_acc = history.history['accuracy']
        tr_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        np.savez(modelSavedPath + "/best_estimator_tr_val_acc_loss_and_param.npz",
          tr_acc = tr_acc,
          tr_loss = tr_loss,
          val_acc = val_acc,
          val_loss = val_loss,
          best_param = bestParams
        )
        print("saved " + nameOfModel + "_best_estimator_tr_val_acc_loss_and_param")

      all_loss_scores.append(curr_loss)
      all_acc_scores.append(curr_acc)
      print("all_loss_scores: ",all_loss_scores)
    
    print("=== finished retraining ===")
    mean_acc_best_estimator_score = statistics.mean(all_acc_scores)
    stdev_acc_best_estimator_score = statistics.stdev(all_acc_scores)
    print("mean_acc_best_estimator_score: ",mean_acc_best_estimator_score)
    print("stdev_acc_best_estimator_score: ",stdev_acc_best_estimator_score)

    # stop timer here
    end = time.time()

    # saving training time
    print("time taken: "+str(end - start)+" seconds")
    np.savez(modelSavedPath + "/trainingTime.npz", timeInSeconds = (end - start))
    return best_estimator

  def frozenAndUnfrozenTraining(self, nameOfModel, params, dataset_name, curr_dictionary, freezeEpochs, unfreezeEpochs, classes,
  ds_train, ds_val, ds_test):
    strategy = tf.distribute.MirroredStrategy()
    history = None
    with strategy.scope():
      currentModel = ModelLoader(classes).loadModel(nameOfModel = nameOfModel, kwargs = params, dataset_name = dataset_name, img_size=self.kwargs['img_size'])
    
    # frozen training
    if freezeEpochs > 0:
      history = currentModel.fit(ds_train, epochs = freezeEpochs, validation_data=ds_val, verbose=1) # working
    self.unfreeze_model(model = currentModel, params = params)
    # unfrozen training
    if unfreezeEpochs > 0:
      history = currentModel.fit(ds_train, epochs = unfreezeEpochs, validation_data=ds_val, verbose=1)
    return currentModel, history
  """
  
  """
  def train(self):
    current_ds_test = None
    # Best hyperparameters search
    # load param list
    trainingSet = self.__kwargsTrain['trainingSet']
    input_shape, classes, paramsList = ParamsRangeRetriever(trainingSet).getParams()
    for nameOfModel in self.__kwargsTrain['namesOfModels']:
      print("nameOfModel: "+nameOfModel)
      modelSavedPath = self.__kwargsTrain['savedModelPath'] + "/" + nameOfModel
      print("training on "+nameOfModel)
      bestValAccScore = None
      bestParams = None
      min_val_loss = MIN_VAL_LOSS_CONSTANT
      currentLossScores = []
      currentAccuracyScores = []

      if not os.path.isdir(self.__kwargsTrain['bestParamsJSONPath']):
          os.mkdir(self.__kwargsTrain['bestParamsJSONPath'])
      bestParamsForGivenModelFilePath = self.__kwargsTrain['bestParamsJSONPath'] + "/" + nameOfModel + "_bestParams.json"

      if not self.isExist(bestParamsForGivenModelFilePath):
        print(bestParamsForGivenModelFilePath + " does not exist !!")
        for params in paramsList:
          print("bestParams thus far: {}".format(bestParams))
          print("current param: "+str(params))
          # modelLoader = ModelLoader(classes)
          kFoldIndicesGenerator = KFoldIndicesGenerator()
          all_indices = kFoldIndicesGenerator.generateIndices(self.__kwargsTrain['kFold_n_splits'])

          for curr_dictionary in all_indices:
            (ds_train, ds_val, ds_test), ds_info = DataLoader().load_tfds_data(
              img_size=self.kwargs['img_size'],
              dataset_name = trainingSet,
              range1 = curr_dictionary["train_1_string_range"],
              range2 = curr_dictionary["train_2_string_range"],
              range3 = curr_dictionary["val_string_range"],
              train_idx_range_1 = curr_dictionary["train_idx_range_1"],
              train_idx_range_2 = curr_dictionary["train_idx_range_2"],
              val_idx_range = curr_dictionary["val_idx_range"]
              )
            currentModel, history = self.frozenAndUnfrozenTraining(
              nameOfModel = nameOfModel,
              params = params,
              dataset_name = trainingSet,
              curr_dictionary = curr_dictionary,
              freezeEpochs = self.__kwargsTrain['freezeEpochs'],
              unfreezeEpochs = self.__kwargsTrain['unfreezeEpochs'],
              classes = classes,
              ds_train = ds_train,
              ds_val = ds_val,
              ds_test = ds_test
            )
            
            if current_ds_test is None:
              current_ds_test = ds_test
            self.runTest(current_ds_test = ds_test, model = currentModel)
            currentLossScores.append(history.history['val_loss'][-1])
            currentAccuracyScores.append(history.history['val_accuracy'][-1])

          # basically record best mean val_loss and val_accuracy and associateed params
          if statistics.mean(currentLossScores) < min_val_loss:
            min_val_loss = statistics.mean(currentLossScores)
            bestValAccScore = statistics.mean(currentAccuracyScores)
            bestParams = params
            
        # saving best params
        with open(bestParamsForGivenModelFilePath, 'w') as outfile:
          json.dump(bestParams, outfile)
      else:
        with open(bestParamsForGivenModelFilePath) as jsonFile:
          bestParams = json.load(jsonFile)

      print("=== RETRAINING WITH BEST PARAMS ===")
      for num in self.__kwargsTrain['numberOfModels']:
        print("retraining number "+str(num) + " / "+str(len(self.__kwargsTrain['numberOfModels']) - 1))
        currModelSavedPath = modelSavedPath + ("_num_" + str(num))
        if not os.path.isdir(self.__kwargsTrain['savedModelPath']):
          os.mkdir(self.__kwargsTrain['savedModelPath'])
        if not os.path.isdir(currModelSavedPath):
          os.mkdir(currModelSavedPath)
        best_estimator = None
        if len(os.listdir(currModelSavedPath)) == 0:
          best_estimator = self.retrainWithBestParams(
            nameOfModel = nameOfModel,
            modelSavedPath = currModelSavedPath,
            bestParams = bestParams,
            dataset_name = trainingSet,
            freezeEpochs=self.__kwargsTrain['freezeEpochs'],
            unfreezeEpochs=self.__kwargsTrain['unfreezeEpochs'],
            classes = classes
          )
        else:
          print(currModelSavedPath," not empty !")
          best_estimator = tf.keras.models.load_model(currModelSavedPath)
          print("load best estimator successfully 00")
        
        print("final model test result:")
        if current_ds_test is None:
          (_, _, current_ds_test), _ = DataLoader().load_tfds_data(
                img_size=self.kwargs['img_size'],
                dataset_name = trainingSet,
                range1 = None,
                range2 = None,
                range3 = None,
                train_idx_range_1 = None,
                train_idx_range_2 = None,
                val_idx_range = None,
                mode="test-only"
              )
        self.runTest(current_ds_test = current_ds_test, model = best_estimator, final = True)

  def runTest(self, current_ds_test, model, final = False):
    y_true = np.concatenate([y for x, y in current_ds_test], axis=0)
    y_true=np.argmax(y_true,axis=1)
    y_pred = model.predict(current_ds_test)
    y_pred=np.argmax(y_pred,axis=1)
    print("The accuracy on the testing data : {:.2f}%".format(100 * accuracy_score(y_true, y_pred)))
    if final:
      print(classification_report(y_true, y_pred))