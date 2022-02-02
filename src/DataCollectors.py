# import tensorflow as tf
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np
import os
import sys
import statistics
import csv
from pathlib import Path
import json
# python3 DataCollectors.py resources/data_collectors.json
# from Utils import substituteString, loggingDictionary
from ConfigParser import ConfigParser
# from DataLoader import DataLoader
from Constants import CSV_HEADER, ATTACKER_PERFORMANCE, ATTACKER_PERFORMANCE_METRICS, \
  DEFENDER_NAME, DEFENDER_PERFORMANCE, BENIGN_DEFENDER_PERFORMANCE_METRICS, ROBUST_DEFENDER_PERFORMANCE_METRICS, \
    BASELINE_PERFORMANCE
def substituteString(my_dict:dict = {}, formatStr:str = None):
    return formatStr.format(**my_dict)
"""
This component is to collect generated data from any tasks for generating report purposes
"""
def _getAllMetricScores(
    dataset_name,
    nameOfModel,
    numOfModel,
    inputModelPath,
    y_pred_path,
    img_size
  ):
  # load test data
  (_, _, ds_test), _ = DataLoader().load_tfds_data( 
    img_size=img_size,
    dataset_name = dataset_name, range1 = None, range2 = None, range3 = None,
    train_idx_range_1 = None, train_idx_range_2 = None, val_idx_range = None,
    mode = ("test-only" if dataset_name == "GTSRB" else ""))
  # true labels
  y_true = np.concatenate([y for x, y in ds_test], axis=0)
  y_true = np.argmax(y_true,axis=1)
  y_pred = None

  my_dict = {
    "dataset_name": dataset_name,
    "nameOfModel": nameOfModel,
    "numOfModel": numOfModel,
  }
  currModelSavedPath = substituteString(
    my_dict = my_dict,
    formatStr = inputModelPath
  )
  y_pred_path = substituteString(
    my_dict = my_dict,
    formatStr = y_pred_path
  )
  
  if not (os.path.exists(y_pred_path)):
    print("y_pred_path "+y_pred_path+" is not yet created...Creating one")
    # load saved model
    best_estimator = tf.keras.models.load_model(currModelSavedPath)
    
    # pred labels
    y_pred = best_estimator.predict(ds_test)
    y_pred = np.argmax(y_pred,axis=1)

    # saving y_pred
    np.savez_compressed(y_pred_path, y_pred = y_pred)
    print("[SUCCESS] saved y_pred to "+y_pred_path)
  else:
    y_pred = np.load(y_pred_path)['y_pred']
    print("[SUCCESS] loaded y_pred from "+y_pred_path)
  return y_true, y_pred

class CollectorBuilder:
  def __init__(self, kwargs):
    self.kwargs = kwargs
    self.collectors = kwargs['collectors']
  
  def _build(self, collectorType = None, collectorParam = None):
    if collectorType == "baseline_classifier_benign_performance":
      return ClassifierBaselineBenignPerformanceDataCollector(collectorKwargs = collectorParam)
    elif collectorType == "baseline_classifier_adversarial_performance":
      return ClassifierBaselineAdversarialPerformanceDataCollector(collectorKwargs = collectorParam)
    elif collectorType == "denoisers":
      return DenoisersPerformanceDataCollector(collectorKwargs = collectorParam)
    else:
      return None

  def buildAndCollect(self):
    for collectorParam in self.collectors:
      collectorType = collectorParam['typeOfCollector']
      collectorObject = self._build(collectorType = collectorType, collectorParam = collectorParam)
      if collectorObject is not None:
        collectorObject.collect()

class DenoisersPerformanceDataCollector:
  def __init__(self, collectorKwargs):
    self.collectorKwargs = collectorKwargs
  
  def writeToCSV(self, csvFileName, data, csv_header=None):
    with open(csvFileName, mode='w') as currFile:
      currFile = csv.writer(currFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      if csv_header:
        currFile.writerow(csv_header)
      for row in data:
        currFile.writerow(row)
    print("[SUCCESS] Data has been written to {} file".format(csvFileName))

  def _getPerformance(self, obj, metrics):
    # TODO: may have bar chart using these values in the future
    values = []
    for metric in metrics:
      values.append(obj[metric])
    return values
  
  def _getDenoiserPerformance(self, obj):
    benignPerformance = self._getPerformance(obj=obj, metrics = BENIGN_DEFENDER_PERFORMANCE_METRICS)
    robustPerformance = self._getPerformance(obj=obj, metrics = ROBUST_DEFENDER_PERFORMANCE_METRICS)
    return benignPerformance, robustPerformance

  def _loadJsonFile(self, filePath):
    loadedData = {}
    with open(filePath, 'r') as fp:
      loadedData = json.load(fp)
    return loadedData

  def _mergeJsonFiles(self, mergedJsonFile, currentLoadedJsonFile):
    keyFormat = self.collectorKwargs['keyFormat']
    dataset_name = self.collectorKwargs['dataset_name']
    nameOfModels = self.collectorKwargs['nameOfModels']
    numOfModels = self.collectorKwargs['numOfModels']
    settingKey = self.collectorKwargs['setting']
    attackerNames = self.collectorKwargs['attackerNames']
    allRows = []
    for nameOfModel in nameOfModels:
      for numOfModel in numOfModels:
        currentRow = []
        modelKey = substituteString(
          my_dict = {
            "nameOfModel": nameOfModel,
            "numOfModel": numOfModel
          },
          formatStr = keyFormat
        )
        # add No Attacks
        currentRow.append(currentLoadedJsonFile[modelKey][BASELINE_PERFORMANCE]['natural_accuracy'])
        for attackerName in attackerNames:
          # add Attack + no denoise
          currentRow.append(currentLoadedJsonFile[modelKey][settingKey][attackerName][ATTACKER_PERFORMANCE]['robust_accuracy'])
          defenderObjects = currentLoadedJsonFile[modelKey][settingKey][attackerName][DEFENDER_NAME]
          for defenderObject in defenderObjects:
            mergedJsonFile[dataset_name][modelKey][settingKey][attackerName][DEFENDER_NAME].append(defenderObject)
            # csv
            for obj in mergedJsonFile[dataset_name][modelKey][settingKey][attackerName][DEFENDER_NAME]:
              currentRow.append(obj[DEFENDER_PERFORMANCE]['robust_accuracy'])
            
        allRows.append(currentRow)

            # TODO: bar chart

    return mergedJsonFile, allRows

  def _saveToCSVFile(self, csvFilePath, mergedJsonFile):
    print("[SUCCESS] saved to {}".format(csvFilePath))
    return None

  def collect(self):
    pathFormat = self.collectorKwargs['pathFormat']
    dataset_name = self.collectorKwargs['dataset_name']
    bestEvalJsonReportsFormat = self.collectorKwargs['bestEvalJsonReportsFormat']
    denoiserNames = self.collectorKwargs['denoiserNames']
    csvFilePath = substituteString(
        my_dict = {
          "dataset_name": dataset_name
          },
        formatStr = self.collectorKwargs['csvFilePath']
      )
    mergedJsonFilePath = substituteString(
        my_dict = {
          "dataset_name": dataset_name
          },
        formatStr = self.collectorKwargs['mergedJsonFilePathFormat']
      )
    mergedJsonFile = {}
    allRows = []
    # if not os.path.exists(mergedJsonFilePath):
    loadedJsonFile = {}
    loadedData = {}

    for denoiserName in denoiserNames:
      print("denoiserName: {}".format(denoiserName))
      bestDenoiserEval = substituteString(
        my_dict = {"denoiserName": denoiserName}, formatStr = bestEvalJsonReportsFormat
      )
      pathToBestDenoiserEval = substituteString(
        my_dict = {
          "dataset_name": dataset_name, 
          "best_eval_json_report": bestDenoiserEval
          },
        formatStr = pathFormat
      )
      loadedJsonFile[denoiserName] = self._loadJsonFile(pathToBestDenoiserEval)
      if mergedJsonFile == {}:
        mergedJsonFile[dataset_name] = loadedJsonFile[denoiserName]
      else:
        mergedJsonFile, allRows = self._mergeJsonFiles(
          mergedJsonFile = mergedJsonFile,
          currentLoadedJsonFile = loadedJsonFile[denoiserName]
        )
    
    with open(mergedJsonFilePath, 'w') as fp:
      json.dump(mergedJsonFile, fp,  indent=4)
      print("[SUCCESS] saved to {}".format(mergedJsonFilePath))
    
    with open(csvFilePath, mode='w') as currFile:
      currFile = csv.writer(currFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      for row in allRows:
        currFile.writerow(row)
      print("[SUCCESS] saved to {}".format(csvFilePath))
    # else:
    #   with open(mergedJsonFilePath, 'r') as fp:
    #     mergedJsonFile = json.load(fp)

    # self._saveToCSVFile(csvFilePath = csvFilePath, mergedJsonFile = mergedJsonFile)

class ClassifierBaselineAdversarialPerformanceDataCollector:
  def __init__(self, collectorKwargs):
    self.collectorKwargs = collectorKwargs

  def collect(self):
    assert('dataset_names' in self.collectorKwargs)
    assert('nameOfModels' in self.collectorKwargs)
    assert('numOfModels' in self.collectorKwargs)

    for dataset_name in self.collectorKwargs['dataset_names']:
      for nameOfModel in self.collectorKwargs['nameOfModels']:
        all_adv_accuracy = []
        all_adv_precision = []
        all_adv_recall = []
        all_adv_f_score = []
        currRowData = []
        for numOfModel in range(self.collectorKwargs['numOfModels']):
          for attackerSavedFilePath in self.collectorKwargs['attackerSavedFilePaths']:
            my_dict = {
              "dataset_name": dataset_name,
              "nameOfModel": nameOfModel,
              "numOfModel": numOfModel,
              "attackerSavedFilePath": attackerSavedFilePath
            }
            currModelSavedPath = substituteString(
              my_dict = my_dict,
              formatStr = self.collectorKwargs['data_path']
            )
            # adv
            adv_acc = "TBD"
            adv_precision = "TBD"
            adv_recall = "TBD"
            adv_f1_score = "TBD"

            if os.path.exists(currModelSavedPath):
              adv_data = np.load(currModelSavedPath)
              benign_acc = adv_data['benign_acc']

              adv_acc = adv_data['adv_acc']
              adv_precision = adv_data['adv_precision'] * 100.0
              adv_recall = adv_data['adv_recall'] * 100.0
              adv_f1_score = adv_data['adv_f1_score'] * 100.0

              mean_l2_distances = statistics.mean(adv_data['l2_distances'])
              mean_l_inf_distances = statistics.mean(adv_data['l_inf_distances'])

              all_adv_accuracy.append(adv_acc)
              all_adv_precision.append(adv_precision)
              all_adv_recall.append(adv_recall)
              all_adv_f_score.append(adv_f1_score)

            # benign
            y_true, y_pred = _getAllMetricScores(
                dataset_name = dataset_name,
                nameOfModel = nameOfModel,
                numOfModel = numOfModel,
                inputModelPath = self.collectorKwargs['inputModelPath'],
                y_pred_path = self.collectorKwargs['y_pred_path'],
                img_size = self.collectorKwargs['img_size']
            )
            
            benign_acc = 100 * accuracy_score(y_true, y_pred)
            benign_precision, benign_recall, benign_fscore,_ = precision_recall_fscore_support(
              y_true = y_true, y_pred = y_pred, average = "macro") # ("macro" if dataset_name == "cifar100" else "micro")
            benign_precision = benign_precision * 100
            benign_recall = benign_recall * 100
            benign_fscore = benign_fscore * 100

            currRowData.append([
              benign_acc, adv_acc, "TBD", "TBD", "TBD",
              benign_precision, adv_precision, "TBD", "TBD", "TBD",
              benign_recall, adv_recall, "TBD", "TBD", "TBD",
              benign_fscore, adv_f1_score, "TBD", "TBD", "TBD"
              ])


        # csv
        my_dict = {
          "dataset_name": dataset_name,
          "nameOfModel": nameOfModel
        }
        csvFilePath = substituteString(
          my_dict = my_dict,
          formatStr = self.collectorKwargs['csvFilePath']
        )
        Path(csvFilePath).mkdir(parents=True, exist_ok=True)

        my_dict = {
          "csvFilePath": csvFilePath
        }
        csvFileName = substituteString(
          my_dict = my_dict,
          formatStr = self.collectorKwargs['csvFileName']
        )
        print("[LOGGING] saving to {}".format(csvFileName))
        with open(csvFileName, mode='w') as currFile:
          currFile = csv.writer(currFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          currFile.writerow(CSV_HEADER)
          for row in currRowData:
            currFile.writerow(row)

        all_adv_accuracy = np.asarray(all_adv_accuracy)
        all_adv_precision = np.asarray(all_adv_precision)
        all_adv_recall = np.asarray(all_adv_recall)
        all_adv_f_score = np.asarray(all_adv_f_score)
        # get mean
        mean_adv_accuracy = statistics.mean(all_adv_accuracy)
        mean_adv_precision = statistics.mean(all_adv_precision)
        mean_adv_recall = statistics.mean(all_adv_recall)
        mean_adv_f_score = statistics.mean(all_adv_f_score)

        # get stdev
        stdev_adv_accuracy = statistics.stdev(all_adv_accuracy)
        stdev_adv_precision = statistics.stdev(all_adv_precision)
        stdev_adv_recall = statistics.stdev(all_adv_recall)
        stdev_adv_f_score = statistics.stdev(all_adv_f_score)

class ClassifierBaselineBenignPerformanceDataCollector:
  def __init__(self, collectorKwargs):
    self.collectorKwargs = collectorKwargs
    
  def collect(self):
    assert('dataset_names' in self.collectorKwargs)
    assert('nameOfModels' in self.collectorKwargs)
    assert('numOfModels' in self.collectorKwargs)
    for dataset_name in self.collectorKwargs['dataset_names']:
      for nameOfModel in self.collectorKwargs['nameOfModels']:
        all_benign_accuracy = []
        all_benign_precision = []
        all_benign_recall = []
        all_benign_f_score = []
        for numOfModel in range(self.collectorKwargs['numOfModels']):
          y_true, y_pred = _getAllMetricScores(
              img_size = self.collectorKwargs['img_size'],
              dataset_name = dataset_name,
              nameOfModel = nameOfModel,
              numOfModel = numOfModel,
              inputModelPath = self.collectorKwargs['inputModelPath'],
              y_pred_path = self.collectorKwargs['y_pred_path']
           )
          benign_acc = 100 * accuracy_score(y_true, y_pred)
          precision, recall, fscore,_ = precision_recall_fscore_support(
             y_true = y_true, y_pred = y_pred, average = "macro") # ("macro" if dataset_name == "cifar100" else "micro")
          all_benign_accuracy.append(benign_acc)
          all_benign_precision.append(precision * 100)
          all_benign_recall.append(recall * 100)
          all_benign_f_score.append(fscore * 100)
        
        # get mean
        mean_benign_accuracy = statistics.mean(all_benign_accuracy)
        mean_benign_precision = statistics.mean(all_benign_precision)
        mean_benign_recall = statistics.mean(all_benign_recall)
        mean_benign_f_score = statistics.mean(all_benign_f_score)

        # get stdev
        stdev_benign_accuracy = statistics.stdev(all_benign_accuracy)
        stdev_benign_precision = statistics.stdev(all_benign_precision)
        stdev_benign_recall = statistics.stdev(all_benign_recall)
        stdev_benign_f_score = statistics.stdev(all_benign_f_score)
        loggingDictionary(myDictionary = {
          "all_benign_accuracy": all_benign_accuracy,
          "all_benign_precision": all_benign_precision,
          "all_benign_recall": all_benign_recall,
          "all_benign_f_score": all_benign_f_score,
        }, context = ("[INFO] For "+dataset_name+" with "+nameOfModel+" model"))


inputConfigFile = sys.argv[1:][0]
configParser = ConfigParser()
data = configParser.parse(inputConfigFile)
# os.environ["CUDA_VISIBLE_DEVICES"] = data['gpu']
# suppress warning when loading model
# tf.get_logger().setLevel('ERROR')
CollectorBuilder(kwargs=data).buildAndCollect()
