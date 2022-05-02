# import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np
import os
import sys
import statistics
import csv
from pathlib import Path
import json
import numpy as np
# python3 DataCollectors.py resources/data_collectors.json
# from Utils import substituteString, loggingDictionary
from ConfigParser import ConfigParser
from collections import defaultdict
from DataLoader import DataLoader
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
  
  # def _getDenoiserPerformance(self, obj):
  #   benignPerformance = self._getPerformance(obj=obj, metrics = BENIGN_DEFENDER_PERFORMANCE_METRICS)
  #   robustPerformance = self._getPerformance(obj=obj, metrics = ROBUST_DEFENDER_PERFORMANCE_METRICS)
  #   return benignPerformance, robustPerformance

  def _loadJsonFile(self, filePath):
    loadedData = {}
    with open(filePath, 'r') as fp:
      loadedData = json.load(fp)
    return loadedData

  def _mergeJsonFiles(self, mergedJsonFile, currentLoadedJsonFile):
    attackerMetric = self.collectorKwargs['attacker_metric']
    defenseMetric = self.collectorKwargs['defense_metric']
    keyFormat = self.collectorKwargs['keyFormat']
    dataset_name = self.collectorKwargs['dataset_name']
    nameOfModels = self.collectorKwargs['nameOfModels']
    numOfModels = self.collectorKwargs['numOfModels']
    settingKey = self.collectorKwargs['setting']
    attackerNames = self.collectorKwargs['attackerNames']
    allRows = []
    timeDict = defaultdict(list)

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
          currentRow.append(currentLoadedJsonFile[modelKey][settingKey][attackerName][ATTACKER_PERFORMANCE][attackerMetric]) # 'robust_f1-score'
          defenderObjects = currentLoadedJsonFile[modelKey][settingKey][attackerName][DEFENDER_NAME]
          for defenderObject in defenderObjects:
            mergedJsonFile[dataset_name][modelKey][settingKey][attackerName][DEFENDER_NAME].append(defenderObject)
            # csv
            for obj in mergedJsonFile[dataset_name][modelKey][settingKey][attackerName][DEFENDER_NAME]:
              if 'nameOfDefenders' in obj:
                timeDict[obj['nameOfDefenders']] = obj[DEFENDER_PERFORMANCE]['inference_elapsed_time_per_1000_in_s']
              elif 'nameOfDefender' in obj:
                timeDict[obj['nameOfDefender']] = obj[DEFENDER_PERFORMANCE]['inference_elapsed_time_per_1000_in_s']
              currentRow.append(obj[DEFENDER_PERFORMANCE][defenseMetric]) # 'robust_f1-score' with defender
            
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
    
    #csv
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
    result_dict = {}
    for data_path in self.collectorKwargs['data_path']:
      for dataset_name in self.collectorKwargs['dataset_names']:
        (_, _, ds_test), _ = DataLoader().load_tfds_data( 
          img_size=224,
          dataset_name = dataset_name, range1 = None, range2 = None, range3 = None,
          train_idx_range_1 = None, train_idx_range_2 = None, val_idx_range = None,
          mode = ("test-only" if dataset_name == "GTSRB" else ""))
        y_true = np.concatenate([y for x, y in ds_test], axis=0)
        y_true = np.argmax(y_true,axis=1)
        for nameOfModel in self.collectorKwargs['nameOfModels']:
          # all_adv_accuracy = []
          # all_adv_precision = []
          # all_adv_recall = []
          
          currRowData = []
          for attackerSavedFilePath in self.collectorKwargs['attackerSavedFilePaths']:
            all_adv_f_score, all_benign_f_score = [],[]
            # print("[DEBUG] numOfModel: {}".format(numOfModel))
            for numOfModel in (self.collectorKwargs['numOfModels']):
              key = "{}_{}".format(nameOfModel,numOfModel)
              if key not in result_dict:
                result_dict[key] = []

              my_dict = {
                "dataset_name": dataset_name,
                "nameOfModel": nameOfModel,
                "numOfModel": numOfModel,
                "attackerSavedFilePath": attackerSavedFilePath
              }
              # print("my_dict: {}".format(my_dict))
              # print("all_adv_f_score: {}".format(all_adv_f_score))
              currModelSavedPath = substituteString(
                my_dict = my_dict,
                formatStr = data_path #self.collectorKwargs['data_path']
              )
              # adv
              adv_acc = "-"
              adv_precision = "-"
              adv_recall = "-"
              adv_f1_score = "-"
              
              if os.path.exists(currModelSavedPath):
                adv_data = np.load(currModelSavedPath)
                adv_y_preds = adv_data['adv_y_preds']
                benign_y_preds = adv_data['benign_y_preds']
                adv_f1_score = adv_data['adv_f1_score'] * 100.0
                _, _, benign_f1_score, _ = precision_recall_fscore_support(y_true, benign_y_preds, average = "macro")

                result_dict[key].append(adv_f1_score)
                all_adv_f_score.append(adv_f1_score)
                all_benign_f_score.append(benign_f1_score)
              else:
                raise Exception("[DEBUG][Not_Found] currModelSavedPath: {}".format(currModelSavedPath))
                
            mean_adv_f_score = np.mean(all_adv_f_score)
            mean_benign_f_score = np.mean(all_benign_f_score)
            print("\ndata_path: {}, attackerSavedFilePath: {}, dataset: {}, nameOfModel: {}, mean_adv_f_score: {}, mean_benign_f_score: {}".format(data_path,attackerSavedFilePath, dataset_name, nameOfModel, mean_adv_f_score, mean_benign_f_score))
            print("all_adv_f_score: {}".format(all_adv_f_score))
            
            
    print("result_dict: {}".format(result_dict))
    allRows = []
    for k,currentRow in result_dict.items():
      allRows.append(currentRow)

    csvFilePath = substituteString(
        my_dict = {
          "dataset_name": dataset_name
          },
        formatStr = self.collectorKwargs['csvFilePath']
      )
    #csv
    with open(csvFilePath, mode='w') as currFile:
      currFile = csv.writer(currFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      for row in allRows:
        currFile.writerow(row)
      print("[SUCCESS] saved to {}".format(csvFilePath))
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
CollectorBuilder(kwargs=data).buildAndCollect()
