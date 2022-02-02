from collections import defaultdict
import json

from Utils import substituteString
"""
This is intended to print out attack results into JSON reports
"""
class ClassifierBaseAttackReport:
  def __init__(self, dataset_name = None):
    self.dataset_name = dataset_name
    self.finalResult = defaultdict(dict)

  def _addDataToFinalResult(self,
    nameOfModel = None,
    package = None):

    # initialize data struct
    if self.dataset_name not in self.finalResult:
      self.finalResult[self.dataset_name] = {}
    if nameOfModel not in self.finalResult[self.dataset_name]:
      self.finalResult[self.dataset_name][nameOfModel] = []
    
    self.finalResult[self.dataset_name][nameOfModel].append(package)

  def addData(self,
    nameOfModel = None,
    numOfModel = None,
    advExAndPredSavedPath = "",
    benign_acc = None,
    adv_acc = None,
    mean_l2_distances = None,
    mean_l_inf_distances = None):
    package = {}
    package['numOfModel'] = numOfModel
    package['advExAndPredSavedPath'] = advExAndPredSavedPath
    package['benign_acc'] = benign_acc
    package['adv_acc'] = adv_acc
    package['mean_l2_distances'] = mean_l2_distances
    package['mean_l_inf_distances'] = mean_l_inf_distances

    self._addDataToFinalResult(nameOfModel = nameOfModel, package = package)
  
  def saveFinalResult(self, path):
    return None
    # with open(path, 'w') as fp:
    #   json.dump(self.finalResult, fp,  indent=4)