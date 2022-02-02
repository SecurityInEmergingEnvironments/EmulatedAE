from sklearn.model_selection import ParameterGrid

"""
This component is to generate all possible combination of hyperparameter values for
hyperparameters fine-tuning
"""
class ParamsRangeRetriever:
  def __init__(self, nameOfDataset:str = None):
    self.nameOfDataset = nameOfDataset
    self.paramGrid = {
      "layersToUnfreeze": [0, -20],
      "pooling" : ["max", "avg"],
      "top_dropout_rate" : [0.2, 0.3],
      # "optimizer_learning_rate": ([1e-2] if self.nameOfDataset == "cifar100" else [1e-3])
      "optimizer_learning_rate": ([1e-2] if self.nameOfDataset == "cifar100" else [1e-3])
    }

  def getParams(self):
    input_shape = None
    classes = None
    if (self.nameOfDataset == "cifar100"):
      input_shape = (32, 32, 3)
      classes = 20
    elif (self.nameOfDataset == "GTSRB"):
      input_shape = (30, 30, 3) # img_rows, img_colums, color_channels
      classes = 43
    elif (self.nameOfDataset == "cifar10"):
      input_shape = (32, 32, 3)
      classes = 10
    else:
      raise Exception("{} is not supported by ParamsRangeRetriever yet".format(self.nameOfDataset))
    return input_shape, classes, list(ParameterGrid(self.paramGrid))

class ParamsRangeForNNDenoisers:
  def __init__(self, name = "", resolution = [32, 32, 3]):
    self.name = name
    self.resolution = resolution
  
  def getParams(self):
    if self.resolution == [224, 224, 3]:
      return self._getParamsForRegularResolution()
    else:
      raise Exception("resolution {} not supported yet by ParamsRangeForNNDenoisers".format(self.resolution))
  
  def _getParamsForRegularResolution(self):
    paramGrid = None
    if self.name == "ae":
      paramGrid = {
        "numOfFilters": [128, 256],
        "optimizer": ["adam"],
        "loss_func": ["mse"],
        "optimizerLR": [1e-5, 1e-4],
        "dropout_rate": [0.25, 0.4]
      }
    elif self.name == "vae":
      paramGrid = {
        "latent_dim": [1024],
        "optimizer":['adam'],
        "optimizerLR": [1e-5, 1e-4, 1e-3]
      }
    elif self.name == "unet":
      paramGrid = {
        "loss_func": ["MSE"],
        "start_neurons": [32, 28],
        "optimizer": ["adam"],
        "dropout_rate": [0.25, 0.5]
      }
    else:
      raise Exception("model {} not supported yet by ParamsRangeForNNDenoisers".format(self.name))
    return list(ParameterGrid(paramGrid))