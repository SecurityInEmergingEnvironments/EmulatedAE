import sys, getopt
import os
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoader

class ClassifierVisualizer:
  # TODO: TBD
  def __init__(self, kwargs = None):
    self.kwargs = kwargs
  
  def generateImages(self, data, name):
    # visulize benign examples
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(data[i].astype("uint8"))
    plt.savefig("images/"+name+"_224_224_bilinear.png")
    return None

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  (_, _, ds_test), _ = DataLoader().load_tfds_data(
    dataset_name = "cifar100",
    range1 = None,
    range2 = None,
    range3 = None,
    train_idx_range_1 = None,
    train_idx_range_2 = None,
    val_idx_range = None,
    mode="test-only"
  )
  x_test = np.concatenate([x for x, y in ds_test], axis=0)
  ClassifierVisualizer().generateImages(data = x_test, name = "cifar100")


main()
