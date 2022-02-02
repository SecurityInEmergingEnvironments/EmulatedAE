from sklearn.model_selection import KFold
import numpy as np
x = np.empty((100,1)) # 0 to 100

"""
This is to generate the training and validation indices for cross-validation training.
For example, if k = 5, training indices are [0,20] and [60,100] and validation indices are [20,40] 
"""
class KFoldIndicesGenerator:
  def generateIndices(self, n: int = 0):
      kf = KFold(n_splits = n)
      result = []
      for train_idx, val_idx in kf.split(x):
          train_idx_range_1 = [0, val_idx[0]]
          train_idx_range_2 = [val_idx[-1] + 1, 100]
          val_idx_range = [val_idx[0], val_idx[-1] + 1]

          train_1_string_range =  "train["+str(train_idx_range_1[0])+"%:"+str(train_idx_range_1[-1])+"%]" if (train_idx_range_1[0] != train_idx_range_1[-1]) else None
          train_2_string_range = "train["+str(train_idx_range_2[0])+"%:"+str(train_idx_range_2[-1])+"%]" if (train_idx_range_2[0] != train_idx_range_2[-1]) else None
          val_string_range = "train["+str(val_idx_range[0])+"%:"+str(val_idx_range[-1])+"%]"
          result.append({
            "train_1_string_range":train_1_string_range,
            "train_idx_range_1":train_idx_range_1,
            "train_2_string_range":train_2_string_range,
            "train_idx_range_2":train_idx_range_2,
            "val_string_range":val_string_range,
            "val_idx_range":val_idx_range
          })
      return result