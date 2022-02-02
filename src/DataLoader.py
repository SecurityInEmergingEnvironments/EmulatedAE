import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils
import tensorflow_datasets as tfds
from Constants import BATCH_SIZE, CIFAR100_ORIGINAL_SIZE, GTSRB_RAW_DATA_PATH, CIFAR100_LABEL_TYPE, CIFAR10_LABEL_TYPE
import os
from sklearn.model_selection import train_test_split

batch_size = BATCH_SIZE

"""
This component is to download the data and resize it properly
"""
class DataLoader:
  def load_tfds_data(self, img_size,
  dataset_name: str = '', range1 = None, range2 = None, range3 = None,
  train_idx_range_1 = None, train_idx_range_2 = None, val_idx_range = None,
  mode = ""):
    (ds_train_one, ds_train_two, ds_val, ds_test), ds_info = (None, None, None, None), None
    ds_train = None

    if (dataset_name == "cifar100" or dataset_name == "cifar10"):
      label_type = None
      if dataset_name == "cifar100":
        label_type = CIFAR100_LABEL_TYPE
      elif dataset_name == "cifar10":
        label_type = CIFAR10_LABEL_TYPE
      else:
        raise Exception("{}'s label is not supported".format(dataset_name))

      if (range1 == None and range2 is not None and range3 is not None):
        (ds_train_one, ds_val, ds_test), ds_info = tfds.load(
              dataset_name, split=[range2, range3, "test"], with_info=True)
      elif (range2 == None and range1 is not None and range3 is not None):
        (ds_train_two, ds_val, ds_test), ds_info = tfds.load(
              dataset_name, split=[range1, range3, "test"], with_info=True)
      elif (range1 is not None and range2 is not None and range3 is not None):
        (ds_train_one, ds_train_two, ds_val, ds_test), ds_info = tfds.load(
              dataset_name, split=[range1, range2, range3, "test"], with_info=True)
      elif (range1 is None and range2 is None and range3 is None):
        print("all cifar100 ranges are None")
        (ds_train, ds_test), ds_info = tfds.load(
              dataset_name, split=["train","test"], with_info=True)
      
      
      if (ds_train_one is None and ds_train_two is not None):
        ds_train = ds_train_two
      elif (ds_train_two is None and ds_train_one is not None):
        ds_train = ds_train_one
      elif (ds_train_one is not None and ds_train_two is not None):
        ds_train = ds_train_one.concatenate(ds_train_two)
      
      NUM_CLASSES = ds_info.features[label_type].num_classes
      size = (img_size, img_size)
      if (ds_train is not None):
        # why bilinear? Recommended as it is reasonably good-quality, faster than Lanczos3Kernel, which high-quality, particularly in upsampling
        # source:https://www.tensorflow.org/api_docs/python/tf/image/resize
        if img_size == CIFAR100_ORIGINAL_SIZE:
          ds_train = ds_train.map(lambda dictionary: (dictionary['image'], dictionary[label_type]))
        else:
          ds_train = ds_train.map(lambda dictionary: (tf.image.resize(dictionary['image'], size, method = "bilinear"), dictionary[label_type]))
      if (ds_val is not None):
        if img_size == CIFAR100_ORIGINAL_SIZE:
          ds_val = ds_val.map(lambda dictionary: (dictionary['image'], dictionary[label_type]))
        else:
          ds_val = ds_val.map(lambda dictionary: (tf.image.resize(dictionary['image'], size, method = "bilinear"), dictionary[label_type]))
      if (ds_test is not None):
        if img_size == CIFAR100_ORIGINAL_SIZE:
          ds_test = ds_test.map(lambda dictionary: (dictionary['image'], dictionary[label_type]))
        else:
          ds_test = ds_test.map(lambda dictionary: (tf.image.resize(dictionary['image'], size, method = "bilinear"), dictionary[label_type]))


      # One-hot / categorical encoding
      def input_preprocess(image, label):
          label = tf.one_hot(label, NUM_CLASSES)
          return image, label

      if (ds_train is not None):
        ds_train = ds_train.map(
            input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
      if (ds_val is not None):
        ds_val = ds_val.map(input_preprocess)
        ds_val = ds_val.batch(batch_size=batch_size, drop_remainder=True)
      if (ds_test is not None):
        ds_test = ds_test.map(input_preprocess)
        ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)
      
      return (ds_train, ds_val, ds_test), ds_info
    
    elif (dataset_name == "GTSRB"):
      ds_train, ds_val = None, None
      TRAIN_DATA_PATH = GTSRB_RAW_DATA_PATH + "train.npz"
      TEST_DATA_PATH = GTSRB_RAW_DATA_PATH + "test.npz"
      if (mode != "test-only"):
        ds_train, ds_val = self.get_ds_train_val(
          train_data_path = TRAIN_DATA_PATH,
          train_idx_range_1 = train_idx_range_1,
          train_idx_range_2 = train_idx_range_2,
          val_idx_range = val_idx_range)
      ds_test = self.get_ds_test(test_data_path = TEST_DATA_PATH)
      NUM_CLASSES = 43
      size = (img_size, img_size)
      
      if (ds_train is not None):
        ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size, method = "bilinear"), label))
      if (ds_val is not None):
        ds_val = ds_val.map(lambda image, label: (tf.image.resize(image, size, method = "bilinear"), label))
      if (ds_test is not None):
        ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size, method = "bilinear"), label))

      # One-hot / categorical encoding
      def input_preprocess(image, label):
          label = tf.one_hot(label, NUM_CLASSES)
          return image, label

      if (ds_train is not None):
        ds_train = ds_train.map(
            input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
      if (ds_val is not None):
        ds_val = ds_val.map(input_preprocess)
        ds_val = ds_val.batch(batch_size=batch_size, drop_remainder=True)
      if (ds_test is not None):
        ds_test = ds_test.map(input_preprocess)
        ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

      return (ds_train, ds_val, ds_test), None
    else:
      raise Exception("dataset "+dataset_name+" is not supported yet")
  
  def get_ds_train_val(self,train_data_path, train_idx_range_1, train_idx_range_2, val_idx_range):
    print("train_idx_range_1: "+str(train_idx_range_1))
    print("train_idx_range_2: "+str(train_idx_range_2))
    print("val_idx_range: "+str(val_idx_range))

    imgs_path = GTSRB_RAW_DATA_PATH + "Train"
    data_list = []
    data = None
    labels = None
    labels_list = []
    classes_list = 43
    if not (os.path.exists(train_data_path)):
      for i in range(classes_list):
          i_path = os.path.join(imgs_path, str(i)) #0-42
          for img in os.listdir(i_path):
              im = Image.open(i_path +'/'+ img)
              im = im.resize((32, 32))
              im = np.array(im)
              data_list.append(im) # add resized image as matrix
              labels_list.append(i)
      data = np.array(data_list)
      labels = np.array(labels_list)
      np.savez(train_data_path, data = data, labels = labels)
    else:
      raw = np.load(train_data_path)
      data = raw['data']
      labels = raw['labels']
    # split train - validation
    # X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size = 0.1, random_state=42, shuffle = True) # random_state=42 works like a charm
    from sklearn.utils import shuffle
    data, labels = shuffle(data, labels, random_state=42)
    num_rows,_,_,_ = data.shape
    
    if val_idx_range != None:
      X_val = data[int((val_idx_range[0] / 100) * num_rows) : int((val_idx_range[-1] / 100) * num_rows)]
      y_val = labels[int((val_idx_range[0] / 100) * num_rows) : int((val_idx_range[-1] / 100) * num_rows)]
      ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    else:
      ds_val = None

    X_train = np.concatenate(
      (data[int((train_idx_range_1[0] / 100) * num_rows) : int((train_idx_range_1[-1] / 100) * num_rows)],
      data[int((train_idx_range_2[0] / 100) * num_rows) : int((train_idx_range_2[-1] / 100) * num_rows)]),
      axis=0
      )
    y_train = np.concatenate(
      (labels[int((train_idx_range_1[0] / 100) * num_rows) : int((train_idx_range_1[-1] / 100) * num_rows)],
      labels[int((train_idx_range_2[0] / 100) * num_rows) : int((train_idx_range_2[-1] / 100) * num_rows)]),
      axis=0
      )

    # print("X_train shape: "+str(X_train.shape))
    # print("X_val shape: "+str(X_val.shape))

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    
    return ds_train, ds_val

  def get_ds_test(self, test_data_path):
    # print("[DEBUG] running get_ds_test")
    imgs_path = GTSRB_RAW_DATA_PATH
    data_list = []
    labels_list = []
    data = None
    labels = None
    if not (os.path.exists(test_data_path)):
      with open(GTSRB_RAW_DATA_PATH + 'Test.csv', newline='') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
          if idx > 0:
            curr_img_path = row[7]
            curr_img_label = row[6]
            im = Image.open(imgs_path +'/'+ curr_img_path)
            im = im.resize((32, 32))
            im = np.array(im)
            data_list.append(im) # add resized image as matrix
            labels_list.append(int(curr_img_label))
      data = np.array(data_list)
      labels = np.array(labels_list)
      np.savez(test_data_path, data = data, labels = labels)
    else:
      raw = np.load(test_data_path)
      data = raw['data']
      labels = raw['labels']
    ds_test = tf.data.Dataset.from_tensor_slices((data, labels))
    return ds_test
  # def loadData(self, nameOfDataset: str = None):
    # if (nameOfDataset == "cifar100"):
      # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
      # # Parse numbers as floats
      # x_train = x_train.astype('float32')
      # x_test = x_test.astype('float32')

      # # Normalize data
      # x_train = x_train / 255
      # x_test = x_test / 255
      # y_train = np_utils.to_categorical(y_train)
      # y_test = np_utils.to_categorical(y_test)
      # assert x_train.shape == (50000, 32, 32, 3)
      # assert x_test.shape == (10000, 32, 32, 3)
      # assert y_train.shape == (50000, 10)
      # assert y_test.shape == (10000, 10)

    # return [x_train, y_train, x_test, y_test]