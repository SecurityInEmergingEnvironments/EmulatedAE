import tensorflow as tf
from tensorflow.keras import layers
from DataAugmentationLayers import cifar100_img_augmentation, cifar10_img_augmentation, GTSRB_img_augmentation

"""
This component is to construct models from pre-trained models, image augmentation layer, and logits
"""
class ModelLoader:
  def __init__(self, classes):
    self.classes = classes
  def loadModel(self, nameOfModel, kwargs, dataset_name, img_size):
    # building model
    inputs = layers.Input(shape=(img_size, img_size, 3))

    img_augmentation = tf.keras.Sequential(
      self.getDataAugmentationModel(dataset_name = dataset_name),
      name="img_augmentation",
    )
    x = img_augmentation(inputs)

    model = None
    if (nameOfModel == "VGG19"):
      model = self.getVGG19(x)
    elif (nameOfModel == "ResNet50"):
      model = self.getResNet50(x)
    elif (nameOfModel == "EfficientNetB0"):
      model = self.getEfficientNetB0(x)
    elif (nameOfModel == "ResNet101"):
      model = self.getResNet101(x)
    

    # Freeze the pretrained weights
    model.trainable = False
    # Rebuild top
    if (kwargs['pooling'] == "avg"):
      x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    elif (kwargs['pooling'] == "max"):
      x = layers.GlobalMaxPool2D(name="max_pool")(model.output)
    else:
      raise Exception("unknow pooling: "+str(kwargs['pooling']))
    x = layers.BatchNormalization()(x)

    top_dropout_rate = kwargs['top_dropout_rate']
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(self.classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name=nameOfModel)
    optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs["optimizer_learning_rate"])
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
  
  def getDataAugmentationModel(self, dataset_name):
    if (dataset_name == "cifar100"):
      return cifar100_img_augmentation
    elif (dataset_name == "GTSRB"):
      return GTSRB_img_augmentation
    elif (dataset_name == "cifar10"):
      return cifar10_img_augmentation
    else:
      raise Exception("no data augmentation layer for "+dataset_name+" yet")

  def getVGG19(self, x):
    return tf.keras.applications.VGG19 (
        input_tensor = x,
        include_top = False,
        weights = "imagenet"
    )

  def getResNet50(self, x):
    return tf.keras.applications.ResNet50 (
        include_top = False,
        input_tensor = x,
        weights = "imagenet"
    )
  
  def getResNet101(self, x):
    return tf.keras.applications.ResNet101 (
        include_top = False,
        input_tensor = x,
        weights = "imagenet"
    )

  def getEfficientNetB0(self, x):
    return tf.keras.applications.efficientnet.EfficientNetB0 (
      # drop_connect_rate=0.4 if regularization required
        include_top = False,
        input_tensor = x,
        weights = "imagenet"
    )
  
  def getEfficientNetB3(self, x):
    return tf.keras.applications.efficientnet.EfficientNetB3(
      include_top = False,
      input_tensor = x,
      weights = "imagenet"
    )
