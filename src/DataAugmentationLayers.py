from tensorflow.keras import layers

cifar100_img_augmentation = [
  layers.experimental.preprocessing.RandomRotation(factor=0.1),
  layers.experimental.preprocessing.RandomZoom(height_factor = (0.05,0.1)), # [+5%, +10%]
  layers.experimental.preprocessing.RandomFlip("horizontal")
]

cifar10_img_augmentation = [
  layers.experimental.preprocessing.RandomRotation(factor=0.1),
  layers.experimental.preprocessing.RandomZoom(height_factor = (0.05,0.1)), # [+5%, +10%]
  layers.experimental.preprocessing.RandomFlip("horizontal")
]

GTSRB_img_augmentation = [
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.experimental.preprocessing.RandomZoom(height_factor = (0.05,0.1)), # [+5%, +10%]
        layers.experimental.preprocessing.RandomContrast(factor=0.1)
]