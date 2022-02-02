import os
import matplotlib.pyplot as plt
def substituteString(my_dict:dict = {}, formatStr:str = None):
    return formatStr.format(**my_dict)

def generateSamples(data, path):
  if not (os.path.exists(path)):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(data[i].astype("uint8"))
    plt.savefig(path)

def loggingDictionary(myDictionary:dict = {}, context = None):
  if context is not None:
    print(context)
  for key, value in myDictionary.items():
    print(str(key) + " is " + str(value))
  print('\n')
  