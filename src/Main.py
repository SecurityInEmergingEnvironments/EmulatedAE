import sys
from ConfigParser import ConfigParser
from DataLoader import DataLoader
from Denoiser import Denoiser

import os
import tensorflow as tf
def main(argv):
  inputConfigFile = argv[0]
  configParser = ConfigParser()
  data = configParser.parse(inputConfigFile)
  os.environ["CUDA_VISIBLE_DEVICES"] = data['gpu']
  tf.get_logger().setLevel('ERROR')
  if 'Train' in data['mode']:
    from Trainer import ClassifierTrainer
    classifierTrainer = ClassifierTrainer(data)
    classifierTrainer.train()
  
  if 'AttackBase' in data['mode']:
    from Attacker import Attacker
    attacker = Attacker(data)
    attacker.runAttack()

  if 'Denoise' in data['mode']:
    denoiser = Denoiser(kwargs = data)
    denoiser.denoise()

main(sys.argv[1:])