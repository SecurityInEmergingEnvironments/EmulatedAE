import json

"""
This component is to load json configuration into an dictionary
"""
class ConfigParser:
  def parse(self, path : str = None) -> str:
    with open(path) as configFile:
      data = json.load(configFile)
    return data