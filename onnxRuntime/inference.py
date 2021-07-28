import numpy as np
import preprocessing as pre
import dbscan as post
from PIL import Image
import onnxruntime as rt
import json
import configparser

config = configparser.ConfigParser()
config.read('config.txt')

# if the values are present in the config file read from file if not return an error
if config.get('properties', 'inputIMG', fallback=0) != 0:
    inputIMG = str(config.get('properties', 'inputIMG'))
else:
    print("No inputIMG was found in the spec file")
if config.get('properties', 'outputIMG', fallback=0) != 0:
    outputIMG = str(config.get('properties', 'outputIMG'))
else:
    print("No outputIMG was found in the spec file")
if config.get('properties', 'onnx', fallback=0) != 0:
    onnx = str(config.get('properties', 'onnx'))
else:
    print("No onnx file was found in the spec file")

Length = 960 # the amount of pixels the input/output length contains
Height = 540 # the amount of pixels the input/output height contains
epsilon = 20
min_samples = 500

# if the values are present in the config file read from file
# if they are not present it uses the system default values instead
if config.get('properties', 'Length', fallback=0) != 0:
    Length = int(config.get('properties', 'Length'))
if config.get('properties', 'Height', fallback=0) != 0:
    Height = int(config.get('properties', 'Height'))
if config.get('properties', 'epsilon', fallback=0) != 0:
    epsilon = int(config.get('properties', 'epsilon'))
if config.get('properties', 'min_samples', fallback=0) != 0:
    min_samples = int(config.get('properties', 'min_samples'))

print("Preprocessing data")

# normalising the data before using onnxRuntime
normalised = pre.preprocess(inputIMG, Height, Length)

# running onnxRuntime
data = json.dumps({'data': (normalised).tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
print("Running onnxRuntime")
session = rt.InferenceSession(onnx, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: data})
print("Postprocessing data")
post.postprocess(np.array(result)[0][0][0], epsilon=epsilon, min_samples=min_samples, outputIMG=outputIMG)
