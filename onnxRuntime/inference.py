import numpy as np
import preprocessing as pre
import postprocessing as post
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
    print("No onnx file was foudn in the spec file")
print("Preprocessing data")

# normalising the data before using onnxRuntime
normalised = pre.preprocess(inputIMG, config)

# running onnxRuntime
data = json.dumps({'data': (normalised).tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
print("Running onnxRuntime")
session = rt.InferenceSession(onnx, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: data})
print("Postprocessing data")
post.postprocess(np.array(result), outputIMG, config)
