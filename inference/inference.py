import numpy as np
import preprocessing as pre
import dbscan as post
import onnxruntime as rt
import configparser

config = configparser.ConfigParser()
config.read('config.txt')

# if the values are present in the config file read from file if not return an error
if config.get('properties', 'mediaIn', fallback=0) != 0:
    mediaIn = str(config.get('properties', 'mediaIn'))
else:
    print("No mediaIn was found in the spec file")
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
print("Running onnxRuntime")
session = rt.InferenceSession(onnx, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: normalised.astype(np.float32)})
print("Postprocessing data")
detection, labels = post.postprocess(np.array(result)[0][0][0], epsilon=epsilon, min_samples=min_samples)
print("Saving detection in " + outputIMG)
post.saveIMG(detection, labels, outputIMG, Length, Height, inputIMG, overlay=0)
