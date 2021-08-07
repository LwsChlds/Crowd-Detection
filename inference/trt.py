#!/usr/bin/env python3
# Import standard modules
import argparse
import sys
# Import Computer Vision Modules
import cv2
from PIL import Image
from PIL import ImageFont, ImageDraw
import numpy as np
# Import cuda and TensorRT
import pycuda.autoinit  # This is needed for initializing CUDA driver
import tensorrt as trt
import pycuda.driver as cuda
import dbscan as post
import preprocessing as pre
import configparser
from mqtt import MQTTPublisher
# import common modules
sys.path.append("..")


def _allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers
    # (i.e. won't be swapped to disk) to hold host inputs/outputs.
    host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
        [], [], [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        # Allocate device memory for inputs and outputs.
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, stream


def detect(frame, host_inputs, cuda_inputs, stream, context, bindings, host_outputs, cuda_outputs):
    # Transfer input data to the GPU.
    np.copyto(host_inputs[0], frame.ravel())
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    # Run inference.
    context.execute_async(
        batch_size=1,
        bindings=bindings,
        stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    #cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    # Synchronize the stream
    stream.synchronize()
    output = host_outputs[0]
    # Return the host output.
    return output


def inference_trt(input, model_path, output, Length, Height, epsilon, min_samples, outputFile, interval, outputTF, overlay):
    """
    Function to perform inference and benchmarking
    :param input: (str) the media file to run inference on
    :param model_path: (str) the onnx engine file
    :param output: (str) the name of the output image/s
    :param Length: (int) the number of pixels in the length of the input/outputs
    :param Height: (int) the number of pixels in the height of the input/outputs
    :param epsilon: (int) the distance between two points to be considered neighbors 
    :param min_samples: (int) minimum number of neighbors a given point should have in order to be classified as a core point
    :param outputFile: (str) the name of the file the image/s should be output into
    :param interval: (int) the interval of what frames should be processed in a video file
    :param outputTF: (int) if output heatmap images should be produced or not
    """
   print("Opening onnx engine into trt...")
    trt_logger = trt.Logger(trt.Logger.INFO)
    # load plugins
    trt.init_libnvinfer_plugins(trt_logger, '')
    # load_trt_model
    # mqtt = MQTTPublisher()
    with open(model_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    # allocate CUDA resources for inference
    try:
        context = engine.create_execution_context()
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, stream = _allocate_buffers(engine)
    except Exception as e:
        raise RuntimeError('fail to allocate CUDA resources') from e
    print("Starting inference...")
    if input.endswith('.jpg'):
        # Load and pre-process image
        print("Running from a image input")
        frame = pre.preprocess(input, Height, Length)
        out = detect(frame, host_inputs, cuda_inputs, stream, context, bindings, host_outputs, cuda_outputs)
        out = out.reshape((Height, Length))
        detection, labels = post.postprocess(out, epsilon=epsilon, min_samples=min_samples)
        if (outputTF == 1 and detection is not None and labels is not None):
            post.saveIMG(detection, labels, output, frame, overlay)
    else:
        print("Running from a video input on every " + str(interval) + " frames")
        capture = cv2.VideoCapture(input)
        frameNr = 0
        arrNum = 0
        outputNum = np.full(shape=(10), fill_value=0.0)
        pastVal = np.copy(outputNum, order='K') + 1 
        while (True):
 
            success, frame = capture.read()
 
            if success:
                if (frameNr/interval).is_integer():
                    sum = 0
                    if (frameNr/10).is_integer():
                        arrNum = 0
                        if frameNr > 10:
                            print(str(np.mean(outputNum)/36) + " " + str(float(np.mean(outputNum))/float(np.mean(pastVal))*100) + "%" )
                            # mqtt.publish(str(np.mean(outputNum)/36) + " " + str(float(np.mean(outputNum))/float(np.mean(pastVal))*100) + "%" )
                        pastVal = np.copy(outputNum, order='K') + 1
                    pro_frame = pre.preprocess(frame, Height, Length)
                    out = detect(pro_frame, host_inputs, cuda_inputs, stream, context, bindings, host_outputs, cuda_outputs)
                    out = out.reshape((Height, Length))
                    detection, labels = post.postprocess(out, epsilon=epsilon, min_samples=min_samples)
                    for i in range(len(np.unique(labels))):
                        sum += int(np.sum(np.array(labels) == i, axis=0))
                    outputNum[arrNum] = sum
                    
                    if (outputTF == 1 and detection is not None and labels is not None):
                        if outputFile != None:
                            post.saveIMG(detection, labels, str(outputFile + "/" + output + str(int(frameNr/interval)) + ".jpg"), Length, Height, frame, overlay)
                        else:
                            post.saveIMG(detection, labels, str(output + str(int(frameNr/interval))), Length, Height, frame, overlay)
 
            else:
                break
            arrNum += 1
            frameNr = frameNr+1
 
        capture.release()
        
        
def main():
    """
    Main method to run
    """
    config = configparser.ConfigParser()
    config.read('config.txt')
    print("Loading values from config.txt...")
    if config.get('properties', 'mediaIn', fallback=0) != 0:
        mediaIn = str(config.get('properties', 'mediaIn'))
    else:
        print("No mediaIn was found in the spec file")
    if config.get('properties', 'outputIMG', fallback=0) != 0:
        outputIMG = str(config.get('properties', 'outputIMG'))
    else:
        print("No outputIMG was found in the spec file")
    if config.get('properties', 'interval', fallback=0) != 0:
        interval = int(config.get('properties', 'interval'))
    else:
        interval = 10
    if config.get('properties', 'outputTF', fallback=0) != 0:
        outputTF = int(config.get('properties', 'outputTF'))
    else:
        outputTF = 0
    if config.get('properties', 'onnxEngine', fallback=0) != 0:
        onnxEngine = str(config.get('properties', 'onnxEngine'))
    else:
        print("No onnxEngine was found in the spec file")
    if config.get('properties', 'Length', fallback=0) != 0:
        Length = int(config.get('properties', 'Length'))
    else:
        Length = 960
    if config.get('properties', 'Height', fallback=0) != 0:
        Height = int(config.get('properties', 'Height'))
    else:
        Height = 540
    if config.get('properties', 'epsilon', fallback=0) != 0:
        epsilon = int(config.get('properties', 'epsilon'))
    else:
        epsilon = 40
    if config.get('properties', 'min_samples', fallback=0) != 0:
        min_samples = int(config.get('properties', 'min_samples'))
    else:
        min_samples = 1500
    if config.get('properties', 'outputFile', fallback=0) != 0:
        outputFile = str(config.get('properties', 'outputFile'))
    else:
        outputFile = None
    if config.get('properties', 'overlay', fallback=0) != 0:
        overlay = int(config.get('properties', 'overlay'))
    else:
        overlay = 0
    
​
    inference_trt(mediaIn, onnxEngine, outputIMG, Length, Height, epsilon, min_samples, outputFile, interval, outputTF, overlay)
    print("Completed inference")
    
if __name__ == "__main__":
    main()
