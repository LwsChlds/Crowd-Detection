import torch
import torchvision
from dataset.run_datasets import VideoDataset
import sys
import numpy
import math
import cv2
import onnx
import onnxruntime
import json

def load_txt_outputs(dataset, batch_size, n_workers, callbacks, file):
    """
    Loads output values from a previous model being run

    @param dataset: torch Dataset object
    @param batch_size: batch size for parallel computing
    @param n_workers: n° of workers for parallel process
    @param callbacks: list of callback function to execute after each item is processed
    @param file: the txt file being loaded
    @return:
    """
    print('Running from txt file')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running using device: ' + str(device))
    # Setup the data loader
    if type(dataset) == VideoDataset:
        n_workers = 0
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    with torch.no_grad():
        for input, other in data_loader:
            input = input.to(device)
            input = input.to('cpu') 
            predictions = load_inputs(file)

            for i in range(input.shape[0]):
                for callback in callbacks:
                    callback(input[i], predictions[i], other[i])

def run_onnx_model(dataset, batch_size, n_workers, callbacks, file):
    """
    Run the model on a given onnx dataset

    @param dataset: torch Dataset object
    @param batch_size: batch size for parallel computing
    @param n_workers: n° of workers for parallel process
    @param callbacks: list of callback function to execute after each item is processed
    @param file: the onnx file to run on
    @return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running using device: ' + str(device))

    # Setup the data loader
    if type(dataset) == VideoDataset:
        n_workers = 0
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )
    # Make sure the model is set to evaluation mode
    with torch.no_grad():
        for input, other in data_loader:
            input = input.to(device)
            input = input.to('cpu') 
            data = json.dumps({'data': input.tolist()})
            data = numpy.array(json.loads(data)['data']).astype('float32')
            print("Running onnxruntime")
            session = onnxruntime.InferenceSession("DroneCrowd11-540x960.onnx", None)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: data})
            print(numpy.array(result).shape)
            numpy.set_printoptions(threshold=sys.maxsize)
            numpy.savetxt('onnxData.txt', numpy.array(result)[0][0][0], fmt='%f')
            print('Saved output into onnxData.txt')
            predictions = load_inputs('onnxData.txt')

            for i in range(input.shape[0]):
                for callback in callbacks:
                    callback(input[i], predictions[i], other[i])

def load_inputs(file):
    """
    Loads values produced by numpy.savetxt to test the outputs of other models
    """
    predictions = (numpy.loadtxt(file, dtype=float))
    predictions = predictions[numpy.newaxis, ...]
    predictions = predictions[numpy.newaxis, ...]
    predictions = torch.from_numpy(predictions)
    print("Values loaded successfully")
    return predictions
