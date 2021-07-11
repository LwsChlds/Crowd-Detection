import argparse

from ast import literal_eval
from callbacks import call_dict
from models.CC import CrowdCounter
from dataset.visdrone import cfg_data
from dataset.run_datasets import make_dataset
from run import run_model, run_transforms
from config import cfg
import numpy as np
import torch

def load_CC_test():

    """
    Load CrowdCounter model net for testing mode
    """
    cc = CrowdCounter([0], cfg.NET)
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc


def run_net(in_file, callbacks):
    """
    Run the model on a given file or folder

    @param in_file: media file or folder of images
    @param callbacks: list of callbacks to be called after every forward operation
    """
    dataset = make_dataset(in_file)

    transforms = run_transforms(cfg_data.MEAN, cfg_data.STD, cfg_data.SIZE)
    dataset.set_transforms(transforms)

    callbacks_list = [(call_dict[call] if type(call) == str else call) for call in callbacks]

    run_model(load_CC_test, dataset, cfg.TEST_BATCH_SIZE, cfg.N_WORKERS, callbacks_list)

def create_onnx():
    """
    Use the model to create an onnx file
    """
    print("Creating onnx file...")
    print("Loading CC model...")
    model = load_CC_test()
    model.eval()
    print("Creating random input values...")
    x = torch.randn(10, 3, 255, 255, device='cuda')
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    print("Exporting model to onnx...")
    torch.onnx.export(model, x, "DroneCrowd110.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
    print("Model created succesfully!")


if __name__ == '__main__':
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Execute a training, an evaluation or run the net on some example')
    parser.add_argument('mode', type=str, help='can be run (run on an image) or onnx (create an onnx file)')
    parser.add_argument('--path', type=str, help='in run mode, the input file or folder to be processed')
    parser.add_argument('--callbacks', type=str,
                        help='List of callbacks, they can be [\'save_callback\', \'count_callback\']')
    args = parser.parse_args()

    if args.callbacks is not None:
        callbacks = literal_eval(args.callbacks)
    else:
        callbacks = []
    if args.mode == 'run':
        run_net(args.path, callbacks)
    elif args.mode == 'onnx':
        create_onnx()
