# Modified from hmdegopose: https://github.com/ybkscht/EfficientPose

import torch
import argparse
import os
import sys
import multiprocessing as mp
from generators.colibri import ColibriGenerator
from backbone import HMDEgoPose
from collections import OrderedDict
from train import TrainModelWithLoss

from eval.common import evaluate, evaluate_model

# CUDA for PyTorch
num_gpu = torch.cuda.device_count()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn_colibri', type=str, help='training dataset to use [syn_colibri, real_colibri]')
parser.add_argument('--fold', default='0', type=int, help='fold to use for training and evaluation [0, 1, 2, 3, 4]')
parser.add_argument('--iter', default='1', type=int, help='number of iterations for translation, rotation, and hands submodules [0, 1]')
parser.add_argument('--img_size', default='256,256', type=str, help='image size [(256,256), (512,512)]')
parser.add_argument('--batch_size', default=1, type=int, help='eval batch size')
parser.add_argument('--phi', default=0, type=int, help='efficient det backbone')
parser.add_argument('--n_workers', default=mp.cpu_count(), type=int, help='number of cpus to use')
parser.add_argument('--weights', default="", type=str, help='weights to use')
parser.add_argument('--is_save_images', default=False, type=bool, help='whether or not to save images from testing, requires batch size of 1')
parser.add_argument('--score-threshold', type=float, default=0.5, help='score threshold for non max suppresion')
parser.add_argument('--validation-image-save-path', type=str, default='predictions/test/', help='path where to save the predicted validation images after each epoch')

args = parser.parse_args()

# Default parameters for network training
params = {'dataset': args.dataset,
          'fold': args.fold,
          'iter': args.iter,
          'img_size': tuple(map(int, str(args.img_size).split(','))),
          'batch_size': args.batch_size,
          'phi': args.phi,
          'num_workers': args.n_workers,
          'device': device,
          'use_cuda': use_cuda,
          'log_dir': 'logs/',
          'weights': args.weights,
          'weights_dir': 'weights/efficientdet-d0.pth',
          'validation_image_save_path': args.validation_image_save_path,
          'is_save_images': args.is_save_images,
          'is_debug_mano': False}

print("\n", params)


def main():
    """
    Evaluate an HMDEgoPose model.

    Args:
        args: parseargs object containing configuration for the evaluation procedure.
    """
    os.makedirs(params["validation_image_save_path"], exist_ok=True)

    if params["is_save_images"]:
        params["batch_size"] = 1
        print("\nForcing batch size of 1 to allow for image saving.")

    # create the generators
    print("\nCreating the Generators...")
    generator = create_generators(args, params)
    print("Done!")

    # Get rotation param, classes and anchor count
    num_rotation_parameters = generator.get_num_rotation_parameters()
    num_classes = generator.num_classes()
    num_anchors = generator.num_anchors

    print("\nBuilding the Model...")

    # Create the base model
    model = HMDEgoPose(
        params,
        num_classes=1,
        compound_coef=params['phi'],
        onnx_export=True,
        input_sizes=[params['img_size'][0], 640, 768, 896, 1024, 1280, 1280, 1536, 1536])

    # Transfer to device
    if use_cuda:
        # # Multiple GPU support
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    # Load best model weights
    state_dict = torch.load(params["weights"])
    new_state_dict = OrderedDict()
    
    if params['dataset'] == 'real_colibri':
        # For weights trained on real dataset
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            # name = k[13:]  # remove `model.`
            new_state_dict[name] = v

    if params['dataset'] =='syn_colibri':    
        # For weights trained on syn dataset
        for k, v in state_dict.items():
            name = k[13:]  # remove `model.module.`
            new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    print("Loaded weights!")

    # Wrap the model with loss
    model = TrainModelWithLoss(model)
    model.eval()

    evaluate_model(model, generator, params["validation_image_save_path"],
                   params, args.score_threshold,
                   device, verbose=1)


def create_generators(args, params):
    """ Create the data generators.
    Args:
        args: parseargs arguments object.

    Returns:
        Generator

    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }

 # Parameters for data loading
    if params['dataset'] == 'syn_colibri':

        print("\nCreating data generators for the syn_colibri dataset.")

        generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/syn_colibri_v1",
            object_id=1,
            partition="test",
            shuffle_dataset=False,
            shuffle_groups=False,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=False,
            use_6DoF_augmentation=False,
            **common_args)
        
        print("Done creating data generators.")

    elif params['dataset'] == 'real_colibri':

        print("\nCreating data generators for the real_colibri dataset.")

        generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/real_colibri_v1",
            object_id=1,
            partition="test",
            shuffle_dataset=False,
            shuffle_groups=False,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=False,
            use_6DoF_augmentation=False,
            **common_args)
        
        print("Done creating data generators.")

    else:
        print('No dataset selected.')

    return generator

if __name__ == '__main__':
    main()
