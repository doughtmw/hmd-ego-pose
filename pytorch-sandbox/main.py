import argparse
import multiprocessing as mp
import torch
import tensorflow as tf
import os

# Local imports
from train import train
from backbone import HMDEgoPose
from hmdegopose.utils import count_parameters
# from test import test
from generators.colibri import ColibriGenerator

from ptflops import get_model_complexity_info
from hmdegopose.misc_utils import export_to_onnx, print_size_of_model

# CUDA for PyTorch
num_gpu = torch.cuda.device_count()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# CPU for Tensorflow
# Set CPU as available physical device
tf.config.set_visible_devices([], 'GPU')

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn_colibri', type=str, help='training dataset to use [syn_colibri, real_colibri]')
parser.add_argument('--fold', default='0', type=int, help='fold to use for training and evaluation [0, 1, 2, 3, 4]')
parser.add_argument('--iter', default='0', type=int, help='number of iterations for translation, rotation, and hands submodules [0, 1]')
parser.add_argument('--fine_tune', default=False, type=bool, help='are we performing fine-tuning or not - use SGD instead of Adam [False, True]')
parser.add_argument('--img_size', default='256,256', type=str, help='image size [(256,256), (512,512)]')
parser.add_argument('--freeze_backbone', default=True, type=bool, help='freeze backbone weights [False, True]')
parser.add_argument('--trans6dof', default=True, type=bool, help='6dof transform training set [False, True]')
parser.add_argument('--transcolor', default=True, type=bool, help='color transform training set [False, True]')
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--phi', default=0, type=int, help='efficient det backbone')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for adam')
parser.add_argument('--epo', default=50, type=int, help='epochs to train')
parser.add_argument('--n_workers', default=mp.cpu_count(), type=int, help='number of cpus to use')
parser.add_argument('--validation-image-save-path', type=str, default='predictions/eval/', help='path where to save the predicted validation images after each epoch')
parser.add_argument('--ckpt', default=None, type=str, help='checkpoint to use for beginning training')
args = parser.parse_args()


# Default parameters for network training
params = {'dataset': args.dataset,
          'fold': args.fold,
          'iter': args.iter,
          'fine_tune': args.fine_tune,
          'img_size': tuple(map(int, str(args.img_size).split(','))),
          'freeze_backbone': args.freeze_backbone,
          'use_6dof_transform': args.trans6dof,
          'use_colorspace_transform': args.transcolor,
          'batch_size': args.batch_size,
          'phi': args.phi,
          'num_workers': args.n_workers,
          'learning_rate': args.lr,
          'epochs': args.epo,
          'device': device,
          'use_cuda': use_cuda,
          'log_dir': 'logs/',
          'weights_dir': 'weights/efficientdet-d0.pth',
          'validation_image_save_path': args.validation_image_save_path,
          'ckpt': args.ckpt,
          'is_debug_mano': False}

print("\n", params)

def main():
    print('device: {} num_gpu: {}'.format(device, num_gpu))
    
    common_args = {
        'batch_size': params['batch_size'],
        'phi': params['phi'],
    }

    if params["validation_image_save_path"]:
        os.makedirs(params["validation_image_save_path"], exist_ok=True)
    
    # Parameters for data loading
    if params['dataset'] == 'syn_colibri':

        print("\nCreating data generators for the syn_colibri dataset.")

        training_generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/syn_colibri_v1",
            object_id=1,
            partition="train",
            shuffle_dataset=True,
            shuffle_groups=True,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=params["use_colorspace_transform"],
            use_6DoF_augmentation=params["use_6dof_transform"],
            **common_args)
        
        validation_generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/syn_colibri_v1",
            object_id=1,
            partition="val",
            shuffle_dataset=False,
            shuffle_groups=False,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=False,
            use_6DoF_augmentation=False,
            **common_args)

        print("Done creating data generators.")

    elif params['dataset'] == 'real_colibri':

        print("\nCreating data generators for the real_colibri dataset.")

        training_generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/real_colibri_v1",
            object_id=1,
            partition="train",
            shuffle_dataset=True,
            shuffle_groups=True,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=params["use_colorspace_transform"],
            use_6DoF_augmentation=params["use_6dof_transform"],
            **common_args)
        
        validation_generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/real_colibri_v1",
            object_id=1,
            partition="val",
            shuffle_dataset=False,
            shuffle_groups=False,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=False,
            use_6DoF_augmentation=False,
            **common_args)

        print("Done creating data generators.")

    else:
        print('No dataset selected.')
        
    # Get information on rotation parameter count and classes from train generator
    params["num_rotation_parameters"] = training_generator.get_num_rotation_parameters()
    params["num_classes"] = training_generator.num_classes()
    params["num_anchors"] = training_generator.num_anchors

    print("num_rot_params:", params["num_rotation_parameters"],
          " num_classes:", params["num_classes"],
          " num_anchors:", params["num_anchors"])

    # Create the model to be used for training
    # pull these parameters from dataset...
    model = HMDEgoPose(
        params,
        num_classes=1,
        compound_coef=params['phi'],
        onnx_export=True,
        input_sizes=[params['img_size'][0], 640, 768, 896, 1024, 1280, 1280, 1536, 1536])
    print("\nSuccessfully created the Hmd-EgoPose backbone.")

    # Count parameters in model
    print('HMD-EgoPose parameter count:', count_parameters(model))

    # Get size, FLOPS and parameters in current model
    print_size_of_model(model)
    model_flops, model_params = get_model_complexity_info(
        model, input_res=(3, params['img_size'][0], params['img_size'][1]),
        as_strings=True,
        print_per_layer_stat=False)
    print('Model flops:  ' + model_flops)
    print('Model params: ' + model_params)

    # Load the pretrained efficientdet weights
    model.init_backbone(params["weights_dir"])
    print("\nSuccessfully loaded pretrained weights for the efficientdet backbone.")

    # freeze backbone layers
    if params['freeze_backbone']:
        # 227, 329, 329, 374, 464, 566, 656
        freeze_total = [227, 329, 329, 374, 464, 566, 656][params['phi']]
        freeze_count = 0

        for param in model.parameters():
            if (freeze_count <= freeze_total):
                param.requires_grad = False
            freeze_count += 1

        print("\nFroze select backbone weights.")

    # Begin training the model
    # Zero losses https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/458
    best_model_str = train(training_generator, validation_generator, model, params)

    # Export the model to onnx
    export_to_onnx(model, best_model_str, params)
    print('\n Done.\n')
 

if __name__ == "__main__":
    mp.freeze_support()
    main()
