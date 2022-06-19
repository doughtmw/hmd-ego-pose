# Modified from hmdegopose: https://github.com/ybkscht/EfficientPose

import argparse
import sys
import cv2
import torch

from generators.colibri import ColibriGenerator
from generators.utils.visualization import draw_annotations, draw_boxes, draw_mano_coords
from generators.utils.anchors import anchors_for_shape, compute_gt_annotations
import multiprocessing as mp

# CUDA for PyTorch
num_gpu = torch.cuda.device_count()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn_colibri', type=str, help='training dataset to use [syn_colibri, real_colibri]')
parser.add_argument('--fold', default='0', type=int, help='fold to use for training and evaluation [0, 1, 2, 3, 4]')
parser.add_argument('--img_size', default='256,256', type=str, help='image size [(256,256), (512,512)]')
parser.add_argument('--trans6dof', default=False, type=bool, help='6dof transform training set [False, True]')
parser.add_argument('--transcolor', default=False, type=bool, help='color transform training set [False, True]')
parser.add_argument('--batch_size', default=4, type=int, help='train batch size')
parser.add_argument('--phi', default=0, type=int, help='efficient det backbone')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for adam')
parser.add_argument('--epo', default=100, type=int, help='epochs to train')
parser.add_argument('--n_workers', default=mp.cpu_count(), type=int, help='number of cpus to use')
args = parser.parse_args()

# Default parameters for network training
params = {'dataset': args.dataset,
          'fold': args.fold,
          'img_size': tuple(map(int, str(args.img_size).split(','))),
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
          'weights_dir': 'weights/efficientdet-d0.pth'}
print("\n", params)

def main():
    """
    Creates dataset generator with the parsed input arguments and starts the dataset visualization
    Args:
        args: command line arguments

    """
    # create the generators
    print("\nCreating the Generators...")
    generator = create_generators(args, params)
    print("Done!")

    run(generator, args)


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
        # Create the synthetic colibri dataset generator
        print("\nCreating data generators for the syn_colibri dataset.")
        generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/syn_colibri_v1",
            object_id=1,
            train=True,
            shuffle_dataset=True,
            shuffle_groups=True,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=params["use_colorspace_transform"],
            use_6DoF_augmentation=params["use_6dof_transform"],
            **common_args)
        print("Done creating data generators.")

    elif params['dataset'] == 'real_colibri':
        # Create the real colibri dataset generator
        print("\nCreating data generators for the real_colibri dataset.")
        generator = ColibriGenerator(
            params,
            fold=params["fold"],
            dataset_base_path="../datasets/real_colibri_v1",
            object_id=1,
            train=True,
            shuffle_dataset=True,
            shuffle_groups=True,
            rotation_representation="axis_angle",
            use_colorspace_augmentation=params["use_colorspace_transform"],
            use_6DoF_augmentation=params["use_6dof_transform"],
            **common_args)
        print("Done creating data generators.")

    else:
        print('No dataset selected.')

    return generator


def run(generator, args):
    """ Main loop in which data is provided by the generator and then displayed
    Args:
        generator: The generator to debug.
        args: parseargs args object.
    """
    while True:
        # display images, one at a time
        for i in range(generator.size()):

            # load the image, annotations, mask and camera matrix from generator
            image = generator.load_image(i)
            annotations = generator.load_annotations(i)
            mask = generator.load_mask(i)
            camera_matrix = generator.load_camera_matrix(i)

            if len(annotations['labels']) > 0:
                # apply random transformations if they exist
                image, annotations = generator.random_transform_group_entry(image, annotations, mask, camera_matrix)
                anchors = anchors_for_shape(image.shape, anchor_params=None)
                positive_indices, _, max_indices = compute_gt_annotations(anchors[0], annotations['bboxes'])

                # switch image RGB to BGR again
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imshow('Image', image)
                if cv2.waitKey() == ord('q'):
                    cv2.destroyAllWindows()
                    return

                # draw anchors on the image
                # draw_boxes(image, anchors[0][positive_indices], (255, 255, 0), thickness=1)

                # draw annotations on the image
                draw_annotations(image,
                                 annotations,
                                 class_to_bbox_3D=generator.get_bbox_3d_dict(),
                                 camera_matrix=camera_matrix,
                                 label_to_name=generator.label_to_name,
                                 draw_bbox_2d=True,
                                 draw_name=True,
                                 draw_mano=True)

                print("Generator idx: {}".format(i))

            cv2.imshow('Image', image)
            if cv2.waitKey() == ord('q'):
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    main()