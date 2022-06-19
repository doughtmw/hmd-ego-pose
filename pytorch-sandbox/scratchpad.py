# https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
import math

import torch
import torchvision

import cv2
import numpy as np
import PIL
from PIL import Image
import onnxruntime as rt

images = ['000000']

# Want to recreate image transforms from PyTorch in OpenCV
def preprocess_image(image, image_size):
    # image, RGB
    image_height, image_width = image.shape[:2]
    print("image_height: ", image_height, " image_width: ", image_width)
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    print("resized_width: ", resized_width, " resized_height: ", resized_height)
    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale

def main():
    for image in images:
        print(image)

        # open the image as an OpenCV image
        openCvImage = cv2.imread('onnx-models/' + image + '.png')

        # Convert the colour channel orientation from BGR to RGB
        openCvImage = cv2.cvtColor(openCvImage, cv2.COLOR_BGR2RGB)

        # preprocess the image as in generators.common.py
        openCvImage, scale = preprocess_image(openCvImage, 256)
        cv2.imshow('opencv-transforms', openCvImage)
        cv2.imwrite('onnx-models/' + image + '-opencv-transforms.jpg', openCvImage)

        # Reshape NWHC -> NCWH
        # [256, 256, 3] -> [3, 256, 256] -> [1, 3, 256, 256]
        openCvImage = openCvImage.transpose([2, 0, 1])
        openCvImage = openCvImage.reshape((1, 3, 256, 256)).astype(np.float32)

        # show results
        print('\nopenCvImage.shape = ' + str(openCvImage.shape))
        print('openCvImage max = ' + str(np.max(openCvImage)))
        print('openCvImage min = ' + str(np.min(openCvImage)))
        print('openCvImage avg = ' + str(np.mean(openCvImage)))
        print('openCvImage: ')

        # Test ONNX inference
        sess = rt.InferenceSession("onnx-models/model.onnx")
        input_name = sess.get_inputs()[0].name
        f1, f2, f3, f4, f5, \
        regression, classification, rotation, translation_raw, hand\
            = sess.run(None, {input_name: openCvImage})

        # Compare against the raw results from OpenCVDNNSandboxNetCore project
        # regression: [ [[  4.3404813   6.3829317   0.5551747 -15.24141  ]] ]
        # classification: [ [[0.0143396]] ]
        # rotation: [ [[-0.05352388  0.51271254 -0.23526134]] ]
        # translation_raw: [ [[1.6507937 0.5715018 0.4628573]] ]

        # OpenCVDNNSandboxNetCore
        # regression: [4.3404803, 6.3829308, 0.5551741, -15.241405]
        # classification: [0.014339566]
        # rotation: [-0.05352387, 0.5127123, -0.23526148]
        # translation_raw: [1.650794, 0.57150227, 0.46285737]
        print('\nregression: [', regression[:, 0, :], ']')
        print('classification: [', classification[:, 0, :], ']')
        print('rotation: [', rotation[:, 0, :], ']')
        print('translation_raw: [', translation_raw[:, 0, :], ']')

        # Not performing ALL processing here... only the raw network prediction results

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nDone...\n')


if __name__ == '__main__':
    main()