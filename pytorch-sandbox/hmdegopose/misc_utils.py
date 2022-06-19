import numpy as np
import torch
import torchvision
import os
import onnx
import time
from collections import OrderedDict
import onnxruntime as rt


# Utility to print the size of the model
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


# Utility to count total (trainable) parameters in network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# Convert tensor to numpy for onnx runtime implementation
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# https://github.com/sovrasov/flops-counter.pytorch/issues/14
# Estimate FLOPS of model
def prepare_input(resolution):
    x = torch.FloatTensor(1, *resolution)
    return dict(x=x)


def export_to_onnx(model, best_model_str, params):
    CPU = torch.device('cpu')
    model.to(CPU)

    # Load best model weights
    state_dict = torch.load(best_model_str)
    new_state_dict = OrderedDict()
    
    # For weights trained on real dataset
    for k, v in state_dict.items():
        name = k[6:]  # remove `model.`
        new_state_dict[name] = v
    
    # # For weights trained on syn dataset
    # for k, v in state_dict.items():
    #     name = k[13:]  # remove `model.`
    #     new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    print("Loaded weights!")

    # Set model to eval mode
    model.eval()

    # Create a random tensor of shape for testing
    random_input = np.random.rand(1, 3, 256, 256)
    img_resized = np.ascontiguousarray(random_input).astype(np.float32)

    # save this random input for re-using by TensorFlow
    np.save("onnx-models/input.npy", img_resized)

    # Get prediction on tensor
    with torch.no_grad():
        sample = torch.from_numpy(img_resized)
        prediction = model.forward(sample)

    # print(prediction.shape)
    print(prediction)

    # Export the onnx model
    torch.onnx.export(model, sample, 'onnx-models/model.onnx',
                      opset_version=9,
                      input_names=['input'],
                      output_names=[
                          'feat1', 'feat2', 'feat3', 'feat4', 'feat5',
                          'regression', 'classification',
                          'rotation', 'translation_raw', 'hand'])

    # Confirm onnx predictions are correct after export
    img_resized = np.load("onnx-models/input.npy")
    img_resized = np.ascontiguousarray(img_resized).astype(np.float32)

    # Load the onnx model and bind to new session with input
    sess = rt.InferenceSession("onnx-models/model.onnx")
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: img_resized})[0]

    # print(pred.shape)
    print(pred)
