import torch
from torch import nn
from torch.nn import functional as F

from efficientnet.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

from efficientdet.model import SeparableConvBlock

from torch import nn

# Create the subnetwork for rotation prediction
class RotationNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, num_rotation_parameters, num_iteration_steps, pyramid_levels=5, onnx_export=False):
        super(RotationNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Added
        self.num_rotation_parameters = num_rotation_parameters
        self.num_iteration_steps = num_iteration_steps

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Added
        self.initial_rotation = SeparableConvBlock(in_channels, num_anchors * num_rotation_parameters, norm=False, activation=False)

        if self.num_iteration_steps >= 1:
            self.iterative_submodel = IterativeRotationSubNet(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
                num_layers=num_layers,
                num_rotation_parameters=num_rotation_parameters,
                num_iteration_steps=num_iteration_steps,
                onnx_export=onnx_export)

            print("Created IterativeRotationSubNet with " + str(self.num_iteration_steps) + " iteration(s).")

    def forward(self, inputs):
        feats = []

        # inputs: 5 tuple, same as self.bn_list
        # for iterative submodule, only get a single feature level
        # 0 = {Tensor: (16, 64, 32, 32)}
        # 1 = {Tensor: (16, 64, 16, 16)}
        # 2 = {Tensor: (16, 64, 8, 8)}
        # 3 = {Tensor: (16, 64, 4, 4)}
        # 4 = {Tensor: (16, 64, 2, 2)}
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)

            rotation = self.initial_rotation(feat)

            # Iterative submodule
            if self.num_iteration_steps >= 1:
                for i in range(self.num_iteration_steps):
                    iterative_input = torch.cat((feat, rotation), dim=1)
                    delta_rotation = self.iterative_submodel(iterative_input, iter_step_py=i)
                    rotation = torch.add(rotation, delta_rotation)

            # Added
            rotation = rotation.permute(0, 2, 3, 1)
            rotation = rotation.contiguous().view(rotation.shape[0], -1, self.num_rotation_parameters)
            # print("rotation 2:", rotation.size())

            feats.append(rotation)  

        feats = torch.cat(feats, dim=1)
        # print("RotationNet: feats:", feats.size())

        return feats

# Create the subnetwork for hand skeleton prediction
class HandNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, num_hand_parameters, num_iteration_steps,
                 pyramid_levels=5, onnx_export=False):
        super(HandNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Added
        self.num_hand_parameters = num_hand_parameters
        self.num_iteration_steps = num_iteration_steps

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Added
        self.initial_hand_coords = SeparableConvBlock(in_channels, num_anchors * num_hand_parameters, norm=False,
                                                      activation=False)

        if self.num_iteration_steps >= 1:
            self.iterative_submodel = IterativeHandSubnet(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
                num_layers=num_layers,
                num_hand_parameters=num_hand_parameters,
                num_iteration_steps=num_iteration_steps,
                onnx_export=onnx_export)
            print("Created IterativeHandSubnet with " + str(self.num_iteration_steps) + " iteration(s).")

    def forward(self, inputs):
        feats = []

        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)

            # Added
            hand_coords = self.initial_hand_coords(feat)

            # Iterative submodule
            if self.num_iteration_steps >= 1:
                for i in range(self.num_iteration_steps):
                    iterative_input = torch.cat((feat, hand_coords), dim=1)
                    delta_hand_coords = self.iterative_submodel(iterative_input, iter_step_py=i)
                    hand_coords = torch.add(hand_coords, delta_hand_coords)

            # Added
            hand_coords = hand_coords.permute(0, 2, 3, 1)
            hand_coords = hand_coords.contiguous().view(hand_coords.shape[0], -1, self.num_hand_parameters)
            # print("rotation 2:", rotation.size())

            feats.append(hand_coords)

        feats = torch.cat(feats, dim=1)
        # print("RotationNet: feats:", feats.size())

        return feats

# Create the subnetwork for translation prediction
class TranslationNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, num_iteration_steps, pyramid_levels=5, onnx_export=False):
        super(TranslationNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Added
        self.num_iteration_steps = num_iteration_steps

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Added
        self.initial_translation_xy = SeparableConvBlock(in_channels, num_anchors * 2, norm=False, activation=False)
        self.initial_translation_z = SeparableConvBlock(in_channels, num_anchors, norm=False, activation=False)

        if self.num_iteration_steps >= 1:
            self.iterative_submodel = IterativeTranslationSubNet(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
                num_layers=num_layers,
                num_iteration_steps=num_iteration_steps,
                onnx_export=onnx_export)

        print("Created IterativeTranslationSubNet with " + str(self.num_iteration_steps) + " iteration(s).")

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)

            # Added
            translation_xy = self.initial_translation_xy(feat)
            translation_z = self.initial_translation_z(feat)
            # print("translation_xy 1:", translation_xy.size())
            # print("translation_z 1:", translation_z.size())

            # Iterative submodule
            if self.num_iteration_steps >= 1:
                for i in range(self.num_iteration_steps):
                    iterative_input = torch.cat((feat, translation_xy, translation_z), dim=1)
                    delta_translation_xy, delta_translation_z = self.iterative_submodel(iterative_input, iter_step_py=i)
                    translation_xy = torch.add(translation_xy, delta_translation_xy)
                    translation_z = torch.add(translation_z, delta_translation_z)

            # Added
            translation_xy = translation_xy.permute(0, 2, 3, 1)
            translation_z = translation_z.permute(0, 2, 3, 1)
            
            translation_xy = translation_xy.contiguous().view(translation_xy.shape[0], -1, 2)
            translation_z = translation_z.contiguous().view(translation_z.shape[0], -1, 1)

            translation_xyz = torch.cat((translation_xy, translation_z), dim=2)
            # print("translation_xyz:", translation_xyz.size())

            feats.append(translation_xyz)  

        feats = torch.cat(feats, dim=1)
        # print("TranslationNet: feats:", feats.size())

        return feats


# Adapted from the EfficientPose architecture -- currently not using iterative subnets
class IterativeRotationSubNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, num_rotation_parameters, num_iteration_steps, pyramid_levels=5, onnx_export=False):
        super(IterativeRotationSubNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Added
        self.num_rotation_parameters = num_rotation_parameters
        self.num_iteration_steps = num_iteration_steps

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(91, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Added
        self.norm_layer = nn.ModuleList(
                [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for k in range(num_iteration_steps)])
        self.head = SeparableConvBlock(in_channels, num_anchors * num_rotation_parameters, norm=False, activation=False)

    def forward(self, feat, iter_step_py, **kwargs):
        feats = []

        for i, bn, conv in zip(range(self.num_layers), self.norm_layer, self.conv_list):
            feat = conv(feat)
            feat = bn[iter_step_py](feat)
            feat = self.swish(feat)

        feat = self.head(feat)
        feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats

# Adapted from the EfficientPose architecture -- currently not using iterative subnets
class IterativeHandSubnet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, num_hand_parameters, num_iteration_steps, pyramid_levels=5, onnx_export=False):
        super(IterativeHandSubnet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Added
        self.num_hand_parameters = num_hand_parameters
        self.num_iteration_steps = num_iteration_steps

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(631, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Added
        self.norm_layer = nn.ModuleList(
                [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for k in range(num_iteration_steps)])
        self.head = SeparableConvBlock(in_channels, num_anchors * num_hand_parameters, norm=False, activation=False)

    def forward(self, feat, iter_step_py, **kwargs):
        feats = []

        for i, bn, conv in zip(range(self.num_layers), self.norm_layer, self.conv_list):
            feat = conv(feat)
            feat = bn[iter_step_py](feat)
            feat = self.swish(feat)

        feat = self.head(feat)
        feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats

# Adapted from the EfficientPose architecture -- currently not using iterative subnets
class IterativeTranslationSubNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, num_iteration_steps, pyramid_levels=5, onnx_export=False):
        super(IterativeTranslationSubNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Added
        self.num_iteration_steps = num_iteration_steps

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(91, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Added
        self.norm_layer = nn.ModuleList(
                [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for k in range(num_iteration_steps)])
        self.head_xy = SeparableConvBlock(in_channels, num_anchors * 2, norm=False, activation=False)
        self.head_z = SeparableConvBlock(in_channels, num_anchors, norm=False, activation=False)

    def forward(self, feat, iter_step_py, **kwargs):

        # Added
        outputs_xy = []
        outputs_z = []

        for i, bn, conv in zip(range(self.num_layers), self.norm_layer, self.conv_list):
            feat = conv(feat)
            feat = bn[iter_step_py](feat)
            feat = self.swish(feat)

        output_xy = self.head_xy(feat)
        output_z = self.head_z(feat)

        outputs_xy.append(output_xy)
        outputs_z.append(output_z)

        # Added
        outputs_xy = torch.cat(outputs_xy, dim=1)
        outputs_z = torch.cat(outputs_z, dim=1)

        # return feats
        return outputs_xy, outputs_z

