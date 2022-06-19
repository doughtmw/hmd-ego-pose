# Adated from: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch and https://github.com/ybkscht/EfficientPose
import math

import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors
from efficientdet.utils import BBoxTransform, ClipBoxes
from hmdegopose.model import TranslationNet, RotationNet, HandNet

# Create the HMDEgoPose
class HMDEgoPose(nn.Module):
    def __init__(self, params, num_classes=1, compound_coef=0, load_weights=False, onnx_export=False,
                 input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536], **kwargs):
        super(HMDEgoPose, self).__init__()
        self.compound_coef = compound_coef
        self.onnx_export = onnx_export
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        # self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_sizes = input_sizes
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    onnx_export=self.onnx_export,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef],
                                   onnx_export=self.onnx_export)
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef],
                                     onnx_export=self.onnx_export)

        # self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
        #                        pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
        #                        **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights,
                                         onnx_export=self.onnx_export)

        # Added rotation and translation networks
        self.rotation_net = RotationNet(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                        num_classes=num_classes,
                                        num_iteration_steps=params["iter"],
                                        num_layers=self.box_class_repeats[self.compound_coef],
                                        num_rotation_parameters=3,
                                        pyramid_levels=self.pyramid_levels[self.compound_coef],
                                        onnx_export=self.onnx_export)
        self.translation_net = TranslationNet(in_channels=self.fpn_num_filters[self.compound_coef],
                                              num_anchors=num_anchors,
                                              num_classes=num_classes,
                                              num_layers=self.box_class_repeats[self.compound_coef],
                                              num_iteration_steps=params["iter"],
                                              pyramid_levels=self.pyramid_levels[self.compound_coef],
                                              onnx_export=self.onnx_export)

        # Added hand network for regression of skeleton params
        self.hand_net = HandNet(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                        num_classes=num_classes,
                                        num_iteration_steps=params["iter"],
                                        num_layers=self.box_class_repeats[self.compound_coef],
                                        num_hand_parameters=63,
                                        pyramid_levels=self.pyramid_levels[self.compound_coef],
                                        onnx_export=self.onnx_export)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        # group the features for feeding to bifpn
        features = (p3, p4, p5)
        features = self.bifpn(features)

        # Get regression (boxes) and classifier (class scores) from features
        regression = self.regressor(features)
        classification = self.classifier(features)
        # anchors = self.anchors(inputs, inputs.dtype)

        # Added for rotation and translation subnets
        rotation = self.rotation_net(features)
        translation = self.translation_net(features)

        # Added for hand pose subnet
        hand = self.hand_net(features)

        return features, regression, classification, rotation, translation, hand

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=True)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')