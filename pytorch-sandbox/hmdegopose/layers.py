"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.

"""

import torch
from torch import nn
import torchvision
import numpy as np


class RegressTranslation(nn.Module):
    """ 
    PyTorch layer for applying regression offset values to translation anchors to get the 2D translation centerpoint and Tz.
    """

    def __init__(self, *args, **kwargs):
        """Initializer for the RegressTranslation layer.
        """
        super(RegressTranslation, self).__init__(*args, **kwargs)

    def forward(self, translation_anchors, regression_offsets, **kwargs):
        return translation_transform_inv(translation_anchors, regression_offsets)

    def compute_output_shape(self, input_shape):
        # return input_shape[0]
        return input_shape[1]

    def get_config(self):
        config = super(RegressTranslation, self).get_config()

        return config


class RegressBoxes(nn.Module):
    """ 
    PyTorch layer for applying regression offset values to anchor boxes to get the 2D bounding boxes.
    """

    def __init__(self, *args, **kwargs):
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def forward(self, anchors, regression, **kwargs):
        return bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        return config


class CalculateTxTy(nn.Module):
    """ PyTorch layer for calculating the Tx- and Ty-Components of the Translationvector with a given 2D-point and the intrinsic camera parameters.
    """

    def __init__(self, *args, **kwargs):
        """ Initializer for an CalculateTxTy layer.
        """
        super(CalculateTxTy, self).__init__(*args, **kwargs)

    def forward(self, inputs, fx=572.4114, fy=573.57043, px=325.2611, py=242.04899, tz_scale=1000.0,
                image_scale=1.6666666666666667, **kwargs):
        # Tx = (cx - px) * Tz / fx
        # Ty = (cy - py) * Tz / fy

        # fx = tf.expand_dims(fx, axis = -1)
        # fy = tf.expand_dims(fy, axis = -1)
        # px = tf.expand_dims(px, axis = -1)
        # py = tf.expand_dims(py, axis = -1)
        # tz_scale = tf.expand_dims(tz_scale, axis = -1)
        # image_scale = tf.expand_dims(image_scale, axis = -1)

        fx = torch.unsqueeze(fx, axis=-1)
        fy = torch.unsqueeze(fy, axis=-1)
        px = torch.unsqueeze(px, axis=-1)
        py = torch.unsqueeze(py, axis=-1)
        tz_scale = torch.unsqueeze(tz_scale, axis=-1)
        image_scale = torch.unsqueeze(image_scale, axis=-1)

        x = inputs[:, :, 0] / image_scale
        y = inputs[:, :, 1] / image_scale
        tz = inputs[:, :, 2] * tz_scale

        x = x - px
        y = y - py

        # tx = tf.math.multiply(x, tz) / fx
        # ty = tf.math.multiply(y, tz) / fy

        tx = torch.mul(x, tz) / fx
        ty = torch.mul(y, tz) / fy

        output = torch.stack([tx, ty, tz], axis=-1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CalculateTxTy, self).get_config()

        return config


class ClipBoxes(nn.Module):
    """
    Layer that clips 2D bounding boxes so that they are inside the image
    """

    def forward(self, image, boxes, **kwargs):
        # shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        height = image.shape[2]
        width = image.shape[3]
        # x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        # y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        # x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        # y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        x1 = torch.clamp(boxes[:, :, 0], 0, width - 1)
        y1 = torch.clamp(boxes[:, :, 1], 0, height - 1)
        x2 = torch.clamp(boxes[:, :, 2], 0, width - 1)
        y2 = torch.clamp(boxes[:, :, 3], 0, height - 1)

        return torch.stack((x1, y1, x2, y2), dim=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def translation_transform_inv(translation_anchors, deltas, scale_factors=None):
    """ Applies the predicted 2D translation center point offsets (deltas) to the translation_anchors

    Args
        translation_anchors : Tensor of shape (B, N, 3), where B is the batch size, N the number of boxes and 2 values for (x, y) +1 value with the stride.
        deltas: Tensor of shape (B, N, 3). The first 2 deltas (d_x, d_y) are a factor of the stride +1 with Tz.

    Returns
        A tensor of the same shape as translation_anchors, but with deltas applied to each translation_anchors and the last coordinate is the concatenated (untouched) Tz value from deltas.
    """

    stride = translation_anchors[:, :, -1]

    if scale_factors:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * scale_factors[0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * scale_factors[1] * stride)
    else:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * stride)

    Tz = deltas[:, :, 2]

    pred_translations = torch.stack([x, y, Tz], axis=2)  # x,y 2D Image coordinates and Tz

    return pred_translations


def bbox_transform_inv(boxes, deltas, scale_factors=None):
    """
    Reconstructs the 2D bounding boxes using the anchor boxes and the predicted deltas of the anchor boxes to the bounding boxes
    Args:
        boxes: Tensor containing the anchor boxes with shape (..., 4)
        deltas: Tensor containing the offsets of the anchor boxes to the bounding boxes with shape (..., 4)
        scale_factors: optional scaling factor for the deltas
    Returns:
        Tensor containing the reconstructed 2D bounding boxes with shape (..., 4)

    """
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.

    bbox = torch.stack((xmin, ymin, xmax, ymax), dim=-1)
    return bbox


class CalculateTxTy(nn.Module):
    """ Keras layer for calculating the Tx- and Ty-Components of the Translationvector with a given 2D-point and the intrinsic camera parameters.
    """

    def __init__(self, *args, **kwargs):
        """ Initializer for an CalculateTxTy layer.
        """
        super(CalculateTxTy, self).__init__(*args, **kwargs)

    def forward(self,
                inputs,
                fx=572.4114, fy=573.57043,
                px=325.2611, py=242.04899,
                tz_scale=1000.0, image_scale=1.6666666666666667, **kwargs):
        # Tx = (cx - px) * Tz / fx
        # Ty = (cy - py) * Tz / fy

        # fx = tf.expand_dims(fx, axis=-1)
        # fy = tf.expand_dims(fy, axis=-1)
        # px = tf.expand_dims(px, axis=-1)
        # py = tf.expand_dims(py, axis=-1)

        fx = torch.unsqueeze(fx, dim=-1)
        fy = torch.unsqueeze(fy, dim=-1)
        px = torch.unsqueeze(px, dim=-1)
        py = torch.unsqueeze(py, dim=-1)

        # tz_scale = tf.expand_dims(tz_scale, axis=-1)
        # image_scale = tf.expand_dims(image_scale, axis=-1)

        tz_scale = torch.unsqueeze(tz_scale, dim=-1)
        image_scale = torch.unsqueeze(image_scale, dim=-1)

        x = inputs[:, :, 0] / image_scale
        y = inputs[:, :, 1] / image_scale
        tz = inputs[:, :, 2] * tz_scale

        x = x - px
        y = y - py

        tx = torch.mul(x, tz) / fx
        ty = torch.mul(y, tz) / fy

        # output = tf.stack([tx, ty, tz], axis=-1)
        output = torch.stack((tx, ty, tz), dim=-1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CalculateTxTy, self).get_config()

        return config


import tensorflow as tf
from tensorflow import keras


def filter_detections(
        boxes,
        classification,
        rotation,
        translation,
        hand,
        num_rotation_parameters,
        num_translation_parameters=3,
        num_hand_parameters=63,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.01,
        max_detections=100,
        nms_threshold=0.5,
):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        rotation: Tensor of shape (num_boxes, num_rotation_parameters) containing the rotations.
        translation: Tensor of shape (num_boxes, 3) containing the translation vectors.
        num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
        num_translation_parameters: Number of translation parameters, usually 3
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, rotation, translation].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        rotation is shaped (max_detections, num_rotation_parameters) and contains the rotations of the non-suppressed predictions.
        translation is shaped (max_detections, num_translation_parameters) and contains the translations of the non-suppressed predictions.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

            # perform NMS
            # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
            #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]], axis=-1)
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)
    rotation = keras.backend.gather(rotation, indices)
    translation = keras.backend.gather(translation, indices)
    hand = keras.backend.gather(hand, indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')
    rotation = tf.pad(rotation, [[0, pad_size], [0, 0]], constant_values=-1)
    translation = tf.pad(translation, [[0, pad_size], [0, 0]], constant_values=-1)
    hand = tf.pad(hand, [[0, pad_size], [0, 0]], constant_values=-1)

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    rotation.set_shape([max_detections, num_rotation_parameters])
    translation.set_shape([max_detections, num_translation_parameters])
    hand.set_shape([max_detections, num_hand_parameters])

    return [
        torch.from_numpy(boxes.numpy()),
        torch.from_numpy(scores.numpy()),
        torch.from_numpy(labels.numpy()),
        torch.from_numpy(rotation.numpy()),
        torch.from_numpy(translation.numpy()),
        torch.from_numpy(hand.numpy())]


class FilterDetections(nn.Module):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            num_rotation_parameters,
            num_translation_parameters=3,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.01,
            max_detections=100,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
            num_translation_parameters: Number of translation parameters, usually 3
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
        super(FilterDetections, self).__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, rotation, translation] tensors.
        """
        boxes = tf.convert_to_tensor(inputs[0].detach().cpu())
        classification = tf.convert_to_tensor(inputs[1].detach().cpu())
        rotation = tf.convert_to_tensor(inputs[2].detach().cpu())
        translation = tf.convert_to_tensor(inputs[3].detach().cpu())
        hand = tf.convert_to_tensor(inputs[4].detach().cpu())

        # iterate across batch
        batch_size = classification.shape[0]

        # Actual
        # bboxes.shape: torch.Size([1, 49104, 4])
        # boxes.shape: torch.Size([1, 49104, 4])
        # classification.shape: torch.Size([1, 49104, 1])
        # rotation.shape: torch.Size([1, 49104, 3])
        # translation.shape: torch.Size([1, 49104, 3])
        # hand.shape: torch.Size([1, 49104, 63])

        # Ensure batch size of 1
        for i in range(batch_size):
            output = filter_detections(
                boxes[i],
                classification[i],
                rotation[i],
                translation[i],
                hand[i],
                self.num_rotation_parameters,
                self.num_translation_parameters,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold)

        # outputs.append()
        return output



    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, rotation, translation].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_rotation.shape, filtered_translation.shape]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[2][0], self.max_detections, self.num_rotation_parameters),
            (input_shape[3][0], self.max_detections, self.num_translation_parameters),
        ]

    def compute_mask(self, inputs, mask=None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
            'num_rotation_parameters': self.num_rotation_parameters,
            'num_translation_parameters': self.num_translation_parameters,
        })

        return config
