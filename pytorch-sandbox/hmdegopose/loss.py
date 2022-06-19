# Modified from: https://github.com/ybkscht/EfficientPose and https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

import math
import torch
import numpy as np

from hmdegopose.layers import RegressTranslation, CalculateTxTy, RegressBoxes, ClipBoxes
from generators.utils.anchors import anchors_for_shape

# Handle regression and clipping of bounding boxes
# based on the image input shape and anchors
def format_bboxes(image_input, anchors, bbox_regression):
    anchors = torch.tensor(anchors)

    if torch.cuda.is_available():
        anchors = anchors.cuda()

    anchors_input = torch.unsqueeze(anchors, dim=0)

    bboxes = RegressBoxes()(anchors_input, bbox_regression[..., :4])
    bboxes = ClipBoxes()(image_input, bboxes)

    return bboxes

def create_anchors(input_size):
    # get anchors and apply predicted translation offsets to translation anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size))  # use default anchors
    return anchors, translation_anchors

def format_translation(translation_anchors, translation_raw, camera_parameters_input):

    # get anchors and apply predicted translation offsets to translation anchors
    # anchors, translation_anchors = anchors_for_shape((input_size, input_size))  # use default anchors
    translation_anchors_input = torch.Tensor(np.expand_dims(translation_anchors, axis=0))

    if torch.cuda.is_available():
        translation_anchors_input = translation_anchors_input.cuda()

    regress_translation = RegressTranslation()
    calculate_txty = CalculateTxTy()

    translation_xy_Tz = regress_translation(translation_anchors_input, translation_raw)
    translation = calculate_txty(translation_xy_Tz,
                                 fx=camera_parameters_input[:, 0],
                                 fy=camera_parameters_input[:, 1],
                                 px=camera_parameters_input[:, 2],
                                 py=camera_parameters_input[:, 3],
                                 tz_scale=camera_parameters_input[:, 4],
                                 image_scale=camera_parameters_input[:, 5])

    return translation

# Compute losses each batch
def batch_iterate(gt_classification, classification,
                  gt_regression, regression,
                  gt_transformation, transformation,
                  gt_hand, hand,
                  model_3d_points, num_rotation_parameter):
    batch_size = classification.shape[0]

    classification_losses = []
    regression_losses = []
    rotation_losses = []
    translation_losses = []
    hand_losses = []

    # Compute losses across batch
    for j in range(batch_size):
        # Classification loss
        classification_losses.append(focal(
            gt_classification=gt_classification[j, :, :],
            classification=classification[j, :, :]))

        # Regression loss
        regression_losses.append(smooth_l1(
            gt_regression=gt_regression[j, :, :],
            regression=regression[j, :, :]))

        # Transformation loss, split into rotation and translation components
        # Using smooth l1 loss for translation, rotation component is based on vertices distance
        rotation_loss, translation_loss = transformation_loss(
            gt_transformation=gt_transformation[j, :, :], transformation=transformation[j, :, :],
            model_3d_points_np=model_3d_points, num_rotation_parameter=num_rotation_parameter)

        rotation_losses.append(rotation_loss)
        translation_losses.append(translation_loss)

        # Append the hand loss
        hand_losses.append(smooth_l1_hands(
            gt_hands=gt_hand[j, :, :],
            hands=hand[j, :, :]))

    # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233
    return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
           torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50, \
           torch.stack(rotation_losses).mean(dim=0, keepdim=True), \
           torch.stack(translation_losses).mean(dim=0, keepdim=True), \
           torch.stack(hand_losses).mean(dim=0, keepdim=True)


def focal(gt_classification, classification, alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

    Args:
        gt_classification: Tensor of target data from the generator with shape (B, N, num_classes).
        classification: Tensor of predicted data from the network with shape (B, N, num_classes).
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns:
        A functor that computes the focal loss using the alpha and gamma.
    """

    # -1 for ignore, 0 for background, 1 for object
    labels = gt_classification[:, :-1]
    anchor_state = gt_classification[:, -1]

    # filter out "ignore" anchors
    # indices = tf.where(keras.backend.not_equal(anchor_state, -1))
    indices = (torch.stack(torch.where(torch.ne(anchor_state, -1)), dim=1)).squeeze(0)

    # labels = tf.gather_nd(labels, indices)
    labels = labels[list(indices.T)]
    # classification = tf.gather_nd(classification, indices)
    classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
    classification = classification[list(indices.T)]

    # compute the focal loss
    # alpha_factor = keras.backend.ones_like(labels) * alpha
    alpha_factor = torch.ones_like(labels) * alpha

    if torch.cuda.is_available():
        alpha_factor = alpha_factor.cuda()

    # alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    alpha_factor = torch.where(labels.eq(1), alpha_factor, 1 - alpha_factor)

    # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
    # focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = torch.where(torch.eq(labels, 1), 1 - classification, classification)

    # focal_weight = alpha_factor * focal_weight ** gamma
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

    # cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)
    # cls_loss = focal_weight * torch.nn.functional.binary_cross_entropy(classification, labels)
    bce = -(labels * torch.log(classification) + (1.0 - labels) * torch.log(1.0 - classification))
    cls_loss = focal_weight * bce

    zeros = torch.zeros_like(cls_loss)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
    cls_loss = torch.where(torch.ne(labels, -1.0), cls_loss, zeros)

    # compute the normalizer: the number of positive anchors
    # normalizer = tf.where(keras.backend.equal(anchor_state, 1))
    normalizer = torch.stack(torch.where(torch.eq(anchor_state, 1)), dim=1).squeeze(0)
    # normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
    normalizer = torch.tensor(normalizer.shape[0]).type(torch.FloatTensor)
    # normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)
    normalizer = torch.max(torch.tensor(1.0).type(torch.FloatTensor), normalizer)

    # return keras.backend.sum(cls_loss) / normalizer
    return cls_loss.sum() / normalizer


def smooth_l1(gt_regression, regression, sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args:
        gt_regression: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
        regression: Tensor from the network of shape (B, N, 4).
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns:
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    # separate target and state
    regression_target = gt_regression[:, :-1]
    anchor_state = gt_regression[:, -1]
    # regression = regression[:, -1]

    # filter out "ignore" anchors
    # indices = tf.where(keras.backend.equal(anchor_state, 1))
    indices = (torch.stack(torch.where(torch.eq(anchor_state, 1)), dim=1)).squeeze(0)
    # regression = tf.gather_nd(regression, indices)
    regression = regression[list(indices.T)]

    # regression_target = tf.gather_nd(regression_target, indices)
    regression_target = regression_target[list(indices.T)]

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    # regression_diff = keras.backend.abs(regression_diff)
    regression_diff = torch.abs(regression_diff)
    # regression_loss = tf.where(
    #     keras.backend.less(regression_diff, 1.0 / sigma_squared),
    #     0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
    #     regression_diff - 0.5 / sigma_squared
    # )
    regression_loss = torch.where(
        torch.le(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    # compute the normalizer: the number of positive anchors
    # normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
    normalizer = torch.max(
        torch.tensor(1.0).type(torch.FloatTensor),
        torch.tensor(indices.shape[0]).type(torch.FloatTensor)).type(torch.FloatTensor)
    # normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

    # return keras.backend.sum(regression_loss) / normalizer
    return regression_loss.sum() / normalizer


def smooth_l1_hands(gt_hands, hands, sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args:
        gt_hands: Tensor from the generator of shape (B, N, 64). The last value for each box is the state of the anchor (ignore, negative, positive).
        hands: Tensor from the network of shape (B, N, 63).
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns:
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    # separate target and state
    hands_target = gt_hands[:, :-1]
    anchor_state = gt_hands[:, -1]
    # regression = regression[:, -1]

    # filter out "ignore" anchors
    # indices = tf.where(keras.backend.equal(anchor_state, 1))
    indices = (torch.stack(torch.where(torch.eq(anchor_state, 1)), dim=1)).squeeze(0)
    # regression = tf.gather_nd(regression, indices)
    hands = hands[list(indices.T)]

    # regression_target = tf.gather_nd(regression_target, indices)
    hands_target = hands_target[list(indices.T)]

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    hands_diff = hands - hands_target
    # regression_diff = keras.backend.abs(regression_diff)
    hands_diff = torch.abs(hands_diff)
    # regression_loss = tf.where(
    #     keras.backend.less(regression_diff, 1.0 / sigma_squared),
    #     0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
    #     regression_diff - 0.5 / sigma_squared
    # )
    hand_loss = torch.where(
        torch.le(hands_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * torch.pow(hands_diff, 2),
        hands_diff - 0.5 / sigma_squared
    )

    # compute the normalizer: the number of positive anchors
    # normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
    normalizer = torch.max(
        torch.tensor(1.0).type(torch.FloatTensor),
        torch.tensor(indices.shape[0]).type(torch.FloatTensor)).type(torch.FloatTensor)
    # normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

    # return keras.backend.sum(regression_loss) / normalizer
    return hand_loss.sum() / normalizer

def transformation_loss(gt_transformation, transformation, model_3d_points_np, num_rotation_parameter):
    """
    Create a transformation loss functor as described in https://arxiv.org/abs/2011.04307
    Args:
        gt_transformation: Tensor from the generator of shape (B, N, num_rotation_parameter + num_translation_parameter + is_symmetric_flag + class_label + anchor_state).
                num_rotation_parameter is 3 for axis angle representation and num_translation parameter is also 3
                is_symmetric_flag is a Boolean indicating if the GT object is symmetric or not, used to calculate the correct loss
                class_label is the class of the GT object, used to take the correct 3D model points from the model_3d_points tensor for the transformation
                The last value for each box is the state of the anchor (ignore, negative, positive).
        transformation: Tensor from the network of shape (B, N, num_rotation_parameter + num_translation_parameter).
        model_3d_points_np: numpy array containing the 3D model points of all classes for calculating the transformed point distances.
                            The shape is (num_classes, num_points, 3)
        num_rotation_parameter: The number of rotation parameters, usually 3 for axis angle representation
    Returns:
        A functor for computing the transformation loss given target data and predicted data.
    """
    model_3d_points = torch.tensor(model_3d_points_np).type(torch.FloatTensor)
    # model_3d_points = tf.convert_to_tensor(value=model_3d_points_np)
    # num_points = tf.shape(model_3d_points)[1]
    num_points = model_3d_points.shape[1]

    # separate target and state
    regression_rotation = transformation[:, :num_rotation_parameter]
    regression_translation = transformation[:, num_rotation_parameter:]
    regression_target_rotation = gt_transformation[:, :num_rotation_parameter]
    regression_target_translation = gt_transformation[:, num_rotation_parameter:-3]
    is_symmetric = gt_transformation[:, -3]
    class_indices = gt_transformation[:, -2]
    # anchor_state = tf.cast(tf.math.round(y_true[:, :, -1]), tf.int32)
    anchor_state = torch.round(gt_transformation[:, -1]).type(torch.IntTensor)

    # filter out "ignore" anchors
    indices = torch.stack(torch.where(torch.eq(anchor_state, 1)), dim=1)
    # indices = tf.where(tf.equal(anchor_state, 1))
    # regression_rotation = tf.gather_nd(regression_rotation, indices) * math.pi
    regression_rotation = regression_rotation[list(indices.T)] * math.pi
    # regression_translation = tf.gather_nd(regression_translation, indices)
    regression_translation = regression_translation[list(indices.T)]

    # regression_target_rotation = tf.gather_nd(regression_target_rotation, indices) * math.pi
    regression_target_rotation = regression_target_rotation[list(indices.T)] * math.pi
    # regression_target_translation = tf.gather_nd(regression_target_translation, indices)
    regression_target_translation = regression_target_translation[list(indices.T)]

    # is_symmetric = tf.gather_nd(is_symmetric, indices)
    is_symmetric = is_symmetric[list(indices.T)]
    # is_symmetric = tf.cast(tf.math.round(is_symmetric), tf.int32)
    is_symmetric = torch.round(is_symmetric).type(torch.IntTensor)
    # class_indices = tf.gather_nd(class_indices, indices)
    class_indices = class_indices[list(indices.T)]
    # class_indices = torch.round(class_indices).type(torch.IntTensor)
    class_indices = torch.round(class_indices).type(torch.LongTensor)
    # class_indices = tf.cast(tf.math.round(class_indices), tf.int32)

    axis_pred, angle_pred = separate_axis_from_angle(regression_rotation)
    axis_target, angle_target = separate_axis_from_angle(regression_target_rotation)

    # rotate the 3d model points with target and predicted rotations
    # select model points according to the class indices
    # selected_model_points = tf.gather(model_3d_points, class_indices, axis=0)
    class_indices = torch.reshape(class_indices, (class_indices.shape[0], 1))
    selected_model_points = model_3d_points[list(class_indices.T)]

    # expand dims of the rotation tensors to rotate all points along the dimension via broadcasting
    # axis_pred = tf.expand_dims(axis_pred, axis=1)
    axis_pred = torch.unsqueeze(axis_pred, dim=1)

    # angle_pred = tf.expand_dims(angle_pred, axis=1)
    angle_pred = torch.unsqueeze(angle_pred, dim=1)

    # axis_target = tf.expand_dims(axis_target, axis=1)
    axis_target = torch.unsqueeze(axis_target, dim=1)

    # angle_target = tf.expand_dims(angle_target, axis=1)
    angle_target = torch.unsqueeze(angle_target, dim=1)

    # also expand dims of the translation tensors to translate all points along the dimension via broadcasting
    # regression_translation = tf.expand_dims(regression_translation, axis=1)
    regression_translation = torch.unsqueeze(regression_translation, dim=1)

    # regression_target_translation = tf.expand_dims(regression_target_translation, axis=1)
    regression_target_translation = torch.unsqueeze(regression_target_translation, dim=1)

    # Try splitting up these components into rotation and translation losses as in:
    # PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes
    # transformed_points_pred = rotate(selected_model_points, axis_pred, angle_pred) + regression_translation
    # transformed_points_target = rotate(selected_model_points, axis_target, angle_target) + regression_target_translation
    rotated_points_pred = rotate(selected_model_points, axis_pred, angle_pred)
    rotated_points_target = rotate(selected_model_points, axis_target, angle_target)

    # distinct between symmetric and asymmetric objects
    sym_indices = torch.stack(torch.where(torch.eq(is_symmetric, 1)), dim=1).squeeze(0)
    # sym_indices = tf.where(keras.backend.equal(is_symmetric, 1))
    # asym_indices = tf.where(keras.backend.not_equal(is_symmetric, 1))
    asym_indices = torch.stack(torch.where(torch.ne(is_symmetric, 1)), dim=1).squeeze(0)

    # sym_points_pred = tf.reshape(tf.gather_nd(transformed_points_pred, sym_indices), (-1, num_points, 3))
    # sym_points_pred = torch.reshape(transformed_points_pred[list(sym_indices.T)], (-1, num_points, 3))
    sym_rot_points_pred = torch.reshape(rotated_points_pred[list(sym_indices.T)], (-1, num_points, 3))

    # asym_points_pred = tf.reshape(tf.gather_nd(transformed_points_pred, asym_indices), (-1, num_points, 3))
    # asym_points_pred = torch.reshape(transformed_points_pred[list(asym_indices.T)], (-1, num_points, 3))
    asym_rot_points_pred = torch.reshape(rotated_points_pred[list(asym_indices.T)], (-1, num_points, 3))

    # sym_points_target = tf.reshape(tf.gather_nd(transformed_points_target, sym_indices), (-1, num_points, 3))
    # sym_points_target = torch.reshape(transformed_points_target[list(sym_indices.T)], (-1, num_points, 3))
    sym_rot_points_target = torch.reshape(rotated_points_target[list(sym_indices.T)], (-1, num_points, 3))

    # asym_points_target = tf.reshape(tf.gather_nd(transformed_points_target, asym_indices), (-1, num_points, 3))
    # asym_points_target = torch.reshape(transformed_points_target[list(asym_indices.T)], (-1, num_points, 3))
    asym_rot_points_target = torch.reshape(rotated_points_target[list(asym_indices.T)], (-1, num_points, 3))

    # compute transformed point distances
    sym_rot_distances = calc_sym_distances(sym_rot_points_pred, sym_rot_points_target)
    asym_rot_distances = calc_asym_distances(asym_rot_points_pred, asym_rot_points_target)

    if torch.cuda.is_available():
        sym_rot_distances = sym_rot_distances.cuda()
        asym_rot_distances = asym_rot_distances.cuda()

    # distances = tf.concat([sym_distances, asym_distances], axis=0)
    rot_distances = torch.cat((sym_rot_distances, asym_rot_distances), dim=0)

    # loss = tf.math.reduce_mean(distances)
    # loss = torch.mean(distances)

    # rot_loss: 0.1
    # trans_loss: 198
    rotation_loss = torch.mean(rot_distances)  # this is now the rotation loss only

    # Linemod losses
    # Translation starts about 4 times the rotation loss, quickly goes
    # down to be about 2 x the loss. After several epochs rotation loss will be about equivalent to translation
    # translation: 286
    # rotation: 50

    # Syn colibri losses, replicate Linemod loss differential with scaling
    # translation * 2
    # rotation * 500
    # translation: 143
    # rotation: 0.0898 (very small compared to the translation component)

    # Compute the smooth l1 loss for the translation component
    smoothL1Loss = torch.nn.SmoothL1Loss()
    translation_loss = smoothL1Loss(regression_translation, regression_target_translation)

    # in case of no annotations the loss is nan => replace with zero
    # loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    # loss = torch.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    rotation_loss = torch.where(torch.isnan(rotation_loss), torch.zeros_like(rotation_loss), rotation_loss)

    # return loss
    return rotation_loss, translation_loss

def hand_loss(gt_hand, pred_hand):
    # Compute the smooth l1 loss for the hand loss component
    smoothL1Loss = torch.nn.SmoothL1Loss()
    loss = smoothL1Loss(pred_hand, gt_hand)
    return loss


def separate_axis_from_angle(axis_angle_tensor):
    """ Separates the compact 3-dimensional axis_angle representation in the rotation axis and a rotation angle
        Args:
            axis_angle_tensor: tensor with a shape of 3 in the last dimension.
        Returns:
            axis: Tensor of the same shape as the input axis_angle_tensor but containing only the rotation axis and not the angle anymore
            angle: Tensor of the same shape as the input axis_angle_tensor except the last dimension is 1 and contains the rotation angle
        """
    # squared = tf.math.square(axis_angle_tensor)
    squared = torch.mul(axis_angle_tensor, axis_angle_tensor)
    # summed = tf.math.reduce_sum(squared, axis=-1)
    summed = torch.sum(squared, dim=-1)
    # angle = tf.expand_dims(tf.math.sqrt(summed), axis=-1)
    angle = torch.unsqueeze(torch.sqrt(summed), dim=-1)

    axis_angle_tensor_safe = torch.where(torch.isnan(axis_angle_tensor), torch.zeros_like(axis_angle_tensor),
                                         axis_angle_tensor)
    axis = torch.div(axis_angle_tensor_safe, angle)
    # axis = tf.math.divide_no_nan(axis_angle_tensor, angle)

    return axis, angle


def calc_sym_distances(sym_points_pred, sym_points_target):
    """ Calculates the average minimum point distance for symmetric objects
        Args:
            sym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            sym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average minimum point distance between both transformed 3D models
        """
    # sym_points_pred = tf.expand_dims(sym_points_pred, axis=2)
    sym_points_pred = torch.unsqueeze(sym_points_pred, dim=2)
    # sym_points_target = tf.expand_dims(sym_points_target, axis=1)
    sym_points_target = torch.unsqueeze(sym_points_target, dim=1)
    # distances = tf.reduce_min(tf.norm(sym_points_pred - sym_points_target, axis=-1), axis=-1)

    # Handle zero condition
    if (sym_points_pred.shape[0] == 0 & sym_points_target.shape[0] == 0):
        distances = torch.zeros((0, sym_points_pred.shape[1]))
    else:
        distances, _ = torch.min(torch.norm(sym_points_pred - sym_points_target, dim=-1), dim=-1)

    # return tf.reduce_mean(distances, axis=-1)
    return torch.mean(distances, dim=-1)


def calc_asym_distances(asym_points_pred, asym_points_target):
    """ Calculates the average pairwise point distance for asymmetric objects
        Args:
            asym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            asym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average point distance between both transformed 3D models
        """
    # distances = tf.norm(asym_points_pred - asym_points_target, axis=-1)
    distances = torch.norm(asym_points_pred - asym_points_target, dim=-1)

    # return tf.reduce_mean(distances, axis=-1)
    return torch.mean(distances, dim=-1)


# copied and adapted the following functions from tensorflow graphics source because they did not work with unknown shape
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def cross(vector1, vector2, name=None):
    """Computes the cross product between two tensors along an axis.
  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.
  Args:
    vector1: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    vector2: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    axis: The dimension along which to compute the cross product.
    name: A name for this op which defaults to "vector_cross".
  Returns:
    A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension i = axis
    represents the result of the cross product.
  """
    # with tf.compat.v1.name_scope(name, "vector_cross", [vector1, vector2]):
    #     vector1_x = vector1[:, :, 0]
    #     vector1_y = vector1[:, :, 1]
    #     vector1_z = vector1[:, :, 2]
    #     vector2_x = vector2[:, :, 0]
    #     vector2_y = vector2[:, :, 1]
    #     vector2_z = vector2[:, :, 2]
    #     n_x = vector1_y * vector2_z - vector1_z * vector2_y
    #     n_y = vector1_z * vector2_x - vector1_x * vector2_z
    #     n_z = vector1_x * vector2_y - vector1_y * vector2_x
    #     return tf.stack((n_x, n_y, n_z), axis=-1)

    vector1_x = vector1[:, :, 0]
    vector1_y = vector1[:, :, 1]
    vector1_z = vector1[:, :, 2]
    vector2_x = vector2[:, :, 0]
    vector2_y = vector2[:, :, 1]
    vector2_z = vector2[:, :, 2]
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return torch.stack((n_x, n_y, n_z), dim=-1)


# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def dot(vector1, vector2, axis=-1, keepdims=True, name=None):
    """Computes the dot product between two tensors along an axis.
  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.
  Args:
    vector1: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    vector2: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    axis: The dimension along which to compute the dot product.
    keepdims: If True, retains reduced dimensions with length 1.
    name: A name for this op which defaults to "vector_dot".
  Returns:
    A tensor of shape `[A1, ..., Ai = 1, ..., An]`, where the dimension i = axis
    represents the result of the dot product.
  """
    # with tf.compat.v1.name_scope(name, "vector_dot", [vector1, vector2]):
    #     return tf.reduce_sum(
    #         input_tensor=vector1 * vector2, axis=axis, keepdims=keepdims)

    if torch.cuda.is_available():
        vector1 = vector1.cuda()
        vector2 = vector2.cuda()

    return torch.sum(input=(vector1 * vector2), dim=axis, keepdims=keepdims)


# copied from https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py
def rotate(point, axis, angle):
    r"""Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.
  Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
  $$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:
  $$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
  +\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to rotate.
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_rotate".
  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.
  Raises:
    ValueError: If `point`, `axis`, or `angle` are of different shape or if
    their respective shape is not supported.
  """
    # with tf.compat.v1.name_scope(name, "axis_angle_rotate", [point, axis, angle]):
    #     cos_angle = tf.cos(angle)
    #     axis_dot_point = dot(axis, point)
    #     return point * cos_angle + cross(
    #         axis, point) * tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)

    cos_angle = torch.cos(angle)
    axis_dot_point = dot(axis, point)

    if torch.cuda.is_available():
        point = point.cuda()
        axis = axis.cuda()
        axis_dot_point = axis_dot_point.cuda()

    return point * cos_angle + cross(axis, point) * torch.sin(angle) + axis * axis_dot_point * (
                torch.tensor(1.0).type(torch.FloatTensor) - cos_angle)
