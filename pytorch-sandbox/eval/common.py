"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under

Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from generators.utils.compute_overlap import compute_overlap, wrapper_c_min_distances
from generators.utils.visualization import draw_detections, draw_annotations, draw_mano_coords
from hmdegopose.samplevis import draw_samplevis
from generators.colibri_common import project_and_normalize
from matplotlib import pyplot as plt

import numpy as np
import os
import math
from tqdm import tqdm

import cv2
import progressbar


def evaluate_model(model, generator, save_path, params, score_threshold, device, writer=None, epoch=0, iou_threshold=0.5,
                   max_detections=100,
                   diameter_threshold=0.1, verbose=0):
    """
    Evaluates a given model using the data from the given generator.

    Args:
        model: The model that should be evaluated.
        generator: Generator that loads the dataset to evaluate.
        save_path: Where to save the evaluated images with the drawn annotations and predictions. Or None if the images should not be saved.
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        iou_threshold: Intersection-over-Union (IoU) threshold between the GT and predicted 2D bboxes when a detection is considered to be correct.
        max_detections: Maximum detections per image.
        diameter_threshold: The threshold relative to the 3D model's diameter at which a 6D pose is considered correct.
                            If the average distance between the 3D model points transformed with the GT pose and estimated pose respectively, is lower than this threshold the pose is considered to be correct.

    """
    # run evaluation
    average_precisions, add_metric, add_s_metric, \
    metric_5cm_5degree, translation_diff_metric, rotation_diff_metric, \
    translation_diff_metric_tip, \
    metric_2d_projection, \
    mixed_add_and_add_s_metric, average_point_distance_error_metric, \
    average_sym_point_distance_error_metric, mixed_average_point_distance_error_metric, \
    hand_diff_metric = evaluate(
        generator,
        model,
        params,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
        save_path=save_path,
        diameter_threshold=diameter_threshold,
        device=device
    )

    weighted_average = False
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    if weighted_average:
        mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
    else:
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

    # compute per class ADD Accuracy
    total_instances_add = []
    add_accuracys = []
    for label, (add_acc, num_annotations) in add_metric.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with ADD accuracy: {:.4f}'.format(add_acc))
        total_instances_add.append(num_annotations)
        add_accuracys.append(add_acc)
    if weighted_average:
        mean_add = sum([a * b for a, b in zip(total_instances_add, add_accuracys)]) / sum(total_instances_add)
    else:
        mean_add = sum(add_accuracys) / sum(x > 0 for x in total_instances_add)

    # same for add-s metric
    total_instances_add_s = []
    add_s_accuracys = []
    for label, (add_s_acc, num_annotations) in add_s_metric.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with ADD-S-Accuracy: {:.4f}'.format(add_s_acc))
        total_instances_add_s.append(num_annotations)
        add_s_accuracys.append(add_s_acc)
    if weighted_average:
        mean_add_s = sum([a * b for a, b in zip(total_instances_add_s, add_s_accuracys)]) / sum(total_instances_add_s)
    else:
        mean_add_s = sum(add_s_accuracys) / sum(x > 0 for x in total_instances_add_s)

    # same for 5cm 5degree metric
    total_instances_5cm_5degree = []
    accuracys_5cm_5degree = []
    for label, (acc_5cm_5_degree, num_annotations) in metric_5cm_5degree.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with 5cm-5degree-Accuracy: {:.4f}'.format(acc_5cm_5_degree))
        total_instances_5cm_5degree.append(num_annotations)
        accuracys_5cm_5degree.append(acc_5cm_5_degree)
    if weighted_average:
        mean_5cm_5degree = sum([a * b for a, b in zip(total_instances_5cm_5degree, accuracys_5cm_5degree)]) / sum(
            total_instances_5cm_5degree)
    else:
        mean_5cm_5degree = sum(accuracys_5cm_5degree) / sum(x > 0 for x in total_instances_5cm_5degree)

    # same for translation diffs
    translation_diffs_mean = []
    translation_diffs_std = []
    for label, (t_mean, t_std) in translation_diff_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Translation Differences in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        translation_diffs_mean.append(t_mean)
        translation_diffs_std.append(t_std)
    mean_translation_mean = sum(translation_diffs_mean) / len(translation_diffs_mean)
    mean_translation_std = sum(translation_diffs_std) / len(translation_diffs_std)

    # same for translation diffs tip
    translation_diffs_tip_mean = []
    translation_diffs_tip_std = []
    for label, (t_mean, t_std) in translation_diff_metric_tip.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Tip Translation Differences in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        translation_diffs_tip_mean.append(t_mean)
        translation_diffs_tip_std.append(t_std)
    mean_translation_tip_mean = sum(translation_diffs_tip_mean) / len(translation_diffs_tip_mean)
    mean_translation_tip_std = sum(translation_diffs_tip_std) / len(translation_diffs_tip_std)

    # same for translation diffs hand
    hand_diffs_mean = []
    hand_diffs_std = []
    for label, (t_mean, t_std) in hand_diff_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Hand Differences in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        hand_diffs_mean.append(t_mean)
        hand_diffs_std.append(t_std)
    mean_hand_mean = sum(hand_diffs_mean) / len(hand_diffs_mean)
    mean_hand_std = sum(hand_diffs_std) / len(hand_diffs_std)

    # same for rotation diffs
    rotation_diffs_mean = []
    rotation_diffs_std = []
    for label, (r_mean, r_std) in rotation_diff_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Rotation Differences in degree: Mean: {:.4f} and Std: {:.4f}'.format(r_mean, r_std))
        rotation_diffs_mean.append(r_mean)
        rotation_diffs_std.append(r_std)
    mean_rotation_mean = sum(rotation_diffs_mean) / len(rotation_diffs_mean)
    mean_rotation_std = sum(rotation_diffs_std) / len(rotation_diffs_std)

    # same for 2d projection metric
    total_instances_2d_projection = []
    accuracys_2d_projection = []
    for label, (acc_2d_projection, num_annotations) in metric_2d_projection.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with 2d-projection-Accuracy: {:.4f}'.format(acc_2d_projection))
        total_instances_2d_projection.append(num_annotations)
        accuracys_2d_projection.append(acc_2d_projection)
    if weighted_average:
        mean_2d_projection = sum([a * b for a, b in zip(total_instances_2d_projection, accuracys_2d_projection)]) / sum(
            total_instances_2d_projection)
    else:
        mean_2d_projection = sum(accuracys_2d_projection) / sum(x > 0 for x in total_instances_2d_projection)

    # same for mixed_add_and_add_s_metric
    total_instances_mixed_add_and_add_s_metric = []
    accuracys_mixed_add_and_add_s_metric = []
    for label, (acc_mixed_add_and_add_s_metric, num_annotations) in mixed_add_and_add_s_metric.items():
        if verbose == 1:
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label),
                  'with ADD(-S)-Accuracy: {:.4f}'.format(acc_mixed_add_and_add_s_metric))
        total_instances_mixed_add_and_add_s_metric.append(num_annotations)
        accuracys_mixed_add_and_add_s_metric.append(acc_mixed_add_and_add_s_metric)
    if weighted_average:
        mean_mixed_add_and_add_s_metric = sum([a * b for a, b in zip(total_instances_mixed_add_and_add_s_metric,
                                                                     accuracys_mixed_add_and_add_s_metric)]) / sum(
            total_instances_mixed_add_and_add_s_metric)
    else:
        mean_mixed_add_and_add_s_metric = sum(accuracys_mixed_add_and_add_s_metric) / sum(
            x > 0 for x in total_instances_mixed_add_and_add_s_metric)

    # same for average transformed point distances
    transformed_diffs_mean = []
    transformed_diffs_std = []
    for label, (t_mean, t_std) in average_point_distance_error_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Transformed Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        transformed_diffs_mean.append(t_mean)
        transformed_diffs_std.append(t_std)
    mean_transformed_mean = sum(transformed_diffs_mean) / len(transformed_diffs_mean)
    mean_transformed_std = sum(transformed_diffs_std) / len(transformed_diffs_std)

    # same for average symmetric transformed point distances
    transformed_sym_diffs_mean = []
    transformed_sym_diffs_std = []
    for label, (t_mean, t_std) in average_sym_point_distance_error_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Transformed Symmetric Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        transformed_sym_diffs_mean.append(t_mean)
        transformed_sym_diffs_std.append(t_std)
    mean_transformed_sym_mean = sum(transformed_sym_diffs_mean) / len(transformed_sym_diffs_mean)
    mean_transformed_sym_std = sum(transformed_sym_diffs_std) / len(transformed_sym_diffs_std)

    # same for mixed average transformed point distances for symmetric and asymmetric objects
    mixed_transformed_diffs_mean = []
    mixed_transformed_diffs_std = []
    for label, (t_mean, t_std) in mixed_average_point_distance_error_metric.items():
        if verbose == 1:
            print('class', generator.label_to_name(label),
                  'with Mixed Transformed Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
        mixed_transformed_diffs_mean.append(t_mean)
        mixed_transformed_diffs_std.append(t_std)
    mean_mixed_transformed_mean = sum(mixed_transformed_diffs_mean) / len(mixed_transformed_diffs_mean)
    mean_mixed_transformed_std = sum(mixed_transformed_diffs_std) / len(mixed_transformed_diffs_std)

    if verbose == 1 or verbose == 2:
        print('mAP: {:.4f}'.format(mean_ap))
        print('ADD: {:.4f}'.format(mean_add))
        print('ADD-S: {:.4f}'.format(mean_add_s))
        print('5cm_5degree: {:.4f}'.format(mean_5cm_5degree))
        print('_________________')
        print('TranslationErrorMean_in_mm: {:.4f}'.format(mean_translation_mean))
        print('TranslationErrorStd_in_mm: {:.4f}'.format(mean_translation_std))
        print('RotationErrorMean_in_degree: {:.4f}'.format(mean_rotation_mean))
        print('RotationErrorStd_in_degree: {:.4f}'.format(mean_rotation_std))
        print('TranslationErrorTipMean_in_mm: {:.4f}'.format(mean_translation_tip_mean))
        print('TranslationErrorTipStd_in_mm: {:.4f}'.format(mean_translation_tip_std))
        print('TranslationErrorHandMean_in_mm: {:.4f}'.format(mean_hand_mean))
        print('TranslationErrorHandStd_in_mm: {:.4f}'.format(mean_hand_std))
        print('_________________')
        print('2D-Projection: {:.4f}'.format(mean_2d_projection))
        print('Summed_Translation_Rotation_Error: {:.4f}'.format(
            mean_translation_mean + mean_translation_std + mean_rotation_mean + mean_rotation_std))
        print('ADD(-S): {:.4f}'.format(mean_mixed_add_and_add_s_metric))
        print('AveragePointDistanceMean_in_mm: {:.4f}'.format(mean_transformed_mean))
        print('AveragePointDistanceStd_in_mm: {:.4f}'.format(mean_transformed_std))
        print('AverageSymmetricPointDistanceMean_in_mm: {:.4f}'.format(mean_transformed_sym_mean))
        print('AverageSymmetricPointDistanceStd_in_mm: {:.4f}'.format(mean_transformed_sym_std))
        print('MixedAveragePointDistanceMean_in_mm: {:.4f}'.format(mean_mixed_transformed_mean))
        print('MixedAveragePointDistanceStd_in_mm: {:.4f}'.format(mean_mixed_transformed_std))
        print('\n')

    # Add to tensorboard if not none
    if writer != None:
        writer.add_scalar('mAP', float(mean_ap), epoch)
        writer.add_scalar('ADD', float(mean_add), epoch)
        writer.add_scalar('ADD-S', float(mean_add_s), epoch)
        writer.add_scalar('5cm_5degree', float(mean_5cm_5degree), epoch)
        writer.add_scalar('TranslationErrorMean_in_mm', float(mean_translation_mean), epoch)
        writer.add_scalar('TranslationErrorStd_in_mm', float(mean_translation_std), epoch)
        writer.add_scalar('RotationErrorMean_in_degree', float(mean_rotation_mean), epoch)
        writer.add_scalar('RotationErrorStd_in_degree', float(mean_rotation_std), epoch)
        writer.add_scalar('TranslationErrorTipMean_in_mm', float(mean_translation_tip_mean), epoch)
        writer.add_scalar('TranslationErrorTipStd_in_mm', float(mean_translation_tip_std), epoch)
        writer.add_scalar('2D-Projection', float(mean_2d_projection), epoch)
        writer.add_scalar('Summed_Translation_Rotation_Error',
                          float(mean_translation_mean + mean_translation_std + mean_rotation_mean + mean_rotation_std),
                          epoch)
        writer.add_scalar('ADD(-S)', float(mean_mixed_add_and_add_s_metric), epoch)
        writer.add_scalar('AveragePointDistanceMean_in_mm', float(mean_transformed_mean), epoch)
        writer.add_scalar('AveragePointDistanceStd_in_mm', float(mean_transformed_std), epoch)
        writer.add_scalar('AverageSymmetricPointDistanceMean_in_mm', float(mean_transformed_sym_mean), epoch)
        writer.add_scalar('AverageSymmetricPointDistanceStd_in_mm', float(mean_transformed_sym_std), epoch)
        writer.add_scalar('MixedAveragePointDistanceMean_in_mm', float(mean_mixed_transformed_mean), epoch)
        writer.add_scalar('MixedAveragePointDistanceStd_in_mm', float(mean_mixed_transformed_std), epoch)

    return mean_add, mean_add_s, mean_mixed_add_and_add_s_metric, \
            mean_mixed_transformed_mean, mean_mixed_transformed_std, \
            mean_rotation_mean, mean_rotation_std, \
            mean_translation_tip_mean, mean_translation_tip_std



assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, params, score_threshold=0.05, max_detections=10, save_path=None, device=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (boxes+classes = detections[num_detections, 4 + num_classes], rotations = detections[num_detections, num_rotation_parameters], translations = detections[num_detections, num_translation_parameters)

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    prev_indices = 0
    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image = generator.load_image(i)
        image, scale = generator.preprocess_image(raw_image.copy())
        # image, scale = generator.resize_image(image)
        camera_matrix = generator.load_camera_matrix(i)
        camera_input = generator.get_camera_parameter_input(camera_matrix, scale, generator.translation_scale_norm)

        # if keras.backend.image_data_format() == 'channels_first':
        #     image = image.transpose((2, 0, 1))

        # Send labels and camera params to local device
        local_images, local_camera_params = \
            torch.tensor(np.expand_dims(image, axis=0)).to(device), \
            torch.tensor(np.expand_dims(camera_input, axis=0)).to(device)

        if params["is_debug_mano"]:
            # Test the mano pose inputs
            # Plot the image for debugging purposes
            image_numpy = local_images[0, :, :, :].cpu().detach().numpy()
            cv2.imshow("input frame", image_numpy)

        local_images = local_images.permute(0, 3, 1, 2)

        # predict
        boxes, scores, labels, rotations, translations, hands = model(local_images, local_camera_params, params=params)

        # Reshape for drawing
        boxes = torch.reshape(boxes, (1, boxes.shape[0], 4))
        scores = torch.reshape(scores, (1, scores.shape[0]))
        labels = torch.reshape(labels, (1, labels.shape[0]))
        rotations = torch.reshape(rotations, (1, rotations.shape[0], 3))
        translations = torch.reshape(translations, (1, translations.shape[0], 3))
        hands = torch.reshape(hands, (1, hands.shape[0], 63))

        # convert input to numpy arrays
        boxes = boxes.numpy()
        scores = scores.numpy()
        labels = labels.numpy()
        rotations = rotations.numpy()
        translations = translations.numpy()
        hands = hands.numpy()

        # correct boxes for image scale
        boxes /= scale

        # rescale rotations and translations
        rotations *= math.pi
        height, width, _ = raw_image.shape

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # For multiple box predictions, choose top probability box
        # if scores_sort.shape != torch.Size([1]):
        #     indices = (np.array([indices[0][0]]),)
        #     scores_sort = 0

        # print("scores_sort:", scores_sort)
        # print("indices:", indices)

        image_boxes = boxes[0, indices[scores_sort], :]
        image_rotations = rotations[0, indices[scores_sort], :]
        image_translations = translations[0, indices[scores_sort], :]
        image_hands = hands[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]

        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None and params["is_save_images"]:
            # RGB
            raw_image_samplevis = raw_image
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

            # print("generator.load_annotations(i):", generator.load_annotations(i))
            # print("generator.get_bbox_3d_dict():", generator.get_bbox_3d_dict())
            # print("generator.load_camera_matrix(i):", generator.load_camera_matrix(i))
            # print("generator.label_to_name:", generator.label_to_name)

            draw_annotations(raw_image,
                             generator.load_annotations(i),
                             class_to_bbox_3D=generator.get_bbox_3d_dict(),
                             camera_matrix=generator.load_camera_matrix(i),
                             label_to_name=generator.label_to_name,
                             draw_mano=True)
            draw_detections(raw_image, image_boxes,
                            image_scores,
                            image_labels,
                            image_rotations,
                            image_translations,
                            image_hands,
                            class_to_bbox_3D=generator.get_bbox_3d_dict(),
                            camera_matrix=generator.load_camera_matrix(i),
                            label_to_name=generator.label_to_name,
                            draw_mano=True)
            
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            
            # [TODO] Mitch
            # gt_objverts2d:  torch.Size([16, 43154, 2])
            # pred_objverts2d:  torch.Size([16, 43154, 2])
            # gt_objverts3dw:  torch.Size([16, 43154, 3])
            # pred_objverts3dw:  torch.Size([16, 43154, 3])
            # gt_handjoints2d:  torch.Size([16, 21, 2])
            # pred_handjoints2d:  torch.Size([16, 21, 2])
            # gt_handjoints3d:  torch.Size([16, 21, 3])
            # pred_handjoints3d:  torch.Size([16, 21, 3])

            camera_matrix = generator.load_camera_matrix(i)
            # print("camera_matrix: ", camera_matrix)

            identity_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            # print("identity_matrix: ", identity_matrix)

            SCALE_FACTOR = 1000.0

            # model_3d_points:  (257224, 3)
            model_3d_points = generator.get_models_3d_points_dict()[0]
            # print("model_3d_points: ", model_3d_points.shape)
            # print("model_3d_points: ", model_3d_points)

            # gt rotation and translation
            # rotation_gt:  (1, 3)
            annotations = generator.load_annotations(i)
            rotation_gt = np.reshape(annotations['rotations'][0, :3], (1,3))
            rotation_gt_mat,_ = cv2.Rodrigues(rotation_gt)
            # print("rotation_gt_mat: ", rotation_gt_mat.shape)
            # print("rotation_gt: ", rotation_gt)

            # translation_gt:  (1, 3)
            translation_gt = np.reshape(annotations['translations'][0, :], (1,3)) / SCALE_FACTOR
            # print("translation_gt: ", translation_gt.shape)
            # print("translation_gt: ", translation_gt)

            # pred rotation and translation
            # rotation_pred:  (1, 3)
            rotation_pred = image_rotations
            rotation_pred_mat,_ = cv2.Rodrigues(rotation_pred)
            # print("rotation_pred_mat: ", rotation_pred_mat.shape)
            # print("rotation_pred: ", rotation_pred)

            # translation_pred:  (1, 3)
            translation_pred = image_translations / SCALE_FACTOR
            # print("translation_pred: ", translation_pred.shape)
            # print("translation_pred: ", translation_pred)

            # gt hands 3d
            # gt_handjoints3d:  (21, 3)
            gt_handjoints3d = np.reshape(annotations['coords_3d'][0, :, :], (21, 3))
            # print("gt_handjoints3d: ", gt_handjoints3d.shape)
            # print("gt_handjoints3d: ", gt_handjoints3d)
            
            # pred hands 3d
            # pred_handjoints3d:  (21, 3)
            pred_handjoints3d = np.reshape(image_hands, (21, 3))
            # print("pred_handjoints3d: ", pred_handjoints3d.shape)
            # print("pred_handjoints3d: ", pred_handjoints3d)

            # gt hands 2d
            # gt_handjoints2d:  (21, 2)
            gt_handjoints2d = project_and_normalize(
                gt_handjoints3d,
                camera_matrix)
            # print("gt_handjoints2d: ", gt_handjoints2d.shape)
            # print("gt_handjoints2d: ", gt_handjoints2d)

            # pred hands 2d
            # pred_handjoints2d:  (21, 2)
            pred_handjoints2d = project_and_normalize(
                pred_handjoints3d,
                camera_matrix)
            # print("pred_handjoints2d: ", pred_handjoints2d.shape)
            # print("pred_handjoints2d: ", pred_handjoints2d)

            # gt obj verts 3d
            # gt_objverts3dw:  (257224, 3)
            gt_objverts3dw = np.dot(model_3d_points, rotation_gt_mat.T) + translation_gt
            # print("gt_objverts3dw: ", gt_objverts3dw.shape)
            # print("gt_objverts3dw: ", gt_objverts3dw)

            # pred obj verts 3d
            # pred_objverts3dw:  (257224, 3)
            pred_objverts3dw = np.dot(model_3d_points, rotation_pred_mat.T) + translation_pred
            # print("pred_objverts3dw: ", pred_objverts3dw.shape)
            # print("pred_objverts3dw: ", pred_objverts3dw)

            # gt obj verts 2d
            # gt_objverts2d:  (257224, 2)
            gt_objverts2d, _ = cv2.projectPoints(
                model_3d_points,
                np.float32(rotation_gt), np.float32(translation_gt),
                camera_matrix,
                None)
            gt_objverts2d = np.squeeze(gt_objverts2d)
            # print("gt_objverts2d: ", gt_objverts2d.shape)
            # print("gt_objverts2d: ", gt_objverts2d)

            # pred obj verts 2d
            # pred_objverts2d:  (257224, 2)
            pred_objverts2d, _ = cv2.projectPoints(
                model_3d_points,
                np.float32(rotation_pred), np.float32(translation_pred),
                camera_matrix,
                None)
            pred_objverts2d = np.squeeze(pred_objverts2d)
            # print("pred_objverts2d: ", pred_objverts2d.shape)
            # print("pred_objverts2d: ", pred_objverts2d)

            fig = plt.figure(figsize=(10, 2))
            draw_samplevis(
                raw_image_samplevis, 
                gt_objverts2d, pred_objverts2d,
                gt_objverts3dw, pred_objverts3dw,
                gt_handjoints2d, pred_handjoints2d,
                gt_handjoints3d, pred_handjoints3d,
                os.path.join(save_path, 'samplevis_{}.png'.format(i)),
                fig=fig
            )


        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = (image_detections[image_detections[:, -1] == label, :-1],
                                        image_rotations[image_detections[:, -1] == label, :],
                                        image_translations[image_detections[:, -1] == label, :],
                                        image_hands[image_detections[:, -1] == label, :])

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (bboxes = annotations[num_detections, 5], rotations = annotations[num_detections, num_rotation_parameters], translations = annotations[num_detections, num_translation_parameters])

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = (annotations['bboxes'][annotations['labels'] == label, :].copy(),
                                         annotations['rotations'][annotations['labels'] == label, :].copy(),
                                         annotations['translations'][annotations['labels'] == label, :].copy(),
                                         annotations['coords_3d'][annotations['labels'] == label, :].copy())

    return all_annotations


def check_6d_pose_2d_reprojection(model_3d_points, rotation_gt, translation_gt, rotation_pred, translation_pred,
                                  camera_matrix, pixel_threshold=5.0):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the 2D reprojection metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        camera_matrix: numpy array with shape (3, 3) containing the camera matrix
        pixel_threshold: Threshold in pixels when a prdicted 6D pose in considered to be correct
    # Returns
        Boolean indicating wheter the predicted 6D pose is correct or not
    """
    # transform points into camera coordinate system with gt and prediction transformation parameters respectively
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred

    # project the points on the 2d image plane
    points_2D_gt, _ = np.squeeze(
        cv2.projectPoints(transformed_points_gt, np.zeros((3,)), np.zeros((3,)), camera_matrix, None))
    points_2D_pred, _ = np.squeeze(
        cv2.projectPoints(transformed_points_pred, np.zeros((3,)), np.zeros((3,)), camera_matrix, None))

    distances = np.linalg.norm(points_2D_gt - points_2D_pred, axis=-1)
    mean_distances = np.mean(distances)

    if mean_distances <= pixel_threshold:
        is_correct = True
    else:
        is_correct = False

    return is_correct


def check_6d_pose_add(model_3d_points, model_3d_diameter, rotation_gt, translation_gt, rotation_pred, translation_pred,
                      diameter_threshold=0.1):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the ADD metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        model_3d_diameter: Diameter of the object
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        mean_distances: The average distance between the object's 3D points transformed with the predicted and ground truth 6D pose respectively
        transformed_points_gt: numpy array with shape (num_3D_points, 3) containing the object's 3D points transformed with the ground truth 6D pose
    """
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred

    distances = np.linalg.norm(transformed_points_gt - transformed_points_pred, axis=-1)
    mean_distances = np.mean(distances)

    if mean_distances <= (model_3d_diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False

    return is_correct, mean_distances, transformed_points_gt


def check_6d_pose_add_s(model_3d_points, model_3d_diameter, rotation_gt, translation_gt, rotation_pred,
                        translation_pred, diameter_threshold=0.1, max_points=1000):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the ADD-S metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        model_3d_diameter: Diameter of the object
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
        max_points: Max number of 3D points to calculate the distances (The computed distance between all points to all points can be very memory consuming)
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        mean_distances: The average distance between the object's 3D points transformed with the predicted and ground truth 6D pose respectively
    """
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred
    # calc all distances between all point pairs and get the minimum distance for every point
    num_points = transformed_points_gt.shape[0]

    # approximate the add-s metric and use max max_points of the 3d model points to reduce computational time
    step = num_points // max_points + 1

    min_distances = wrapper_c_min_distances(transformed_points_gt[::step, :], transformed_points_pred[::step, :])
    mean_distances = np.mean(min_distances)

    if mean_distances <= (model_3d_diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False

    return is_correct, mean_distances


def calc_translation_diff(translation_gt, translation_pred):
    """ Computes the distance between the predicted and ground truth translation

    # Arguments
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        The translation distance
    """
    return np.linalg.norm(translation_gt - translation_pred)


def calc_rotation_diff(rotation_gt, rotation_pred):
    """ Calculates the distance between two rotations in degree
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
    # Returns
        the rotation distance in degree
    """
    rotation_diff = np.dot(rotation_pred, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = (trace - 1.) / 2.
    if trace < -1.:
        trace = -1.
    elif trace > 1.:
        trace = 1.
    angular_distance = np.rad2deg(np.arccos(trace))

    return abs(angular_distance)


def check_6d_pose_5cm_5degree(rotation_gt, translation_gt, rotation_pred, translation_pred, DRILL_TIP):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the 5cm 5 degree metric
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py def cm_degree_5_metric(self, pose_pred, pose_targets):
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        translation_distance: the translation distance
        rotation_distance: the rotation distance
    """
    translation_distance = calc_translation_diff(translation_gt, translation_pred)

    rotation_distance = calc_rotation_diff(rotation_gt, rotation_pred)

    # (1, 4) -> (4, 1)
    DRILL_TIP = np.transpose(DRILL_TIP)

    # Compute drill tip rotation distance as in:
    # https://github.com/jonashein/handobjectnet_baseline/blob/29175be4528f68b8a2aa6dc6aa37ee0a042f93ab/meshreg/netscripts/metrics.py#L218
    # Compute drill bit position, orientation
    # THIS IS ONLY VALID FOR OUR EXACT DRILL MODEL!
    # (4, )
    # Scale from m to mm
    # Syn_colibri
    # DRILL_TIP = np.array([0.101554 * 1000, -0.338261 * 1000, 0.326146 * 1000, 1])
    # DRILL_SHANK = np.array([0.105141 * 1000, -0.333694 * 1000, 0.206045 * 1000, 1])

    # reshape (3, ) to (3, 1)
    translation_gt = np.expand_dims(translation_gt, axis=1)
    translation_pred = np.expand_dims(translation_pred, axis=1)

    # Compose the transform (3, 4)
    T_gt = np.concatenate([rotation_gt, translation_gt], axis=1)
    T_pred = np.concatenate([rotation_pred, translation_pred], axis=1)

    # Transform the points, gather rotation and transformation for tip error
    translation_gt_tip = np.dot(T_gt, DRILL_TIP)
    # T_gt_shank = np.dot(T_gt, DRILL_SHANK)
    translation_pred_tip = np.dot(T_pred, DRILL_TIP)
    # T_pred_shank = np.dot(T_pred, DRILL_SHANK)

    # Grab the translation components of tip and compute distance metric
    translation_distance_tip = calc_translation_diff(translation_gt_tip, translation_pred_tip)

    # # Follow the approach by Jonas in the handobjectnet baseline repo
    # translation_gt_vec = translation_gt_tip - T_gt_shank
    # translation_gt_vec = translation_gt_vec / np.expand_dims(np.linalg.norm(translation_gt_vec), axis=0)
    #
    # T_pred_vec = translation_pred_tip - T_pred_shank
    # T_pred_vec = T_pred_vec / np.expand_dims(np.linalg.norm(T_pred_vec), axis=0)
    #
    # rotation_distance_tip = calc_rotation_diff(translation_gt_vec, T_pred_vec)

    if translation_distance <= 50 and rotation_distance <= 5:
        is_correct = True
    else:
        is_correct = False

    return is_correct, translation_distance, rotation_distance, translation_distance_tip


def test_draw(image, camera_matrix, points_3d):
    """ Projects and draws 3D points onto a 2D image and shows the image for debugging purposes

    # Arguments
        image: The image to draw on
        camera_matrix: numpy array with shape (3, 3) containing the camera matrix
        points_3d: numpy array with shape (num_3D_points, 3) containing the 3D points to project and draw (usually the object's 3D points transformed with the ground truth 6D pose)
    """
    points_2D, jacobian = cv2.projectPoints(points_3d, np.zeros((3,)), np.zeros((3,)), camera_matrix, None)
    points_2D = np.squeeze(points_2D)
    points_2D = np.copy(points_2D).astype(np.int32)

    tuple_points = tuple(map(tuple, points_2D))
    for point in tuple_points:
        cv2.circle(image, point, 2, (255, 0, 0), -1)

    cv2.imshow('image', image)
    cv2.waitKey(0)


def evaluate(
        generator,
        model,
        params,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=10,
        save_path=None,
        diameter_threshold=0.1,
        device=None,
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save images with visualized detections to.
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        Several dictionaries mapping class names to the computed metrics.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, params,
                                     score_threshold=score_threshold, max_detections=max_detections,
                                     save_path=save_path, device=device)
    all_annotations = _get_annotations(generator)
    all_3d_models = generator.get_models_3d_points_dict()
    all_3d_model_diameters = generator.get_objects_diameter_dict()
    average_precisions = {}
    add_metric = {}
    add_s_metric = {}
    metric_5cm_5degree = {}
    translation_diff_metric = {}
    translation_diff_metric_tip = {}
    hand_diff_metric = {}
    rotation_diff_metric = {}
    metric_2d_projection = {}
    mixed_add_and_add_s_metric = {}
    average_point_distance_error_metric = {}
    average_sym_point_distance_error_metric = {}
    mixed_average_point_distance_error_metric = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        true_positives_add = np.zeros((0,))
        true_positives_add_s = np.zeros((0,))
        model_3d_points = all_3d_models[label]
        model_3d_diameter = all_3d_model_diameters[label]
        true_positives_5cm_5degree = np.zeros((0,))
        translation_diffs = np.zeros((0,))
        translation_diffs_tip = np.zeros((0,))
        rotation_diffs = np.zeros((0,))
        true_positives_2d_projection = np.zeros((0,))
        point_distance_errors = np.zeros((0,))
        point_sym_distance_errors = np.zeros((0,))
        hand_diffs = np.zeros((0,))

        for i in tqdm(range(generator.size())):
            detections = all_detections[i][label][0]
            detections_rotations = all_detections[i][label][1]
            detections_translations = all_detections[i][label][2]
            detections_hand = all_detections[i][label][3]
            annotations = all_annotations[i][label][0]
            annotations_rotations = all_annotations[i][label][1]
            annotations_translations = all_annotations[i][label][2]
            annotations_hand = all_annotations[i][label][3]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d, d_rotation, d_translation, d_hand in zip(detections, detections_rotations, detections_translations, detections_hand):
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
                assigned_rotation = annotations_rotations[assigned_annotation, :3]
                assigned_translation = annotations_translations[assigned_annotation, :]
                assigned_hand = annotations_hand[assigned_annotation, :]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    # correct 2d object detection => check if the 6d pose is also correct
                    is_correct_6d_pose_add, mean_distances_add, transformed_points_gt = check_6d_pose_add(
                        model_3d_points,
                        model_3d_diameter,
                        rotation_gt=generator.axis_angle_to_rotation_mat(assigned_rotation),
                        translation_gt=np.squeeze(assigned_translation),
                        rotation_pred=generator.axis_angle_to_rotation_mat(d_rotation),
                        translation_pred=d_translation,
                        diameter_threshold=diameter_threshold)

                    # [TODO] measure the 3D hand errors as the mean end-point error in mm over the 21 hand joints
                    # (63, ) -> (21, 3)
                    d_hand = np.reshape(d_hand, (21, 3))
                    # print("d_hand: ", d_hand.shape)
                    # (1, 21, 3)
                    # print("assigned_hand: ", assigned_hand.shape)
                    
                    hand_distances_add = np.linalg.norm(np.squeeze(assigned_hand) - d_hand, axis=-1)
                    mean_hand_distances_add = np.mean(hand_distances_add) * 1000.0 # convert to mm
                    # print("mean_hand_distances_add: ", mean_hand_distances_add)

                    is_correct_6d_pose_add_s, mean_distances_add_s = check_6d_pose_add_s(model_3d_points,
                                                                                         model_3d_diameter,
                                                                                         rotation_gt=generator.axis_angle_to_rotation_mat(
                                                                                             assigned_rotation),
                                                                                         translation_gt=np.squeeze(
                                                                                             assigned_translation),
                                                                                         rotation_pred=generator.axis_angle_to_rotation_mat(
                                                                                             d_rotation),
                                                                                         translation_pred=d_translation,
                                                                                         diameter_threshold=diameter_threshold)

                    is_correct_6d_pose_5cm_5degree, \
                    translation_distance, rotation_distance,\
                    translation_distance_tip = check_6d_pose_5cm_5degree(
                        rotation_gt=generator.axis_angle_to_rotation_mat(assigned_rotation),
                        translation_gt=np.squeeze(assigned_translation),
                        rotation_pred=generator.axis_angle_to_rotation_mat(d_rotation),
                        translation_pred=d_translation,
                        DRILL_TIP=generator.load_drill_tip_offset(i))

                    is_correct_2d_projection = check_6d_pose_2d_reprojection(model_3d_points,
                                                                             rotation_gt=generator.axis_angle_to_rotation_mat(
                                                                                 assigned_rotation),
                                                                             translation_gt=np.squeeze(
                                                                                 assigned_translation),
                                                                             rotation_pred=generator.axis_angle_to_rotation_mat(
                                                                                 d_rotation),
                                                                             translation_pred=d_translation,
                                                                             camera_matrix=generator.load_camera_matrix(
                                                                                 i),
                                                                             pixel_threshold=5.0)

                    # #draw transformed gt points in image to test the transformation
                    # test_draw(generator.load_image(i), generator.load_camera_matrix(i), transformed_points_gt)

                    if is_correct_6d_pose_add:
                        true_positives_add = np.append(true_positives_add, 1)
                    if is_correct_6d_pose_add_s:
                        true_positives_add_s = np.append(true_positives_add_s, 1)
                    if is_correct_6d_pose_5cm_5degree:
                        true_positives_5cm_5degree = np.append(true_positives_5cm_5degree, 1)
                    if is_correct_2d_projection:
                        true_positives_2d_projection = np.append(true_positives_2d_projection, 1)

                    translation_diffs = np.append(translation_diffs, translation_distance)
                    translation_diffs_tip = np.append(translation_diffs_tip, translation_distance_tip)
                    rotation_diffs = np.append(rotation_diffs, rotation_distance)
                    point_distance_errors = np.append(point_distance_errors, mean_distances_add)
                    point_sym_distance_errors = np.append(point_sym_distance_errors, mean_distances_add_s)
                    hand_diffs = np.append(hand_diffs, mean_hand_distances_add)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        # compute add accuracy
        add_accuracy = np.sum(true_positives_add) / num_annotations
        add_metric[label] = add_accuracy, num_annotations

        # compute add-s accuracy
        add_s_accuracy = np.sum(true_positives_add_s) / num_annotations
        add_s_metric[label] = add_s_accuracy, num_annotations

        # compute 5cm 5degree accuracy
        accuracy_5cm_5degree = np.sum(true_positives_5cm_5degree) / num_annotations
        metric_5cm_5degree[label] = accuracy_5cm_5degree, num_annotations

        # compute the mean and std of the translation- and rotation differences
        mean_translations = np.mean(translation_diffs)
        std_translations = np.std(translation_diffs)
        translation_diff_metric[label] = mean_translations, std_translations

        mean_translations_tip = np.mean(translation_diffs_tip)
        std_translations_tip = np.std(translation_diffs_tip)
        translation_diff_metric_tip[label] = mean_translations_tip, std_translations_tip

        mean_rotations = np.mean(rotation_diffs)
        std_rotations = np.std(rotation_diffs)
        rotation_diff_metric[label] = mean_rotations, std_rotations

        # [TODO] compute mean and std of the hand vertex positions
        mean_hand = np.mean(hand_diffs)
        std_hand = np.std(hand_diffs)
        hand_diff_metric[label] = mean_hand, std_hand

        # compute 2d projection accuracy
        accuracy_2d_projection = np.sum(true_positives_2d_projection) / num_annotations
        metric_2d_projection[label] = accuracy_2d_projection, num_annotations

        # compute the mean and std of the transformed point errors
        mean_point_distance_errors = np.mean(point_distance_errors)
        std_point_distance_errors = np.std(point_distance_errors)
        average_point_distance_error_metric[label] = mean_point_distance_errors, std_point_distance_errors

        # compute the mean and std of the symmetric transformed point errors
        mean_point_sym_distance_errors = np.mean(point_sym_distance_errors)
        std_point_sym_distance_errors = np.std(point_sym_distance_errors)
        average_sym_point_distance_error_metric[label] = mean_point_sym_distance_errors, std_point_sym_distance_errors

    # fill in the add values for asymmetric objects and add-s for symmetric objects
    for label, add_tuple in add_metric.items():
        add_s_tuple = add_s_metric[label]
        if generator.class_labels_to_object_ids[label] in generator.symmetric_objects:
            mixed_add_and_add_s_metric[label] = add_s_tuple
        else:
            mixed_add_and_add_s_metric[label] = add_tuple

    # fill in the average point distance values for asymmetric objects and the corresponding average sym point distances for symmetric objects
    for label, asym_tuple in average_point_distance_error_metric.items():
        sym_tuple = average_sym_point_distance_error_metric[label]
        if generator.class_labels_to_object_ids[label] in generator.symmetric_objects:
            mixed_average_point_distance_error_metric[label] = sym_tuple
        else:
            mixed_average_point_distance_error_metric[label] = asym_tuple

    return average_precisions, add_metric, add_s_metric, metric_5cm_5degree, translation_diff_metric, rotation_diff_metric, translation_diff_metric_tip, metric_2d_projection, mixed_add_and_add_s_metric, average_point_distance_error_metric, average_sym_point_distance_error_metric, mixed_average_point_distance_error_metric, hand_diff_metric
