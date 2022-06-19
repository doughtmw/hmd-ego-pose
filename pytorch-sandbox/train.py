import copy
import os
import cv2
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from collections import OrderedDict

from backbone import HMDEgoPose
from hmdegopose.layers import FilterDetections
from hmdegopose.loss import batch_iterate, format_translation, format_bboxes, create_anchors
from eval.common import evaluate_model


# Used for training and loss calculation
class TrainModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs, camera_params, is_losses=False,
                model_3d_points=None,
                classification_gt=None,
                regression_gt=None,
                transformation_gt=None,
                coords_3d_gt=None,
                params=None):

        # Forward pass
        _, regression, classification, rotation, translation_raw, hand = self.model(imgs)

        # Create the anchors
        input_size = params["img_size"][0]
        anchors, translation_anchors = create_anchors(input_size=input_size)

        # Format the raw translation using camera parameters and anchors
        translation = format_translation(translation_anchors, translation_raw,
                                         camera_parameters_input=camera_params)

        if is_losses:
            # Compute the losses
            classification_loss, regression_loss, \
            rotation_loss, translation_loss, \
            hands_loss = batch_iterate(
                gt_classification=classification_gt, classification=classification,
                gt_regression=regression_gt, regression=regression,
                gt_transformation=transformation_gt, transformation=torch.cat((rotation, translation), dim=2),
                gt_hand=coords_3d_gt, hand=hand,
                model_3d_points=model_3d_points, num_rotation_parameter=params["num_rotation_parameters"])

            # Get mean of losses
            classification_loss = classification_loss.mean()  # 250
            regression_loss = regression_loss.mean()  # 28
            rotation_loss = rotation_loss.mean()  # 0.09 (too small relative to translation loss)
            translation_loss = translation_loss.mean()  # 230
            hands_loss = hands_loss.mean()  # 15

            # Weigh the losses relative to one another, 
            # this selection attempted to make the weights of similar
            # relative size during training
            classification_loss *= 1.0  # 250
            regression_loss *= 1.0  # 28
            rotation_loss *= 100  # 0.09 * 100 = 9
            translation_loss *= 0.1  # 230 * 0.1 = 23
            hands_loss *= 1.0  # 15
            loss = classification_loss + regression_loss + rotation_loss + translation_loss + hands_loss

            return [classification_loss, regression_loss, rotation_loss, translation_loss, hands_loss, loss]

        else:
            # Do not compute the losses
            # Get the formatted bounding boxes for validation
            bboxes = format_bboxes(imgs, anchors, regression)

            # filter detections (apply NMS / score threshold / select top-k)
            filter_detections = FilterDetections(num_rotation_parameters=3,
                                                 num_translation_parameters=3,
                                                 score_threshold=0.5,
                                                 max_detections=100)
            filtered_detections = filter_detections([bboxes, classification,
                                                     rotation, translation, hand])

            return filtered_detections


def train(training_generator, validation_generator, model, params):
    os.makedirs(params["validation_image_save_path"], exist_ok=True)

    # Create tensorboard logger
    writer_name = "runs/" + "_img_size_" + str(params['img_size'])
    writer = SummaryWriter(writer_name)

    # Wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = TrainModelWithLoss(model)

    # Optimization
    if not params["fine_tune"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        print("Using Adam optmizer - not fine tuning.")
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, nesterov=True)
        print("Using SGD optimizer - fine tuning.")

    # Reduce LR on training plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)

    # Get the total length of the training and validation dataset
    train_data_len = training_generator.size() * 10
    # train_data_len = 50
    val_data_len = validation_generator.size()
    print('train_data_len:', train_data_len)
    print('val_data_len:', val_data_len)

    # Cache the best mean_mixed_transformed_mean loss of best model
    best_mean_mixed_transformed_mean = 100

    # Transfer to device
    if params['use_cuda']:
        # # Multiple GPU support
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = torch.nn.DataParallel(model)
        model.to(params['device'])
    else:
        model.to(params['device'])


    # Load checkpoint weights if using
    # if params['ckpt'] is not None:
    #     state_dict = torch.load(params['ckpt'])
    #     model.load_state_dict(state_dict)
    #     print("\nSuccessfully loaded pretrained weights from checkpoint.")

    if params['ckpt'] is not None:
        state_dict = torch.load(params['ckpt'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # load params
        print("\nSuccessfully loaded pretrained weights from checkpoint.")

    # Loop over epochs
    for epoch in range(params['epochs']):

        batch_progress = 0
        train_loss = 0
        train_reg_loss = 0
        train_cls_loss = 0
        train_rot_loss = 0
        train_trans_loss = 0
        train_hand_loss = 0
        train_num_of_images = 0
        train_start_time = time.time()

        # Training
        model.train()
        for local_batch, local_labels in training_generator:
            optimizer.zero_grad()

            # Transfer data to GPU
            local_images, local_camera_params, \
            local_classification_gt, \
            local_regression_gt, \
            local_transformation_gt,\
            local_coords_3d_gt = \
                torch.tensor(local_batch[0]).to(params['device']), \
                torch.tensor(local_batch[1]).to(params['device']), \
                torch.tensor(local_labels[0]).to(params['device']), \
                torch.tensor(local_labels[1]).to(params['device']), \
                torch.tensor(local_labels[2]).to(params['device']), \
                torch.tensor(local_labels[3]).to(params['device'])


            if params["is_debug_mano"]:
                # Test the mano pose inputs
                # Plot the image for debugging purposes
                image_numpy = local_images[0,:,:,:].cpu().detach().numpy()
                cv2.imshow("input frame", image_numpy)

            local_images = local_images.permute(0, 3, 1, 2)

            # Model inference to get output
            cls_loss, reg_loss, rot_loss, trans_loss, hand_loss, loss = model(
                local_images, local_camera_params,
                is_losses=True,
                model_3d_points=training_generator.get_all_3d_model_points_array_for_loss(),
                classification_gt=local_classification_gt,
                regression_gt=local_regression_gt,
                transformation_gt=local_transformation_gt,
                coords_3d_gt=local_coords_3d_gt,
                params=params)

            # Increment losses
            train_loss += loss.data.item()
            train_cls_loss += cls_loss.data.item()
            train_reg_loss += reg_loss.data.item()
            train_rot_loss += rot_loss.data.item()
            train_trans_loss += trans_loss.data.item()
            train_hand_loss += hand_loss.data.item()
            train_num_of_images += len(local_labels)

            # Backward pass, next optimizer step
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            batch_progress += 1

            # If we have reached the end of a batch
            if batch_progress * params['batch_size'] >= train_data_len:
                percent = 100.0
                print(
                    '\rBatch progress: {:.2f} % [{} / {}] Loss: [{:.2f}] Cls loss: [{:.2f}] Reg loss: [{:.5f}] Rot loss: [{:.2f}] Trans loss: [{:.2f}] Hand loss: [{:.2f}]'.format(
                        percent,
                        train_data_len,
                        train_data_len,
                        loss.data.item(),
                        cls_loss.data.item(),
                        reg_loss.data.item(),
                        rot_loss.data.item(),
                        trans_loss.data.item(),
                        hand_loss.data.item()),
                    end='\n')
                break
            else:
                percent = batch_progress * params['batch_size'] / train_data_len * 100
                print(
                    '\rBatch progress: {:.2f} % [{} / {}] Loss: [{:.2f}] Cls loss: [{:.2f}] Reg loss: [{:.5f}] Rot loss: [{:.2f}] Trans loss: [{:.2f}] Hand loss: [{:.2f}]'.format(
                        percent,
                        batch_progress * params['batch_size'],
                        train_data_len,
                        loss.data.item(),
                        cls_loss.data.item(),
                        reg_loss.data.item(),
                        rot_loss.data.item(),
                        trans_loss.data.item(),
                        hand_loss.data.item()),
                    end='')

        # Get elapsed time, accuracy and loss for epoch
        train_elapsed_time = time.time() - train_start_time
        train_loss = float(train_loss) / train_num_of_images
        train_cls_loss = float(train_cls_loss) / train_num_of_images
        train_reg_loss = float(train_reg_loss) / train_num_of_images
        train_rot_loss = float(train_rot_loss) / train_num_of_images
        train_trans_loss = float(train_trans_loss) / train_num_of_images
        train_hand_loss = float(train_hand_loss) / train_num_of_images

        # Validation
        model.eval()
        val_start_time = time.time()
        score_threshold = 0.5

        # Compute the relevant metrics across validation set
        mean_add, mean_add_s, mean_mixed_add_and_add_s_metric, \
        mean_mixed_transformed_mean, mean_mixed_transformed_std, \
        mean_rotation_mean, mean_rotation_std, \
        mean_translation_tip_mean, mean_translation_tip_std = \
            evaluate_model(model, 
                validation_generator, params["validation_image_save_path"], 
                params, score_threshold,
                params['device'], writer=writer, epoch=epoch)

        val_elapsed_time = time.time() - val_start_time

        # Step scheduler forward using MixedAveragePointDistanceMean_in_mm
        scheduler.step(mean_mixed_transformed_mean)

        # Print stats for train and validation for this epoch
        print('Epoch: {} / {} \n'
              'Train time: [{} m {:.2f} s ] '
              'Loss: [{:.2f}] '
              'Cls loss: [{:.2f}] '
              'Reg loss: [{:.5f}] '
              'Rot loss: [{:.2f}] '
              'Trans loss: [{:.2f}] '
              'Mano loss: [{:.2f}] \n'
              'Val time: [{} m {:.2f} s] '
              'ADD: [{:.2f}] '
              'ADD-S: [{:.2f}] '
              'ADD(-S): [{:.2f}] '
              'MixedAveragePointDistanceMean_in_mm: [{:.2f}] '
              'MixedAveragePointDistanceStd_in_mm: [{:.2f}] '
              'RotationErrorMean_in_degree: [{:.2f}] '
              'RotationErrorStd_in_degree: [{:.2f}] '
              'MixedAveragePointDistanceTipMean_in_mm: [{:.2f}] '
              'MixedAveragePointDistanceTipStd_in_mm: [{:.2f}] '
            .format(
            epoch, params['epochs'],
            train_elapsed_time // 60, train_elapsed_time % 60,
            train_loss,
            train_cls_loss,
            train_reg_loss,
            train_rot_loss,
            train_trans_loss,
            train_hand_loss,

            val_elapsed_time // 60, val_elapsed_time % 60,
            mean_add,
            mean_add_s,
            mean_mixed_add_and_add_s_metric,
            mean_mixed_transformed_mean,
            mean_mixed_transformed_std,
            mean_rotation_mean,
            mean_rotation_std,
            mean_translation_tip_mean,
            mean_translation_tip_std))

        # Check to see if the model is best performing on validation dataset
        if mean_mixed_transformed_mean < best_mean_mixed_transformed_mean:
            best_mean_mixed_transformed_mean = mean_mixed_transformed_mean

            # Get model weights
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_str = '{}__fold_{}__iter_{}__mixed_t_mean_{:.2f}__epo_{}.pth'.format(
                params["dataset"],
                params["fold"],
                params["iter"],
                best_mean_mixed_transformed_mean,
                epoch)

            # Save only the model parameters
            if not os.path.exists('train_weights'):
                os.makedirs('train_weights')
            torch.save(best_model_wts, "train_weights/" + best_model_str)
            print('Saved model: {} Best epoch: {}\n'.format(
                best_model_str,
                epoch))
        else:
            print('')

        # log training stats to tensorboard
        writer.add_scalar('training loss', float(train_loss), epoch)

    print('Done training model.\n')
    return best_model_str
