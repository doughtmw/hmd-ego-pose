import argparse
import os
import numpy as np
import pickle as pkl
from shutil import copyfile
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

from convert_train_test_val_labels import create_dict

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='real_colibri', type=str, help='dataset for formatting [real_colibri, syn_colibri]')
parser.add_argument('--folds', default=[0, 1, 2, 3, 4], type=np.array, help='fold to use for cross validation (1 -> 5)')
parser.add_argument('--in_dir', default='C:/Users/Mitch/Downloads/', type=str, help='location of downloaded dataset (parent folder)')
parser.add_argument('--out_dir', default='C:/git/public/hmd-ego-pose/datasets/', type=str, help='location to save dataset')
args = parser.parse_args()

# Zero bounding box datapoints -> remove from dataset
# C:/Users/Mitch/Downloads/real_colibri_v1/rgb/rec08_62851033.jpg
# C:/Users/Mitch/Downloads/real_colibri_v1/rgb/rec09_102456033.jpg
# C:/Users/Mitch/Downloads/real_colibri_v1/rgb/rec10_67589388.jpg
# C:/Users/Mitch/Downloads/real_colibri_v1/rgb/rec14_48956033.jpg

# Remove the contribution of hands to the masks and just use the
# instruments. Compute the bbox for each frame as well (though not adjusting here
# for image bounds as in the actual code we are using the mask to recreate the bbox).
def filter_and_binarize_mask(mask_path, out_mask_path, channel_id):

    # Load mask from path
    image = cv2.imread(mask_path)

    # convert image to grayscale, and blur
    b, g, r = cv2.split(image)
    
    # cv2.imshow("b", b)
    # cv2.imshow("g", g)
    # cv2.imshow("r", r) 
    # cv2.waitKey(0)

    thresh = 0
    if (channel_id == 'b'):
        thresh = b
    if (channel_id == 'g'):
        thresh = g
    if (channel_id == 'r'):
        thresh = r

    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)[1] 

    # Write the new thresholded mask to png
    # thresh = cv2.resize(thresh, (512, 512)) 
    cv2.imwrite(out_mask_path, thresh)

    # find contours in thresholded image, then grab the largest
    # one
    contours, _ = cv2.findContours(thresh.copy(), 1, 1)
    
    boundingBox = (0,0,0,0)
    # find the biggest countour (c) by the area
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c) 
        x,y,w,h = cv2.boundingRect(c)    
        (x,y),(w,h), a = rect # a - angle

        box = cv2.boxPoints(rect)
        boundingBox = cv2.boundingRect(box)
        box = np.int0(box) #turn into ints
        boundingBox = np.int0(boundingBox)

        # print("box: ", box)
        # print("boundingBox: ", boundingBox)
        
        image = cv2.drawContours(image.copy(), [box], 0, (0,100,255), 1)
        image = cv2.rectangle(image.copy(), boundingBox, (255,100,0), 2)

        # # show the output image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)

    return boundingBox

def main(pkl_file_list, base_dir, hands_dir, save_path, dict, fold):
    zero_bb_count = 0

    # Parse the json files and clear any previous data
    with open(save_path + "gt_" + str(fold) + ".yml", 'w') as yml:
        yml.write("")

    with open(save_path + "info_" + str(fold) + ".yml", 'w') as yml:
        yml.write("")

    # Create the train and test files
    with open(save_path + "train_" + str(fold) + ".txt", 'w') as txt:
        txt.write("")
    with open(save_path + "test_" + str(fold) + ".txt", 'w') as txt:
        txt.write("")
    with open(save_path + "val_" + str(fold) + ".txt", 'w') as txt:
        txt.write("")

    # Iterate across list of pickle files
    instance_count = 0
    for pkl_file in pkl_file_list:
        # print(pkl_file)

        # Troublesome files, skip for the creation of the train, test, val datasets...
        if pkl_file in ["rec08_62851033.pkl", "rec09_102456033.pkl", "rec10_67589388.pkl", "rec14_48956033.pkl"]:
            print("Skipping current file: ", pkl_file)
            continue

        with open(base_dir + pkl_file, 'rb') as f:
            unpickled = []
            while True:
                try:
                    unpickled.append(pkl.load(f))
                except EOFError:
                    break

            # Rename rgb files and masks to ensure consistency with label scheme, eg: 0001
            # Convert from jpg to png
            # RGB
            # rgb_jpg = Image.open(base_dir[:-5] + "rgb/" + pkl_file[:-3] + "jpg")
            # rgb_jpg.save(rgb_dir + str('{:06}'.format(instance_count)) + ".png")
            rgb_jpg = cv2.imread(base_dir[:-5] + "rgb/" + pkl_file[:-3] + "jpg")
            # rgb_jpg = cv2.resize(rgb_jpg, (512, 512))
            cv2.imwrite(rgb_dir + str('{:06}'.format(instance_count)) + ".png", rgb_jpg)


            # MASK, adjust to remove contribution of hand and binarize
            # THRESHOLD:
            #   20 for synthetic data (to remove the hand)
            bbox = filter_and_binarize_mask(
                base_dir[:-5] + "segm/" + pkl_file[:-3] + "png",
                mask_dir + str('{:06}'.format(instance_count)) + ".png",
                'r') # use the red channel from RGB image

            # [TODO] replicate the test setup from paper
            # Check to see if the current file name exists in a dictionary and
            # whether it belongs to the train, test or val configuration
            file_name = pkl_file[:-4]
            file_config = dict[file_name]

            if file_config == "train":
                with open(save_path + "train_" + str(fold) + ".txt", 'a') as txt:
                    txt.write(str('{:06}'.format(instance_count)) + "\n")
            if file_config == "test":
                with open(save_path + "test_" + str(fold) + ".txt", 'a') as txt:
                    txt.write(str('{:06}'.format(instance_count)) + "\n")
            if file_config == "val":
                with open(save_path + "val_" + str(fold) + ".txt", 'a') as txt:
                    txt.write(str('{:06}'.format(instance_count)) + "\n")

            # Parse the file, gather important parameters and format to copy LINEMOD structure
            # REAL
            # shape: (10, )
            # verts_3d: (778, 3)
            # coords_3d: (21, 3)

            # SYN
            # shape: (10, )
            # verts_3d: (778, 3)
            # coords_3d: (21, 3)
            loaded_data = unpickled[0]

            # Camera extrinsic parameters and calibration parameters
            cam_extr = np.array(loaded_data['cam_extr'])
            cam_calib = loaded_data['cam_calib']
            # cam_calib = loaded_data['cam_calib'] * 2 # when using 512 x 512 images

            # Transform the affine transform to the correct coordinate system
            # Affine transform informs the pose of the rigid object model
            affine_transform = np.array(loaded_data['affine_transform'])

            # Compute drill tip rotation distance as in:
            # https://github.com/jonashein/handobjectnet_baseline/blob/29175be4528f68b8a2aa6dc6aa37ee0a042f93ab/meshreg/netscripts/metrics.py#L218
            # THIS IS ONLY VALID FOR OUR EXACT DRILL MODEL!
            # (4, 1)
            drill_tip_transform = np.array(
                [[1, 0, 0, 0.053554],
                 [0, 1, 0, 0.225361],
                 [0, 0, 1, -0.241646],
                 [0, 0, 0, 1]])
            # drill_shank_transform = np.array(
            #     [[1, 0, 0, 0.057141],
            #      [0, 1, 0, 0.220794],
            #      [0, 0, 1, -0.121545],
            #      [0, 0, 0, 1]])

            # Adjust using the camera extrinsic params
            adj_affine_transform = np.dot(cam_extr, affine_transform)
            drill_tip_transform = np.dot(cam_extr, drill_tip_transform)
            # drill_shank_transform = np.dot(cam_extr, drill_shank_transform)

            # X, Y, Z offset from previous model center to new model center of mass
            # Gathered offset information from the Blender project, recentered the object
            # model using its center in Unity
            # (-0.048 m, -0.1129 m, 0.0845 m)
            recenter_vector = np.array([0.048, 0.1129, -0.0845, 1])

            # Get the new transform offset for affine transform
            t_new = np.dot(adj_affine_transform, recenter_vector)
            t_new_drill_tip = np.dot(drill_tip_transform, recenter_vector)
            # t_new_shank = np.dot(drill_shank_transform, recenter_vector)

            adj_affine_transform[0][3] = t_new[0]
            adj_affine_transform[1][3] = t_new[1]
            adj_affine_transform[2][3] = t_new[2]

            drill_tip_transform[0][3] = t_new_drill_tip[0]
            drill_tip_transform[1][3] = t_new_drill_tip[1]
            drill_tip_transform[2][3] = t_new_drill_tip[2]

            # drill_shank_transform[0][3] = t_new_shank[0]
            # drill_shank_transform[1][3] = t_new_shank[1]
            # drill_shank_transform[2][3] = t_new_shank[2]

            # Hand vertices (778, 3), save to npy files for loading into PyTorch at runtime
            # otherwise too much ram is used for loading data...

            # 3d coords (21, 3)
            coords_3d = np.array(loaded_data['coords_3d'])
            hom_coords3d = np.concatenate([coords_3d, np.ones((coords_3d.shape[0], 1))], 1).transpose()  # shape: (4,21)
            trans_coords_3d = np.dot(cam_extr, hom_coords3d).transpose()  # shape: (21,3)

            # Save hand coords after checking shape
            np.save(hands_dir + str('{:06}'.format(instance_count)) + '_coords_3d.npy', trans_coords_3d)  # save

            # Check shape of each save data
            assert trans_coords_3d.shape == (21, 3)
            assert adj_affine_transform.shape == (3, 4)
            assert bbox.shape == (4,)
            assert drill_tip_transform.shape == (3, 4)
            assert cam_calib.shape == (3, 3)

            # Want to copy format of the gt.yml and info.yml files
            # gt.yml
            #   frame count
            #   per frame rotation matrix (9 element vector)
            #   per frame translation matrix (3 element vector)
            #   object bounding box coordinates (scale?)
            #   object id
            with open(save_path + "gt_" + str(fold) + ".yml", 'a') as yml:
                yml.write(str(instance_count) + ": \n")
                yml.write("- cam_R_m2c: " +
                str([
                    adj_affine_transform[0][0], adj_affine_transform[0][1], adj_affine_transform[0][2],
                    adj_affine_transform[1][0], adj_affine_transform[1][1], adj_affine_transform[1][2],
                    adj_affine_transform[2][0], adj_affine_transform[2][1], adj_affine_transform[2][2]]) + "\n")

                # Scale from metres to mm
                yml.write("  cam_t_m2c: " +
                str([adj_affine_transform[0][3] * 1000, adj_affine_transform[1][3] * 1000, adj_affine_transform[2][3] * 1000]) + "\n")

                # Get the bboxes in code from image masks
                yml.write("  obj_bb: " + str([bbox[2], bbox[3], bbox[0], bbox[1]]) + "\n")
                yml.write("  obj_id: " + str(1) + "\n")

                # Write the drill tip transform for use in calculating the tip error, scale from
                # m to mm
                yml.write("  drill_tip_transform: " +
                str([drill_tip_transform[0][3] * 1000, drill_tip_transform[1][3] * 1000, drill_tip_transform[2][3] * 1000, 1]) + "\n")

            # info.yml
            #   frame count
            #   camera projection matrix (9 element vector)
            #   depth scale coefficient = 1.0
            with open(save_path + "info_" + str(fold) + ".yml", 'a') as yml:
                yml.write(str(instance_count) + ": \n")
                yml.write("  cam_K: " + str([cam_calib[0][0], cam_calib[0][1], cam_calib[0][2], cam_calib[1][0], cam_calib[1][1], cam_calib[1][2], cam_calib[2][0], cam_calib[2][1], cam_calib[2][2]]) + "\n")
                yml.write("  depth_scale: " + str(1.0) + "\n")

        instance_count += 1
        

if __name__ == "__main__":

    if (args.dataset == "real_colibri"):
        args.in_dir = args.in_dir + 'real_colibri_v1/'
        args.out_dir = args.out_dir + 'real_colibri_v1/'
        print("Using real colibri dataset.")

    elif (args.dataset == "syn_colibri"):
        args.in_dir = args.in_dir + 'syn_colibri_v1/'
        args.out_dir = args.out_dir + 'syn_colibri_v1/'
        print("Using syn colibri dataset.")

    else:
        print("No dataset selected.")

    # define the train, test, val splits based on Hein et al.
    base_dir = args.in_dir + "meta/"
    out_dir = args.out_dir + "data/01/"
    rgb_dir = args.out_dir + "data/01/rgb/"
    mask_dir = args.out_dir + "data/01/mask/"
    hands_dir = args.out_dir + "data/01/hands/"

    # create the paths if they don't exist
    if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    if not os.path.exists(rgb_dir):
            os.makedirs(rgb_dir)
    if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
    if not os.path.exists(hands_dir):
            os.makedirs(hands_dir)

    all_train_files = [
        [args.in_dir + "/cv0_train.txt"],
        [args.in_dir + "/cv1_train.txt"],
        [args.in_dir + "/cv2_train.txt"],
        [args.in_dir + "/cv3_train.txt"],
        [args.in_dir + "/cv4_train.txt"]]

    all_test_files = [
        [args.in_dir + "/cv0_test.txt"],
        [args.in_dir + "/cv1_test.txt"],
        [args.in_dir + "/cv2_test.txt"],
        [args.in_dir + "/cv3_test.txt"],
        [args.in_dir + "/cv4_test.txt"]]

    all_val_files = [
        [args.in_dir + "/cv0_val.txt"],
        [args.in_dir + "/cv1_val.txt"],
        [args.in_dir + "/cv2_val.txt"],
        [args.in_dir + "/cv3_val.txt"],
        [args.in_dir + "/cv4_val.txt"]]

    # Iterate across all folds to create data
    for fold in args.folds:
        print("Creating data for fold ", str(fold) + "...")

        # Create the dictionary to relate to the training scheme used in the paper
        # and bin individual file names into train, test, val config
        dict = create_dict(all_train_files[fold], all_test_files[fold], all_val_files[fold])
        print("Finished creating data for fold ", str(fold) + "...")

        all_pkl_files = os.listdir(base_dir)
        sorted_pkl_files = sorted(all_pkl_files)
        print("Begin formatting masks, rgb, and hand data for fold ", str(fold) + "...")
        main(sorted_pkl_files, base_dir, hands_dir, out_dir, dict, fold)
        print("Completed formatting masks, rgb, and hand data for fold ", str(fold) + "...")


