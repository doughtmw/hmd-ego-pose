import torch
import numpy as np
from libyana.visutils.viz2d import visualize_joints_2d
import random
# from meshreg.datasets.queries import  , TransQueries
from hmdegopose import consistdisplay


def get_check_none(data, key, cpu=True):
    if key in data and data[key] is not None:
        if cpu:
            return data[key].cpu().detach()
        else:
            return data[key].detach().cuda()
    else:
        return None


# def sample_vis(sample, results, save_img_path, fig=None, max_rows=5, display_centered=False):
def draw_samplevis(
    raw_image, 
    gt_objverts2d, pred_objverts2d,
    gt_objverts3dw, pred_objverts3dw,
    gt_handjoints2d, pred_handjoints2d,
    gt_handjoints3d, pred_handjoints3d,
    save_img_path, fig=None, max_rows=5, display_centered=False):
    fig.clf()
    
    # raw_image = np.transpose(raw_image, (1, 2, 0)) + 0.5
    # images = sample[TransQueries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
    batch_size = 1
    
    row_nb = min(max_rows, batch_size)
    if display_centered:
        col_nb = 7
    else:
        col_nb = 4
    axes = fig.subplots(row_nb, col_nb)

    # Column 0
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    # Visualize 2D hand joints
    if pred_handjoints2d is not None:
        visualize_joints_2d(axes[0], pred_handjoints2d, alpha=0.75, joint_idxs=False, color="r")
    if gt_handjoints2d is not None:
        visualize_joints_2d(axes[0], gt_handjoints2d, alpha=0.5, joint_idxs=False, color="b")

    # Column 1
    axes[1].imshow(raw_image)
    axes[1].axis("off")

    # Visualize 2D object vertices
    if pred_objverts2d is not None:
        axes[1].scatter(
            pred_objverts2d[:, 0], pred_objverts2d[:, 1], c="r", s=1, alpha=0.0075
        )
    if gt_objverts2d is not None:
        axes[1].scatter(
            gt_objverts2d[:, 0], gt_objverts2d[:, 1], c="b", s=1, alpha=0.005
        )

    # Visualize some (vertex position) errors for the 2D object vertices
    if gt_objverts2d is not None and pred_objverts2d is not None:
        
        # Sample evenly spaced vertex positions across drill fbx model
        # idxs = list(random.sample(range(0, 257223), 6))
        # idxs = list(range(6))
        idxs = list([0, 42870, 85740, 128610, 171480, 214350, 257220])
        arrow_nb = len(idxs)

        # gt_objverts2d[:, idxs].float():  torch.Size([6, 2])
        # pred_objverts2d[:, idxs].float():  torch.Size([6, 2])
        # gt_objverts2d = np.reshape(gt_objverts2d, (1, 257224, 2))
        # pred_objverts2d = np.reshape(pred_objverts2d, (1, 257224, 2))
        # arrows = torch.cat([torch.tensor(gt_objverts2d[:, idxs]).float(), torch.tensor(pred_objverts2d[:, idxs]).float()], 1)
        # links = [[i, i + arrow_nb] for i in range(arrow_nb)]
        # visualize_joints_2d(
        #     axes[1],
        #     arrows[0],
        #     alpha=0.5,
        #     joint_idxs=False,
        #     links=links,
        #     color=["k"] * arrow_nb,
        # )

        # Column 2
        # view from the top
        col_idx = 2
        # axes[row_idx, col_idx].set_title("rotY: {:.1f}".format(gt_drill_angle_Y[row_idx]))
        if gt_objverts3dw is not None:
            axes[col_idx].scatter(
                gt_objverts3dw[:, 2], gt_objverts3dw[:, 0], c="b", s=1, alpha=0.005
            )
        if pred_objverts3dw is not None:
            axes[col_idx].scatter(
                pred_objverts3dw[:, 2], pred_objverts3dw[:, 0], c="r", s=1, alpha=0.0075
            )
        if pred_handjoints3d is not None:
            visualize_joints_2d(
                axes[col_idx], pred_handjoints3d[:, [2, 0]], alpha=0.75, joint_idxs=False, color="r"
            )
        if gt_handjoints3d is not None:
            visualize_joints_2d(
                axes[col_idx], gt_handjoints3d[:, [2, 0]], alpha=0.5, joint_idxs=False, color="b"
            )
        axes[col_idx].invert_yaxis()

        # Column 3
        # view from the right
        col_idx = 3
        # axes[row_idx, col_idx].set_title("rotX: {:.1f}".format(gt_drill_angle_X[row_idx]))
        # invert second axis here for more consistent viewpoints
        if gt_objverts3dw is not None:
            axes[col_idx].scatter(
                gt_objverts3dw[:, 2], -gt_objverts3dw[:, 1], c="b", s=1, alpha=0.005
            )
        if pred_objverts3dw is not None:
            axes[col_idx].scatter(
                pred_objverts3dw[:, 2], -pred_objverts3dw[:, 1], c="r", s=1, alpha=0.0075
            )
        if pred_handjoints3d is not None:
            pred_handjoints3d_inv = np.stack([pred_handjoints3d[:, 2], -pred_handjoints3d[:, 1]], axis=-1)
            visualize_joints_2d(
                axes[col_idx], pred_handjoints3d_inv, alpha=0.75, joint_idxs=False, color="r"
            )
        if gt_handjoints3d is not None:
            gt_handjoints3d_inv = np.stack([gt_handjoints3d[:, 2], -gt_handjoints3d[:, 1]], axis=-1)
            visualize_joints_2d(
                axes[col_idx], gt_handjoints3d_inv, alpha=0.5, joint_idxs=False, color="b"
            )

    consistdisplay.squashfig(fig)
    fig.savefig(save_img_path, dpi=300)
