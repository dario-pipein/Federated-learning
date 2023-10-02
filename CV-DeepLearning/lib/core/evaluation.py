# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """

    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    #rmse = np.zeros(N)
    rmse = np.zeros((N,6))

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]

        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w or Toronto
            """
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
            intermouth = np.linalg.norm(pts_gt[48, ] - pts_gt[54, ])
            internose = np.linalg.norm(pts_gt[31, ] - pts_gt[35, ])
            intereyebrow = np.linalg.norm(pts_gt[17, ] - pts_gt[26, ])
            interchin = np.linalg.norm(pts_gt[0, ] - pts_gt[16, ])
            """
            box_diagonal = meta['box_diagonal'][i]

        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        
        """
        rmse[i][0] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)
        rmse[i][1] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (intermouth * L)
        rmse[i][2] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (internose * L)
        rmse[i][3] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (intereyebrow * L)
        rmse[i][4] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interchin * L)
        """
        rmse[i][0] = np.sum(np.linalg.norm(pts_pred[36:48] - pts_gt[36:48], axis=(0,1))) / (box_diagonal * 12)
        rmse[i][1] = np.sum(np.linalg.norm(pts_pred[48:] - pts_gt[48:], axis=(0,1))) / (box_diagonal * 20)
        rmse[i][2] = np.sum(np.linalg.norm(pts_pred[27:36] - pts_gt[27:36], axis=(0,1))) / (box_diagonal * 9)
        rmse[i][3] = np.sum(np.linalg.norm(pts_pred[17:27] - pts_gt[17:27], axis=(0,1))) / (box_diagonal * 10)
        rmse[i][4] = np.sum(np.linalg.norm(pts_pred[:17] - pts_gt[:17], axis=(0,1))) / (box_diagonal * 17)
        rmse[i][5] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=(0,1))) / (box_diagonal * L)

    return rmse


def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
