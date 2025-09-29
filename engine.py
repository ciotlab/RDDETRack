from typing import Iterable, Dict, List
import numpy as np
from pathlib import Path
import os
import mpl_toolkits.mplot3d.axes3d as p3
import torch
import wandb
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from einops import rearrange

import util.misc as utils
from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, box_3d_iou
from models.matcher import HungarianMatcher


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, weight_dict: Dict, optimizer: torch.optim.Optimizer,
                    lr_scheduler, device: torch.device, epoch: int, num_frames, max_norm: float = 0, use_wandb=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(data_loader, print_freq, header):
        point_cloud = torch.tensor(data['point_cloud']).to(device)
        point_cloud_padding_mask = torch.tensor(data['point_cloud_padding_mask']).to(device)
        boxes = data['box']
        keypoints = data['keypoint']
        id = data['id']

        outputs, targets = model(point_cloud, point_cloud_padding_mask, boxes, keypoints, id, num_frames)
        loss_dict = criterion(outputs, targets)
        total_loss = torch.tensor(0.0, device=device)
        for k in loss_dict.keys():
            if k in weight_dict:
                total_loss += loss_dict[k] * weight_dict[k]
        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()
        if use_wandb:
            wandb.log({"loss": total_loss.item(), "loss_keypoints": loss_dict['loss_keypoints'],
                       "loss_boxes": loss_dict['loss_boxes'], "loss_giou": loss_dict['loss_giou'],
                       "loss_object": loss_dict['loss_object']})
        metric_logger.update(loss=total_loss.item(), **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, area_min, area_size, num_keypoints, device, keypoint_thresh_list, nms_iou_thresh=0.5,
             matching_iou_thresh=0.5, conf_thresh=0.9):
    model.to(device).eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    conf_matched_list = []
    total_num_target = 0
    skeleton_list = []
    keypoint_error_list = []
    for data in metric_logger.log_every(data_loader, 10, header):
        point_cloud = torch.tensor(data['point_cloud']).to(device)
        point_cloud_padding_mask = torch.tensor(data['point_cloud_padding_mask']).to(device)
        boxes = data['box']
        keypoints = data['keypoint']
        id = data['id']
        targets = {'boxes':boxes, 'keypoints':keypoints, 'id':id}

        outputs, _ = model(point_cloud, point_cloud_padding_mask, boxes, keypoints, id)
        outputs = nms(outputs[0], iou_thresh=nms_iou_thresh)
        conf_matched, num_target, skeleton, keypoint_error \
            = get_batch_statistics(outputs, targets, area_min, area_size, num_keypoints, matching_iou_thresh, conf_thresh)
        conf_matched_list.append(conf_matched)
        total_num_target += num_target
        skeleton_list.extend(skeleton)
        keypoint_error_list.append(keypoint_error)
    conf_matched = torch.concat(conf_matched_list, dim=0)
    conf_matched = conf_matched.cpu().numpy()
    ap, pr_curve = compute_ap(conf_matched, total_num_target)
    keypoint_error = np.concatenate(keypoint_error_list, axis=0)
    keypoint_pck, per_keypoint_pck, keypoint_cdf, per_keypoint_cdf \
        = compute_pck(keypoint_error, keypoint_thresh_list)
    stats = {'AP': ap, 'pr_curve': pr_curve, 'keypoint_pck': keypoint_pck, 'per_keypoint_pck': per_keypoint_pck,
             'keypoint_cdf': keypoint_cdf, 'per_keypoint_cdf': per_keypoint_cdf, 'skeleton': skeleton_list}
    return stats


def nms(outputs, iou_thresh):
    batch_size, num_queries = outputs["pred_boxes"].shape[:2]
    for b in range(batch_size):
        out_boxes = rearrange(outputs['pred_boxes'][b], "q p c -> q (p c)")
        out_confidence = outputs['pred_confidence'][b, :, 0]
        sort_idx = torch.argsort(out_confidence, dim=0, descending=True).long()
        out_boxes = box_3d_cxcyczwhd_to_xyzxyz(out_boxes[sort_idx, :])
        out_confidence = out_confidence[sort_idx]
        iou, union = box_3d_iou(out_boxes, out_boxes)
        overlap = iou > iou_thresh
        for idx, conf in enumerate(out_confidence):
            if conf > 0:
                zero_idx = torch.full_like(out_confidence, False, dtype=torch.bool)
                zero_idx[idx+1:] = overlap[idx, idx+1:]
                out_confidence[zero_idx] = 0
        outputs['pred_confidence'][b, sort_idx, 0] = out_confidence
    return outputs


def get_batch_statistics(outputs, targets, area_min, area_size, num_keypoints, iou_thresh, conf_thresh):
    matcher = HungarianMatcher(cost_boxes=0, cost_keypoint=0, cost_giou=0, cost_obj=1,
                               iou_thresh=iou_thresh)
    indices = matcher(outputs, targets)
    pred_confidence = outputs['pred_confidence']
    pred_boxes = outputs['pred_boxes']
    pred_keypoints = outputs['pred_keypoints']
    tgt_boxes = targets['boxes']
    tgt_keypoints = targets['keypoints']
    conf_matched = []
    total_num_target = 0
    # extract data for AP computation
    for i, index in enumerate(indices):
        conf = pred_confidence[i]
        matched = torch.zeros_like(conf)
        matched[index[0]] = 1
        conf_matched.append(torch.concat((conf, matched), 1))
        num_target = tgt_boxes[i].shape[0]
        total_num_target += num_target
    conf_matched = torch.concat(conf_matched, 0)
    # extract skeleton and compute keypoint errors
    skeleton = []
    keypoint_error = []
    area_min, area_size = np.array(area_min), np.array(area_size)
    for i, index in enumerate(indices):
        conf = pred_confidence[i, :, 0].cpu().numpy()
        valid = conf > conf_thresh
        pbox = pred_boxes[i, valid, :, :].cpu().numpy()
        pkpt = pred_keypoints[i, valid, :, :].cpu().numpy()
        tbox = tgt_boxes[i]
        tkpt = tgt_keypoints[i]
        pbox_center = pbox[:, 0:1, :] * area_size + area_min
        pbox_size = pbox[:, 1:2, :] * area_size
        pbox_min = pbox_center - pbox_size / 2
        pkpt_orig = pkpt * pbox_size + pbox_min
        tbox_center = tbox[:, 0:1, :] * area_size + area_min
        tbox_size = tbox[:, 1:2, :] * area_size
        tbox_min = tbox_center - tbox_size / 2
        tkpt_orig = tkpt * tbox_size + tbox_min
        skeleton.append((pkpt_orig, tkpt_orig))
        valid_ind = np.arange(conf.shape[0])[valid]
        for p_idx, t_idx in zip(index[0].cpu().numpy(), index[1].cpu().numpy()):
            p_vidx = np.where(valid_ind == p_idx)[0]
            if p_vidx.size == 1:
                keypoint_error.append(np.linalg.norm(pkpt_orig[p_vidx[0]] - tkpt_orig[t_idx], axis=1))
    if len(keypoint_error) > 0:
        keypoint_error = np.stack(keypoint_error, axis=0)
    else:
        keypoint_error = np.empty([0, num_keypoints])
    return conf_matched, total_num_target, skeleton, keypoint_error


def compute_ap(conf_matched, total_num_target):
    sort_idx = np.argsort(conf_matched[:, 0])
    matched = np.flip(conf_matched[sort_idx, 1])
    matched_cum = np.cumsum(matched)
    precision = matched_cum / np.arange(1, len(matched_cum)+1)
    recall = matched_cum / total_num_target
    ap = 0
    prev_p = 0
    prev_r = 0
    for p, r in zip(precision, recall):
        if p < prev_p:
            ap += prev_p * (r - prev_r)
            prev_r = r
        prev_p = p
    return ap, (precision, recall)


def compute_pck(keypoint_error: np.ndarray, threshold_list: List[float]):
    sorted_err = np.sort(keypoint_error, axis=0).transpose()
    num_sample = np.sum(~np.isnan(sorted_err), axis=1, keepdims=True)
    prob = np.arange(1, sorted_err.shape[1] + 1) / (num_sample + 1e-20)
    prob[prob > 1.0] = np.nan
    per_keypoint_cdf = np.stack((sorted_err, prob), axis=2)
    sorted_err = np.sort(keypoint_error, axis=None)
    sorted_err = sorted_err[~np.isnan(sorted_err)]
    num_sample = sorted_err.shape[0]
    prob = np.arange(1, num_sample+1) / (num_sample + 1e-20)
    keypoint_cdf = np.stack((sorted_err, prob), axis=1)
    per_keypoint_pck = {}
    keypoint_pck = {}
    for threshold in threshold_list:
        cond = np.sum(np.float32(keypoint_error < threshold), axis=0)
        num_sample = np.sum(~np.isnan(keypoint_error), axis=0)
        per_keypoint_pck[threshold] = cond / (num_sample + 1e-20)
        keypoint_pck[threshold] = (cond.sum() / (num_sample.sum() + 1e-20)).item()
    return keypoint_pck, per_keypoint_pck, keypoint_cdf, per_keypoint_cdf

