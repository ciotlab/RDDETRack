import argparse
from pathlib import Path
import time
import json
import datetime
import wandb
import datetime
import torch
import yaml
import numpy as np
import random

import util.misc as utils
from engine import train_one_epoch, evaluate
from models.rddetr import RDDETR
from models.criterion import SetCriterion
from data_processing.dataset import get_dataset_and_dataloader
from transformers import get_cosine_schedule_with_warmup
from models.matcher import HungarianMatcher
import logging


def main(conf_file='config.yaml', device='cuda:0'):
    conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
    with open(conf_path) as f:
        conf = yaml.safe_load(f)

    if conf['wandb']:
        wandb.init(project='radar_detr_jy', config=conf)
        conf = wandb.config

    device = torch.device(device)
    seed = conf['training.seed'] + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Model and criterion
    matcher_weights = {'cost_boxes': conf['matcher.cost_boxes'], 'cost_keypoint': conf['matcher.cost_keypoint'],
                       'cost_giou': conf['matcher.cost_giou'], 'cost_obj': conf['matcher.cost_obj']}
    matcher = HungarianMatcher(matcher_weights['cost_boxes'], matcher_weights['cost_keypoint'],
                               matcher_weights['cost_giou'], matcher_weights['cost_obj'])
    model = RDDETR(conf['model.num_stacked_frames'], conf['dataset.num_keypoints'], conf['model.input_mlp_num_layers'],
                   conf['model.box_mlp_num_layers'], conf['model.d_model'], conf['model.num_queries'],
                   conf['model.n_heads'], conf['model.num_layers'], conf['model.dim_feedforward'],
                   conf['model.trans_dropout'], conf['model.trans_activation'], matcher, conf['model.pre_ln']).to(device)
    criterion = SetCriterion(conf['loss.empty_weight'], matcher_weights).to(device)
    weight_dict = {'loss_keypoints': conf['loss.keypoints_coef'], 'loss_giou': conf['loss.giou_coef'],
                   'loss_boxes': conf['loss.boxes_coef'], 'loss_object': conf['loss.object_coef']}
    if conf['wandb']:
        wandb.watch(model)

    # Dataset
    train_dataloader, train_dataset = (
        get_dataset_and_dataloader(name='train', num_stacked_frames=conf['model.num_stacked_frames'],
                                   num_prev_frames=conf['dataset.num_prev_frames'],
                                   max_num_points=conf['model.max_num_points'],
                                   area_min=conf['dataset.area_min'],
                                   area_size=conf['dataset.area_size'],
                                   box_margin=conf['dataset.box_margin'],
                                   min_max_velocity=conf['dataset.min_max_velocity'],
                                   min_max_power_db=conf['dataset.min_max_power_db'],
                                   camera_displacement_y=conf['dataset.camera_displacement_y'],
                                   batch_size=conf['training.batch_size'],
                                   num_workers=conf['training.num_dataset_workers']))
    test_dataloader, test_dataset = (
        get_dataset_and_dataloader(name='test', num_stacked_frames=conf['model.num_stacked_frames'],
                                   num_prev_frames=conf['dataset.num_prev_frames'],
                                   max_num_points=conf['model.max_num_points'],
                                   area_min=conf['dataset.area_min'],
                                   area_size=conf['dataset.area_size'],
                                   box_margin=conf['dataset.box_margin'],
                                   min_max_velocity=conf['dataset.min_max_velocity'],
                                   min_max_power_db=conf['dataset.min_max_power_db'],
                                   camera_displacement_y=conf['dataset.camera_displacement_y'],
                                   batch_size=conf['training.batch_size'],
                                   num_workers=conf['training.num_dataset_workers']))


    # Optimizer and LR scheduler
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=conf['training.lr'], weight_decay=conf['training.weight_decay'])
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf['training.lr_drop'], gamma=0.1)
    num_steps_per_epoch = len(train_dataset) // conf['training.batch_size']
    num_total_training_steps = num_steps_per_epoch * conf['training.epochs']
    num_warmup_steps = int(num_total_training_steps * conf['training.warmup_proportion'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_total_training_steps,
    )

    logging.info("Start training")
    start_time = time.time()
    for epoch in range(conf['training.epochs']):
        train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=train_dataloader,
                                      weight_dict=weight_dict, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                      device=device, epoch=epoch, max_norm=conf['training.clip_max_norm'],
                                      use_wandb=conf['wandb'])
        val_stats = evaluate(model=model, data_loader=test_dataloader,
                             area_min=conf['dataset.area_min'], area_size=conf['dataset.area_size'],
                             num_keypoints=conf['dataset.num_keypoints'],
                             device=device, keypoint_thresh_list=conf['validation.keypoint_thresh_list'],
                             nms_iou_thresh=conf['validation.nms_iou_thresh'],
                             matching_iou_thresh=conf['validation.matching_iou_thresh'],
                             conf_thresh=conf['validation.conf_thresh'])
        ap = val_stats['AP']
        pck = val_stats['keypoint_pck']
        logging.info(f'AP: {ap}')
        logging.info(f'PCK: {pck}')
        if conf['wandb']:
            pck_metric = pck[conf['wandb.keypoint_metric_thresh']]
            combined = ap + conf['wandb.keypoint_metric_weight'] * pck_metric
            wandb.log({'AP': ap, 'PCK': pck_metric, 'combined_metric': combined})
            pr_curve = val_stats['pr_curve']
            keypoint_cdf = val_stats['keypoint_cdf']
            pr_size = 10000
            pr_skip = int(pr_curve[0].shape[0] / pr_size)
            if pr_skip > 0:
                pr_curve = [[x, y] for (x, y) in zip(pr_curve[1][::pr_skip], pr_curve[0][::pr_skip])]
                pr_curve_table = wandb.Table(data=pr_curve, columns=["recall", "precision"])
                wandb.log({"pr_curve": wandb.plot.line(pr_curve_table, "recall", "precision", title="Precision-recall curve")})
            kp_size = 1000
            kp_skip = int(keypoint_cdf.shape[0] / kp_size)
            if kp_skip > 0:
                keypoint_cdf = [[x, y] for (x, y) in keypoint_cdf[::kp_skip, :]]
                keypoint_cdf_table = wandb.Table(data=keypoint_cdf, columns=["error_dist", "cdf"])
                wandb.log({"keypoint_cdf": wandb.plot.line(keypoint_cdf_table, "error_dist", "cdf", title="Keypoint error")})

    if conf['wandb']:
        output_file = Path(wandb.run.dir) / conf['training.model_file_name']
        torch.save(model, output_file)
    else:
        output_file = Path(__file__).parents[0].resolve() / 'saved_model' / conf['training.model_file_name']
        torch.save(model, output_file)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debugging
    main(conf_file='config.yaml', device='cuda:0')
