import torch
from torch import nn
import torch.nn.functional as F
from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, generalized_box_3d_iou
import logging
from data_processing.dataset import get_dataset_and_dataloader
from models.rddetr import RDDETR
from models.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(self, num_frames, empty_weight, matcher_weights=None):
        super().__init__()
        self.register_buffer('empty_weight', torch.tensor(empty_weight))
        self.num_frames = num_frames
        if matcher_weights:
            cost_boxes = matcher_weights['cost_boxes']
            cost_keypoint = matcher_weights['cost_keypoint']
            cost_giou = matcher_weights['cost_giou']
            cost_obj = matcher_weights['cost_obj']
            self.matcher = HungarianMatcher(cost_boxes, cost_keypoint, cost_giou, cost_obj,
                                            iou_thresh=None)
        else:
            self.matcher = HungarianMatcher(iou_thresh=None)

    def get_losses(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)  # matched index

        out_keypoints = outputs['pred_keypoints'][idx]
        out_boxes = outputs['pred_boxes'][idx]
        out_confidence = outputs['pred_confidence_logit'].squeeze(-1)

        target_keypoints = torch.cat([torch.tensor(t).float().to(out_keypoints.device)[i]
                                      for t, (_, i) in zip(targets['keypoints'], indices)], dim=0)
        target_boxes = torch.cat([torch.tensor(t).float().to(out_boxes.device)[i]
                                  for t, (_, i) in zip(targets['boxes'], indices)], dim=0)
        target_confidence = torch.zeros_like(out_confidence)
        target_confidence[idx] = 1

        loss_boxes = F.l1_loss(out_boxes, target_boxes, reduction='none')
        loss_keypoints = torch.nan_to_num(F.l1_loss(out_keypoints, target_keypoints, reduction='none'), nan=0.0)
        loss_objectness = F.binary_cross_entropy_with_logits(out_confidence, target_confidence, pos_weight=self.empty_weight,
                                                             reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_3d_iou(box_3d_cxcyczwhd_to_xyzxyz(out_boxes.flatten(1)),
                                                          box_3d_cxcyczwhd_to_xyzxyz(target_boxes.flatten(1))))

        losses = {}
        losses['loss_keypoints'] = loss_keypoints.sum() / loss_keypoints.shape[0]
        losses['loss_boxes'] = loss_boxes.sum() / loss_boxes.shape[0]
        losses['loss_object'] = loss_objectness.mean()
        losses['loss_giou'] = loss_giou.sum() / loss_giou.shape[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        device = outputs[-1]['pred_boxes'].device
        losses = {'loss_keypoints':torch.tensor(0.0).to(device), 'loss_boxes':torch.tensor(0.0).to(device),
                  'loss_object':torch.tensor(0.0).to(device), 'loss_giou':torch.tensor(0.0).to(device)}
        for output, target in zip(outputs, targets):
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = target['indices']

            # Compute loss
            loss = self.get_losses(output, target, indices)
            losses['loss_keypoints'] += loss['loss_keypoints']
            losses['loss_boxes'] += loss['loss_boxes']
            losses['loss_object'] += loss['loss_object']
            losses['loss_giou'] += loss['loss_giou']
        return losses


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    num_stacked_seqs = 4
    d_model = 128
    num_queries = 200
    n_head = 8
    num_layers = 3
    dim_feedforward = 2048
    dropout = 0.1
    activation = 'gelu'
    device = 'cuda:0'
    batch_size = 32
    num_dataset_workers = 4
    rddetr = RDDETR(num_stacked_seqs, d_model, num_queries, n_head, num_layers, dim_feedforward,
                    dropout, activation).to(device)
    empty_weight = 300
    criterion = SetCriterion(empty_weight).to(device)
    dataloader, dataset = get_dataset_and_dataloader(batch_size=batch_size, num_workers=num_dataset_workers,
                                                     num_stacked_seqs=num_stacked_seqs, mode='test')
    data_it = iter(dataloader)
    radar, label = next(data_it)
    prediction = rddetr(radar.to(device))
    loss = criterion(prediction, label)
    pass