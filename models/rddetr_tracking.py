import numpy as np
import torch
from torch import nn
from models.rddetr import RDDETR
from models.matcher import HungarianMatcher

class RDDETRTracking(nn.Module):
    def __init__(self, num_stacked_seqs, num_keypoints, input_mlp_num_layers, box_mlp_num_layers, d_model, num_queries,
                 n_head, num_layers, dim_feedforward, dropout, activation, matcher: HungarianMatcher, pre_ln=True):
        super().__init__()
        # Backbone: original single-frame model
        self.backbone = RDDETR(num_stacked_seqs, num_keypoints, input_mlp_num_layers, box_mlp_num_layers,
                                   d_model, num_queries, n_head, num_layers, dim_feedforward,
                                   dropout, activation, pre_ln)
        # Matcher for per-frame assignment (used between frames)
        self.matcher = matcher

        # Convenience attributes (mirror backbone so downstream code can introspect)
        self.num_stacked_seqs = num_stacked_seqs
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.num_queries = num_queries
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

    def track_query_generation(self, targets, prev_target, prev_out):
        device = prev_out['pred_boxes'].device
        prev_indices = prev_target['indices']

        max_prev_target_ind = max(len(prev_ind[1]) for prev_ind in prev_indices)

        track_query = {}

        track_query_match_matrices = []
        track_query_hs_embed = []
        track_query_mask = []

        B = len(prev_indices)
        for b in range(B):
            prev_out_ind, prev_tgt_ind = prev_indices[b]

            hs_with_padding = torch.zeros((max_prev_target_ind, self.d_model), device=device)
            if prev_out_ind.numel() > 0:
                hs = prev_out['hs_embed'][b, prev_out_ind]
                hs_with_padding[:hs.shape[0]] = hs

            prev_ids_arr = np.array(prev_target['ids'][b])
            if prev_tgt_ind.numel() > 0:
                prev_ids = torch.as_tensor(prev_ids_arr[prev_tgt_ind.cpu().numpy()], dtype=torch.long, device=device)
            else:
                prev_ids = torch.zeros((0,), dtype=torch.long, device=device)
            cur_ids = torch.as_tensor(targets['ids'][b], dtype=torch.long, device=device)

            match_matrix = prev_ids.unsqueeze(dim=1).eq(cur_ids).nonzero()

            query_mask = torch.ones(max_prev_target_ind, dtype=torch.bool, device=device)
            if prev_out_ind.numel() > 0:
                query_mask[:len(prev_out_ind)] = False
            query_mask = torch.cat((query_mask, torch.zeros(self.num_queries, dtype=torch.bool, device=device)),axis=0)

            track_query_match_matrices.append(match_matrix)
            track_query_hs_embed.append(hs_with_padding)
            track_query_mask.append(query_mask)

        track_query['track_query_match_matrix'] = track_query_match_matrices
        track_query['track_query_hs_embed'] = torch.stack(track_query_hs_embed, dim=0).to(device)
        track_query['track_query_mask'] = torch.stack(track_query_mask, dim=0).to(device)

        return track_query

    def forward(self, point_cloud, point_cloud_padding_mask, boxes, keypoints, id, num_tracking_frames=1):
        prediction_list = []
        target_list = []

        prev_targets = None
        prev_out = None
        track_query = None

        for t in range(num_tracking_frames):
            point_cloud_t = point_cloud[t]
            pc_mask_t = point_cloud_padding_mask[t]
            boxes_t = boxes[t]
            keypoints_t = keypoints[t]
            ids_t = id[t]

            targets = {'boxes': boxes_t, 'keypoints': keypoints_t, 'ids': ids_t, 'indices': None}

            if t > 0 and prev_targets['indices'] is not None:
                track_query = self.track_query_generation(targets, prev_targets, prev_out)

            out = self.backbone(point_cloud_t, pc_mask_t, track_query=track_query)

            prev_indices = self.matcher(out, targets, track_query)
            targets['indices'] = prev_indices
            prev_targets = targets
            prev_out = out

            prediction_list.append(out)
            target_list.append(targets)

        return prediction_list, target_list
