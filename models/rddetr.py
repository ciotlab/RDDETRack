import numpy as np
from ray.data import from_numpy
from tqdm import tqdm
import torch
from torch import nn
from models.misc_model import positional_encoding_sine, MLP
from models.transformer import Transformer
import logging
from data_processing.dataset import get_dataset_and_dataloader
from .matcher import HungarianMatcher


class RDDETR(nn.Module):
    def __init__(self, num_stacked_seqs, num_keypoints, input_mlp_num_layers, box_mlp_num_layers, d_model, num_queries,
                 n_head, num_layers, dim_feedforward, dropout, activation, matcher, pre_ln=True):
        super(RDDETR, self).__init__()
        self.num_stacked_seqs = num_stacked_seqs
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.num_queries = num_queries
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.input_model = MLP(input_dim=5, hidden_dim=self.d_model, output_dim=self.d_model,
                               num_layers=input_mlp_num_layers)
        self.transformer = Transformer(d_model=d_model, n_head=n_head, num_layers=num_layers,
                                       dim_feedforward=dim_feedforward, dropout=dropout,
                                       activation=activation, return_intermediate=True, pre_ln=pre_ln)
        self.box_head = MLP(input_dim=d_model, hidden_dim=d_model, output_dim=6,
                            num_layers=box_mlp_num_layers)
        self.keypoint_dist_head = nn.Linear(d_model, num_keypoints * 3)
        self.objectness_head = nn.Linear(d_model, 1)
        query_pos = positional_encoding_sine(num_embedding=num_queries, d_model=d_model,
                                             max_num_embedding=num_queries, normalize=False, scale=None)
        self.register_buffer('query_pos', query_pos)
        self.query = nn.Embedding(num_queries, d_model)
        self.matcher = matcher

        #query_zero = torch.zeros(num_queries, d_model)
        #self.register_buffer('query_zero', query_zero)

    def add_track_query_to_target(self, targets, prev_target, prev_indices, prev_out):
        device = prev_out['pred_boxes'].device

        max_prev_target_ind = max(len(prev_ind[1]) for prev_ind in prev_indices)

        track_query_match_ids = []
        track_query_hs_embed = []
        track_query_mask = []

        for i, prev_ind in enumerate(prev_indices):
            prev_out_ind, prev_target_ind = prev_ind

            hs_with_padding = torch.zeros((max_prev_target_ind, self.d_model)).to(device)
            hs = prev_out['hs_embed'][i, prev_out_ind]
            hs_with_padding[:hs.shape[0]] = hs
            prev_ids = np.array(prev_target['ids'][i])[prev_target_ind]

            target_ids = torch.tensor(targets['ids'][i])
            prev_ids = torch.tensor(prev_ids)
            hs_ind = target_ids.unsqueeze(dim=1).eq(prev_ids).nonzero()[:,1]
            ids_ind = prev_ids.unsqueeze(dim=1).eq(target_ids).nonzero()[:,1]

            track_query_match_ids.append((hs_ind, ids_ind))
            track_query_hs_embed.append(hs_with_padding)

            query_mask = torch.ones(max_prev_target_ind, dtype=bool)
            query_mask[:max_prev_target_ind] = False
            query_mask = torch.cat((query_mask, torch.zeros(self.num_queries, dtype=bool)), axis=0)
            track_query_mask.append(query_mask)

        targets['track_query_match_ids'] = track_query_match_ids
        targets['track_query_hs_embed'] = torch.stack(track_query_hs_embed).to(device)
        targets['track_query_mask'] = torch.stack(track_query_mask).to(device)

    def forward(self, point_cloud, point_cloud_padding_mask, boxes, keypoints, id, num_frames=1):
        batch_size = point_cloud.shape[0] // num_frames
        device = point_cloud.device
        prediction_list = []
        target_list = []
        frame_filter = [False] * num_frames
        frame_filter[0] = True
        frame_filter = frame_filter * batch_size
        for i in range(num_frames):
            point_cloud_per_frame = point_cloud[frame_filter]
            point_cloud_padding_mask_per_frame = point_cloud_padding_mask[frame_filter]
            boxes_per_frame = [b for b, m in zip(boxes, frame_filter) if m]
            keypoints_per_frame = [k for k, m in zip(keypoints, frame_filter) if m]
            ids_per_frame = [i for i, m in zip(id, frame_filter) if m]
            targets = {'boxes': boxes_per_frame, 'keypoints': keypoints_per_frame, 'ids': ids_per_frame}

            query = self.query.weight
            query_pos = self.query_pos

            if i > 0:
                self.add_track_query_to_target(targets, prev_targets, indices, prediction)
            x = self.input_model(point_cloud_per_frame)  # batch, token, data
            x, attn, intermediate_output, intermediate_attn = self.transformer(query=query, source=x,
                                                                               query_pos=query_pos, source_pos=None,
                                                                               key_padding_mask=point_cloud_padding_mask_per_frame,
                                                                               targets=targets)
            output_3d_box = self.box_head(x).reshape(-1, x.shape[1], 2, 3).sigmoid()
            output_keypoint_dist = self.keypoint_dist_head(x).reshape(-1, x.shape[1], self.num_keypoints, 3).sigmoid()
            output_confidence_logit = self.objectness_head(x)
            output_confidence = output_confidence_logit.sigmoid()
            prediction = {'pred_keypoints': output_keypoint_dist, 'pred_boxes': output_3d_box,
                          'attention_map': intermediate_attn + [attn], 'pred_confidence_logit': output_confidence_logit,
                          'pred_confidence': output_confidence, 'hs_embed': x}
            prediction_list.append(prediction)
            target_list.append(targets)
            indices = self.matcher(prediction, targets)
            prev_targets = targets
            frame_filter.insert(0, False)
            frame_filter.pop()

        return prediction_list, target_list


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    num_stacked_frames = 4
    num_keypoints = 18
    input_mlp_num_layers = 0
    box_mlp_num_layers = 2
    d_model = 128
    num_queries = 200
    n_head = 8
    num_layers = 3
    dim_feedforward = 2048
    dropout = 0.1
    activation = 'gelu'
    pre_ln = True
    device = 'cuda:0'
    batch_size = 32
    num_dataset_workers = 4
    rddetr = RDDETR(num_stacked_frames, num_keypoints, input_mlp_num_layers, box_mlp_num_layers, d_model,
                    num_queries, n_head, num_layers, dim_feedforward,
                    dropout, activation, pre_ln).to(device)
    dataloader, dataset = get_dataset_and_dataloader(name='train', num_stacked_frames=num_stacked_frames, max_num_points=2000,
                                                     area_min=[0.0, -12.0, -2.0], area_size=[9.0, 24.0, 4.0],
                                                     box_margin=[0.2, 0.2, 0.2],
                                                     min_max_velocity=[-2.0, 2.0], min_max_power_db=[-50.0, 50.0],
                                                     camera_displacement_y=-0.1,
                                                     batch_size=16, num_workers=0)
    logging.info(f'Parameters: {sum(p.numel() for p in rddetr.parameters() if p.requires_grad)}')
    iter_per_epoch = int(len(dataset) // batch_size)
    pbar = tqdm(enumerate(dataloader), total=iter_per_epoch, desc="Testing")
    for iter, data in pbar:
        point_cloud = torch.tensor(data['point_cloud']).to(device)
        point_cloud_padding_mask = torch.tensor(data['point_cloud_padding_mask']).to(device)
        prediction = rddetr(point_cloud, point_cloud_padding_mask)






