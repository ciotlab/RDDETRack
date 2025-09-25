import numpy as np
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

        all_track_query_match_ids = []
        all_track_query_hs_embeds = []
        all_track_query_boxes = []
        all_track_queries_mask = []

        # Loop over the batch using the indices from the matcher
        for i, prev_ind in enumerate(prev_indices):
            prev_out_ind, prev_target_ind = prev_ind

            # This line assumes prev_target_ind correctly indexes into the batched tensor
            prev_track_ids = prev_target['id'][prev_target_ind[1]]

            # We need to get the target IDs for the current item in the batch
            # This assumes 'id' is a batched tensor in the targets dictionary
            current_target_ids = torch.from_numpy(targets['id'][i])

            target_ind_match_matrix = torch.from_numpy(prev_track_ids).unsqueeze(dim=1).eq(current_target_ids)
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            # Append the results for the current batch item to our lists
            targets['track_query_match_ids'] = target_ind_matched_idx

            track_queries_mask = torch.ones_like(target_ind_matching).to(device).bool()

            hs_embeds = prev_out['hs_embed'][i, prev_out_ind]
            targets['track_query_hs_embeds'] = hs_embeds

            boxes = prev_out['pred_boxes'][i, prev_out_ind].detach()
            targets['track_query_boxes'] = boxes

            mask = torch.cat([track_queries_mask, torch.tensor([False, ] * self.num_queries).to(device)]).bool()
            targets['track_queries_mask'] = mask

    def forward(self, data):
        prev_indices = None
        while True:
            point_cloud = data['input']['point_cloud']
            point_cloud_padding_mask = data['input']['point_cloud_padding_mask']
            targets = data['target']


            query_pos = self.query_pos
            query = self.query.weight
            if prev_indices != None:
                self.add_track_query_to_target(targets, prev_targets, prev_indices, prediction)
                if 'track_query_hs_embeds' in targets.keys():
                    track_query_hs_embeds = torch.tensor(targets['track_query_hs_embeds'])
                    num_track_queries = track_query_hs_embeds.shape[0]
                    track_query_pos = torch.zeros(num_track_queries, self.d_model).to(query_pos.device)
                    query_pos = torch.cat([track_query_pos, query_pos], dim=0)
                    query = torch.cat([track_query_hs_embeds, query], dim=0)

            if isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).to(query_pos.device)
            if isinstance(point_cloud_padding_mask, np.ndarray):
                point_cloud_padding_mask = torch.from_numpy(point_cloud_padding_mask).to(point_cloud.device)
            x = self.input_model(point_cloud)  # batch, token, data
            x, attn, intermediate_output, intermediate_attn = self.transformer(query=query, source=x,
                                                                               query_pos=query_pos, source_pos=None,
                                                                               key_padding_mask=point_cloud_padding_mask,
                                                                               targets=targets)
            output_3d_box = self.box_head(x).reshape(-1, query.shape[0], 2, 3).sigmoid()
            output_keypoint_dist = self.keypoint_dist_head(x).reshape(-1, query.shape[0], self.num_keypoints, 3).sigmoid()
            output_confidence_logit = self.objectness_head(x)
            output_confidence = output_confidence_logit.sigmoid()
            prediction = {'pred_keypoints': output_keypoint_dist, 'pred_boxes': output_3d_box,
                          'attention_map': intermediate_attn + [attn], 'pred_confidence_logit': output_confidence_logit,
                          'pred_confidence': output_confidence, 'hs_embed': x}

            if data.get('next_frame') is None:
                break
            else:
                prev_targets = targets
                data = data['next_frame']

            prev_indices = self.matcher(prediction, data)

        return prediction


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






