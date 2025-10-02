import yaml
from torch.utils.data import Dataset, DataLoader
from asyncio import Queue
from pathlib import Path
import numpy as np
import time
import os
import glob
import dill


class RadarDataset(Dataset):
    def __init__(self, name, num_stacked_frames, num_tracking_frames, max_num_points, area_min, area_size, box_margin,
                 min_max_velocity, min_max_power_db, camera_displacement_y):
        self.directory = Path(__file__).parents[1].resolve() / 'data' / name
        self.num_stacked_frames = num_stacked_frames
        self.num_tracking_frames = num_tracking_frames
        self.max_num_points = max_num_points
        self.area_min = np.array(area_min)
        self.area_size = np.array(area_size)
        self.box_margin = np.array(box_margin)
        self.min_max_velocity = np.array(min_max_velocity)
        self.min_max_power_db = np.array(min_max_power_db)
        self.camera_displacement = np.array([0.0, camera_displacement_y, 0.0])
        self.meta_data_file = self.directory / 'meta_data.dill'
        self.meta_data = None
        if not self.meta_data_file.exists():
            self.create_meta_data_file()
        with open(self.meta_data_file, 'rb') as f:
            self.meta_data = dill.load(f)
        self.num_sessions = len(self.meta_data)
        self.length = 0
        self.session_start_idx = []
        start_idx = 0
        for sess in self.meta_data:
            length = sess['length'] - (num_stacked_frames - 1) - (num_tracking_frames - 1)
            self.length += length
            self.session_start_idx.append(start_idx)
            start_idx += length
        self.session_start_idx.append(start_idx)
        self.data = []
        for sess in self.meta_data:
            file = Path(sess['file'])
            with open(file, 'rb') as f:
                self.data.append(dill.load(f))


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i, sess_idx = 0, 0
        for sess_idx in range(self.num_sessions):
            start_idx, end_idx = self.session_start_idx[sess_idx], self.session_start_idx[sess_idx + 1]
            if start_idx <= idx < end_idx:
                i = idx - start_idx
                break
        data = self.data[sess_idx]
        frame_data = []
        for j in range(self.num_tracking_frames):
            frame_idx = i + j
            point_cloud = []
            for n in range(frame_idx, frame_idx + self.num_stacked_frames):
                point_cloud.append(data[n]['point_cloud'])
            point_cloud = np.concatenate(point_cloud, axis=0)  # points, xyzvp
            pc_len = point_cloud.shape[0]
            point_cloud_padding_mask = np.ones((self.max_num_points,)).astype(np.bool)  # points
            point_cloud_padding_mask[:pc_len] = False
            if pc_len > self.max_num_points:
                point_cloud = point_cloud[:self.max_num_points, :]
            else:
                pad_width = self.max_num_points - pc_len
                point_cloud = np.pad(point_cloud, pad_width=((0, pad_width), (0, 0)), mode='constant')
            keypoint = data[frame_idx + self.num_stacked_frames - 1]['keypoint']  # person, keypoints, xyz
            box = data[frame_idx + self.num_stacked_frames - 1]['box']  # person, center/size, xyz
            id = data[frame_idx + self.num_stacked_frames - 1]['id']
            tmp = {'point_cloud': point_cloud, 'point_cloud_padding_mask': point_cloud_padding_mask,
                   'keypoint': keypoint, 'box': box, 'id': id}
            frame_data.append(tmp)
        return frame_data

    def normalization(self, point_cloud, keypoint):
        # point_cloud (points, xyzvp), keypoint (person, keypoints, xyz)
        # point cloud normalization
        filter = np.ones((point_cloud.shape[0],)).astype(np.bool)
        pc_xyz = point_cloud[:, :3]
        norm_pc_xyz = (pc_xyz - self.area_min) / self.area_size
        filter = np.logical_and(filter, np.all(np.logical_and(norm_pc_xyz >= 0.0, norm_pc_xyz <= 1.0), axis=1))
        pc_v = point_cloud[:, 3:4]
        norm_pc_v = (pc_v - self.min_max_velocity[0]) / (self.min_max_velocity[1] - self.min_max_velocity[0])
        norm_pc_v = np.clip(norm_pc_v, a_min=0.0, a_max=1.0)
        pc_p = point_cloud[:, 4:5]
        pc_p_db = 10.0 * np.log10(pc_p + 1e-12)
        norm_pc_p_db = (pc_p_db - self.min_max_power_db[0]) / (self.min_max_power_db[1] - self.min_max_power_db[0])
        filter = np.logical_and(filter, (norm_pc_p_db >= 0.0)[:, 0])
        norm_pc_p_db = np.clip(norm_pc_p_db, a_min=0.0, a_max=1.0)
        norm_point_cloud = np.concatenate((norm_pc_xyz, norm_pc_v, norm_pc_p_db), axis=1)  # points, xyzvp
        norm_point_cloud = norm_point_cloud[filter, :]
        # get normalized box
        keypoint = keypoint + self.camera_displacement
        box_min = np.nanmin(keypoint, axis=1, keepdims=True, initial=np.inf) - self.box_margin/2.0  # person, 1, xyz
        box_max = np.nanmax(keypoint, axis=1, keepdims=True, initial=-np.inf) + self.box_margin/2.0  # person, 1, xyz
        box_center = (box_min + box_max) / 2
        box_size = box_max - box_min
        norm_box_center = (box_center - self.area_min) / self.area_size
        norm_box_size = box_size / self.area_size
        norm_box = np.concatenate((norm_box_center, norm_box_size), axis=1)  # person, center/size, xyz
        # keypoint normalization
        norm_keypoint = (keypoint - box_min) / box_size  # person, keypoints, xyz
        return norm_box, norm_point_cloud, norm_keypoint

    def create_meta_data_file(self):
        meta_data = []
        sessions = [x for x in self.directory.iterdir() if x.is_dir()]
        for sess in sessions:
            session_data = []
            files = [x for x in sess.iterdir() if x.is_file()]
            for file in files:
                with open(file, 'rb') as f:
                    file_data = dill.load(f)
                    for d in file_data:
                        if d['radar'] is None or d['camera'] is None:
                            continue
                        frame_idx = int(d['radar']['frame_idx'])
                        point_cloud = d['radar']['point_cloud']
                        keypoint = d['camera']['keypoint_3d']
                        id = d['camera']['id']
                        norm_box, norm_point_cloud, norm_keypoint = self.normalization(point_cloud, keypoint)
                        session_data.append({'frame_idx': frame_idx, 'point_cloud': norm_point_cloud.astype(np.float32),
                                             'box': norm_box.astype(np.float32),
                                             'keypoint': norm_keypoint.astype(np.float32),
                                             'id': id},)
            session_data_file = self.directory / (sess.name + '_data.dill')
            with open(session_data_file, 'wb') as f:
                dill.dump(session_data, f)
            meta_data.append({'name': sess.name, 'length': len(session_data), 'file': str(session_data_file)})
        with open(self.meta_data_file, 'wb') as f:
            dill.dump(meta_data, f)


def collate_fn(data):
    data = list(map(list, zip(*data)))  # time, batch, dict
    point_cloud = np.stack([np.stack([bd['point_cloud'] for bd in td], axis=0) for td in data],
                           axis=0)  # time, batch, points, xyzvp
    point_cloud_padding_mask = np.stack([np.stack([bd['point_cloud_padding_mask'] for bd in td], axis=0) for td in data],
                                        axis=0)  # time, batch, points
    keypoint = [[bd['keypoint'] for bd in td] for td in data]  # time, batch, (person, keypoints, xyz)
    box = [[bd['box'] for bd in td] for td in data]  # time, batch, (person, center/size, xyz)
    id = [[bd['id'] for bd in td] for td in data]  # time, batch, (person)
    return {'point_cloud': point_cloud, 'point_cloud_padding_mask': point_cloud_padding_mask,
            'keypoint': keypoint, 'box': box, 'id': id}


def get_dataset_and_dataloader(name, num_stacked_frames, num_tracking_frames, max_num_points, area_min, area_size,
                               box_margin, min_max_velocity, min_max_power_db, camera_displacement_y,
                               batch_size, num_workers=0):
    dataset = RadarDataset(name=name, num_stacked_frames=num_stacked_frames, num_tracking_frames= num_tracking_frames,
                           max_num_points=max_num_points, area_min=area_min, area_size=area_size, box_margin=box_margin,
                           min_max_velocity=min_max_velocity, min_max_power_db=min_max_power_db, camera_displacement_y=camera_displacement_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataloader, dataset


if __name__ == '__main__':
    dataloader, dataset = get_dataset_and_dataloader(name='train', num_stacked_frames=4, num_tracking_frames=16, max_num_points=2000,
                                                     area_min=[0.0, -12.0, -2.0], area_size=[9.0, 24.0, 4.0],
                                                     box_margin=[0.2, 0.2, 0.2],
                                                     min_max_velocity=[-2.0, 2.0], min_max_power_db=[-50.0, 50.0],
                                                     camera_displacement_y=-0.1,
                                                     batch_size=64, num_workers=0)
    #d = dataset.__getitem__(1024)
    for data in dataloader:
        pass

