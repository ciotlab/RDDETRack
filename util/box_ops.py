import torch


def box_3d_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def box_3d_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)


def box_3d_iou(boxes1, boxes2):
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    maxmin = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    minmax = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])

    wlh = (minmax - maxmin).clamp(min=0)
    inter = wlh[:, :, 0] * wlh[:, :, 1] * wlh[:, :, 2]

    union = volume1[:, None] + volume2 - inter

    iou = inter / union
    return iou, union


def generalized_box_3d_iou(boxes1, boxes2):
    iou, union = box_3d_iou(boxes1, boxes2)

    minmin = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    maxmax = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wlh = (maxmax - minmin).clamp(min=0)
    volume = wlh[:, :, 0] * wlh[:, :, 1] * wlh[:, :, 2]

    return iou - (volume - union) / volume
