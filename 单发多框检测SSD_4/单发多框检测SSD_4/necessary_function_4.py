import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2   # 常见做法：中间通道数取一半
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut 需要匹配 in_channels → out_channels，且下采样 4 倍
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=4)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return y + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(self, class_num: int,
                 anchor_size_num: int,
                 ):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.MaxPool = nn.MaxPool2d(kernel_size = 3,stride = 1)
            #多尺度特征
        self.feature1 = nn.Sequential(Resnet_block(64, 256))
        self.feature2 = nn.Sequential(Resnet_block(256, 512))
        self.feature3 = nn.Sequential(Resnet_block(512, 512))

        self.class_num = class_num
        self.anchor_size_num = anchor_size_num
            #分类头1
        self.class_head1 = nn.Conv2d(256,
                                    out_channels = anchor_size_num * (class_num + 1),
                                    kernel_size = 3,
                                     padding = 1
                                    )
            #边框回归头1
        self.bbox_head1 = nn.Conv2d(256,
                                   out_channels = 4 * anchor_size_num,
                                   kernel_size = 3,
                                    padding=1
                                   )
            #分类头2
        self.class_head2 = nn.Conv2d(512,
                                    out_channels = anchor_size_num * (class_num + 1),
                                    kernel_size = 3,
                                     padding=1
                                    )
            #边框回归头2
        self.bbox_head2 = nn.Conv2d(512,
                                   out_channels = 4 * anchor_size_num,
                                   kernel_size = 3,
                                    padding=1
                                   )
            #分类头3
        self.class_head3 = nn.Conv2d(512,
                                    out_channels = anchor_size_num * (class_num + 1),
                                    kernel_size = 3,
                                     padding=1
                                    )
            #边框回归头3
        self.bbox_head3 = nn.Conv2d(512,
                                   out_channels = 4 * anchor_size_num,
                                   kernel_size = 3,
                                    padding=1
                                   )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.MaxPool(x)
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
            #多尺度预测
        class_prediction1 = self.class_head1(feature1)
        bbox_prediction1 = self.bbox_head1(feature1)

        class_prediction2 = self.class_head2(feature2)
        bbox_prediction2 = self.bbox_head2(feature2)

        class_prediction3 = self.class_head3(feature3)
        bbox_prediction3 = self.bbox_head3(feature3)

            # 返回三个元组 class_prediction和 bbox_prediction 和 featuremap
        return (class_prediction1, class_prediction2, class_prediction3),(bbox_prediction1, bbox_prediction2, bbox_prediction3),(feature1, feature2, feature3)


    def prediction_flatten(self,
                           class_prediction: tuple,
                           bbox_prediction: tuple,):
        """
        将多尺度的分类和回归预测展平并拼接为 (batch, total_anchors, -1)
        参数:
            class_prediction: 元组 (cls1, cls2, cls3)，每个形状 [B, C, H, W]
            bbox_prediction: 元组 (bbox1, bbox2, bbox3)，每个形状 [B, 4*A, H, W]
        返回:
            flat_cls:  [B, total_anchors, class_num+1]
            flat_bbox: [B, total_anchors, 4]
        """
        flatten_class_prediction = []
        flatten_bbox_prediction = []

        for cls_p, bbox_p in zip(class_prediction, bbox_prediction):
            b = cls_p.shape[0]
                # cls_p: [B, A*(class_num+1), H, W] -> [B, H, W, A*(class_num+1)]
            cls_p = cls_p.permute(0, 2, 3, 1).reshape(b, -1, self.class_num + 1)
            flatten_class_prediction.append(cls_p)
                # bbox_p: [B, A*4, H, W] -> [B, H, W, A*4]
            bbox_p = bbox_p.permute(0, 2, 3, 1).reshape(b, -1, 4)
            flatten_bbox_prediction.append(bbox_p)

        flatten_class_prediction = torch.cat(flatten_class_prediction, dim = 1)
        flatten_bbox_prediction = torch.cat(flatten_bbox_prediction, dim = 1)

        return flatten_class_prediction, flatten_bbox_prediction


# t = torch.rand(8, 3, 255, 255)
# net = ResNet(class_num = 1,     # 生成多尺度featuremap
#              anchor_size_num = 2)
# cls_preds, bbox_preds = net(t)    # 得到两个元组
# flat_cls, flat_bbox = net.prediction_flatten(cls_preds, bbox_preds)
# print(flat_cls.shape, flat_bbox.shape)


# class Head(nn.Module):
#     def __init__(self, class_num: int = 1,
#                  anchor_size_num: int = 2):
#         super(Head, self).__init__()
#         self.class_num = class_num
#         self.anchor_size_num = anchor_size_num
#             #分类头
#         self.class_head = nn.Conv2d(256,
#                                     out_channels = anchor_size_num * (class_num + 1),
#                                     kernel_size = 3,
#                                     )
#             #边框回归头
#         self.bbox_head = nn.Conv2d(256,
#                                    out_channels = 4 * anchor_size_num,
#                                    kernel_size = 3,
#                                    )
#
#     def forward(self, x):
#         """
#         分类头和回归头的前向传播
#         :param x: featuremap
#         :return:
#         class_prediction: [b, anchor_size_num * w * h, class_num + 1]
#         bbox_prediction: [b, anchor_size_num * w * h, 4]
#         """
#         featuremap = x
#         class_num = self.class_num
#         anchor_size_num = self.anchor_size_num
#         class_prediction = self.class_head(featuremap)
#         bbox_prediction = self.bbox_head(featuremap)
#
#         b, c, w, h = class_prediction.shape
#         class_prediction = class_prediction.permute(0, 2, 3, 1).reshape(b, -1, class_num + 1)
#         b, c, w, h = bbox_prediction.shape
#         bbox_prediction = bbox_prediction.permute(0, 2, 3, 1).reshape(b, -1, 4)
#
#         return class_prediction, bbox_prediction


class anchor_generator:
    def __init__(self, featuremap: torch.Tensor,
                 anchor_size:list = None,
                 stride:int = 1,
                 ):
        if anchor_size is None:
            anchor_size = [32, 64]

        self.featuremap = featuremap
        self.anchor_size = anchor_size
        self.stride = stride

    def create(self):
        b, c, w, h = self.featuremap.shape
        anchors = []

        for i in range(w):
            for j in range(h):
                cx = (j + 0.5) * self.stride
                cy = (i + 0.5) * self.stride
                for k in self.anchor_size:
                    anchor_h = k
                    anchor_w = k
                    x1 = cx - anchor_w / 2
                    y1 = cy - anchor_h / 2
                    x2 = cx + anchor_w / 2
                    y2 = cy + anchor_h / 2
                    anchors.append([x1, y1, x2, y2])

        return torch.tensor(anchors, dtype = torch.float64)  # [anchors_num, 4]


def iou(boxes1:torch.Tensor,
        boxes2:torch.Tensor):
    """
    计算交并比
    :param boxes1: [anchors_num, 4]
    :param boxes2: [gt_num, 4]
    :return: [anchors_num, gt_num]
    """
    esp = 1e-7
    area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                          (boxes[:, 3] - boxes[:, 1]))
    boxes1_area = area(boxes1)
    boxes2_area = area(boxes2)

    inter_upperleft = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerright = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter = (inter_lowerright - inter_upperleft).clamp(min = esp)   # [anchor_num, gt_num, 2]

    inter_area = (inter[:, :, 0] * inter[:, :, 1]).clamp(min = esp)  #[num_anchors, num_gt]
    union_area = boxes1_area[:, None] + boxes2_area - inter_area  #[num_anchors, num_gt]

    return inter_area / union_area  #[num_anchors, num_gt]


def offset(bbox:torch.Tensor,
            gt_box:torch.Tensor,
            match_gt_idx: torch.Tensor,
           ):
    esp = 1e-7
    gt_boxes = gt_box[match_gt_idx]

    bx = (bbox[:, 0] + bbox[:, 2]) * 0.5
    by = (bbox[:, 1] + bbox[:, 3]) * 0.5
    bw = (bbox[:, 2] - bbox[:, 0]).clamp(min = esp)
    bh = (bbox[:, 3] - bbox[:, 1]).clamp(min = esp)

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min = esp)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min = esp)

    dx = (gx - bx) / bw
    dy = (gy - by) / bh
    dw = torch.log(gw / (bw + esp))
    dh = torch.log(gh / (bh + esp))

    return torch.stack([dx, dy, dw, dh], dim = 1)


def match(bbox: torch.Tensor,
          gt_box: torch.Tensor,
          gt_labels: torch.Tensor,
          pos_thresh = 0.5, neg_thresh = 0.4
          ):
    anchor_num, gt_num = bbox.shape[0], gt_box.shape[0]
    jaccard = iou(bbox, gt_box)

    max_gt_iou, max_gt_idx = torch.max(jaccard, dim = 1)
    max_anchor_iou, max_anchor_idx = torch.max(jaccard, dim = 0)
        #强制给每个gt 匹配一个anchor
    match_gt_idx = max_gt_idx.clone()
    match_gt_idx[max_anchor_idx] = torch.arange(gt_num)
        #匹配正负样本
    positive_sample = max_gt_iou >= pos_thresh
    negative_sample = max_gt_iou < neg_thresh
        #强制给每个anchor 匹配一个gt
    positive_sample[max_anchor_idx] = True
    negative_sample[max_anchor_idx] = False

    cls_targets = torch.full((anchor_num,), -1, dtype = torch.long, device = bbox.device)
    cls_targets[negative_sample] = 0
    cls_targets[positive_sample] = gt_labels[match_gt_idx[positive_sample]]

    return cls_targets, positive_sample, negative_sample, match_gt_idx
