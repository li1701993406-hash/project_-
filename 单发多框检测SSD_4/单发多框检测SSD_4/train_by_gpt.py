# from necessary_function_4 import *
# from get_data_4 import *
# import time
# import tkinter as tk
#
#
# device = get_device()
# train_dataloader = get_train_dataloader(
#     root_dir = r"C:\Users\li170\Desktop\banana-detection\bananas_train",
#     batch_size = 8,
#     shuffle = True,
#     num_workers = 6,
# )
#
# net = ResNet(class_num = 1,     # 生成多尺度featuremap
#              anchor_size_num = 2)
#
# cla_loss = nn.CrossEntropyLoss()
# bb_loss = nn.L1Loss()
# optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
#
#
# def train(num_epochs=20, dataloader=train_dataloader, device=device):
#     net.to(device)
#     net.train()
#
#     print('开始训练')
#     for ep in range(num_epochs):
#         start_time = time.time()
#
#         running_total_loss = 0.0
#         running_cls_loss = 0.0
#         running_bbox_loss = 0.0
#         count = 0
#
#         for images, labels in dataloader:
#             images = images.to(device)
#
#             cls_preds, bbox_preds, features = net(images)
#             cls_preds, bbox_preds = net.prediction_flatten(cls_preds, bbox_preds)
#
#             anchors = []
#             for feature in features:
#                 anchor = anchor_generator(
#                     featuremap=feature,
#                     anchor_size=[32, 64],
#                 ).crate().to(device)
#                 anchors.append(anchor)
#             anchors = torch.cat(anchors, dim=0)
#
#             batch_cls_targets = []
#             batch_bbox_targets = []
#             batch_positive_mask = []
#
#             for b in range(len(labels)):
#                 gt_boxes = labels[b]['boxes'].to(device)
#                 gt_labels = labels[b]['labels'].to(device)
#
#                 cls_targets, positive_sample, negative_sample, match_gt_idx = match(
#                     anchors, gt_boxes, gt_labels
#                 )
#                 bbox_targets = offset(anchors, gt_boxes, match_gt_idx)
#
#                 batch_cls_targets.append(cls_targets)
#                 batch_bbox_targets.append(bbox_targets)
#                 batch_positive_mask.append(positive_sample)
#
#             batch_cls_targets = torch.stack(batch_cls_targets, dim=0)
#             batch_bbox_targets = torch.stack(batch_bbox_targets, dim=0)
#             batch_positive_mask = torch.stack(batch_positive_mask, dim=0)
#
#             cls_loss = cla_loss(
#                 cls_preds.reshape(-1, cls_preds.shape[-1]),
#                 batch_cls_targets.reshape(-1)
#             )
#
#             bb_loss = bb_loss(
#                 bbox_preds[batch_positive_mask],
#                 batch_bbox_targets[batch_positive_mask]
#             )
#
#             loss = cls_loss + bb_loss
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_total_loss += loss.item()
#             running_cls_loss += cls_loss.item()
#             running_bbox_loss += bb_loss.item()
#             count += 1
#
#         spend_time = time.time() - start_time
#         print(
#             f'epoch {ep+1}: '
#             f'total_loss={running_total_loss/count:.4f}, '
#             f'cls_loss={running_cls_loss/count:.4f}, '
#             f'bbox_loss={running_bbox_loss/count:.4f}, '
#             f'time={spend_time:.2f}s'
#         )