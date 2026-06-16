import torch
from necessary_function_4 import *
from get_data_4 import *
import time
import tkinter as tk



device = get_device()
train_dataloader = get_train_dataloader(
    root_dir = r"C:\Users\li170\Desktop\banana-detection\bananas_train",
    batch_size = 8,
    shuffle = True,
    num_workers = 6,
)

net = ResNet(class_num = 1,     # 生成多尺度featuremap
             anchor_size_num = 2).to(device)

cla_loss = nn.CrossEntropyLoss(
    ignore_index = -1,
)
bb_loss = nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)


def train(epoch = 20,
          dataloader = train_dataloader,
          device = device,
          ):
    net.to(device)
    net.train()
    print(f'device: {device}')

    print(f'开始训练')
    for epoch in range(epoch):
        start_time = time.time()

        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_bbox_loss = 0.0
        count = 0

        for images, labels in dataloader:
            images = images.to(device)

            cls_preds, bbox_preds, features = net(images)
            cls_prediction, bbox_prediction = net.prediction_flatten(cls_preds, bbox_preds)

                # 获得anchors
            anchors = []
            for feature in features:
                anchor = anchor_generator(featuremap = feature,
                                          anchor_size = [32, 64],
                                          ).create().to(device)
                anchors.append(anchor)
            anchors = torch.cat(anchors, dim = 0)

            batch_class_targets = []
            batch_bbox_targets = []
            batch_positive_sample = []

            for b in range(len(labels)):
                gt_boxes = labels[b]['boxes'].to(device)
                gt_labels = labels[b]['labels'].to(device)

                (cls_targets, positive_sample,
                 negative_sample, match_gt_idx) = match(anchors,
                                                        gt_boxes,
                                                        gt_labels,
                                                        )
                bbox_targets = offset(anchors, gt_boxes, match_gt_idx)

                batch_class_targets.append(cls_targets)
                batch_bbox_targets.append(bbox_targets)
                batch_positive_sample.append(positive_sample)

                # 拼出batch维度(其中 batch_positive_sample.shape[-1]
                # 为每个batch的总正样本数量)
            batch_class_targets = torch.stack(batch_class_targets, dim = 0).to(device)
            batch_bbox_targets = torch.stack(batch_bbox_targets, dim = 0).to(device)
            batch_positive_sample = torch.stack(batch_positive_sample, dim = 0).to(device)

                # 计算损失函数
            #分类损失
            class_loss = cla_loss(
                cls_prediction.reshape(-1, 2),
                batch_class_targets.reshape(-1),
            )
            #回归损失
            bbox_loss = bb_loss(
                bbox_prediction[batch_positive_sample],
                batch_bbox_targets[batch_positive_sample]
            )

            loss = class_loss + bbox_loss

                # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #指标
            running_total_loss += loss.item()
            running_cls_loss += class_loss.item()
            running_bbox_loss += bbox_loss.item()
            count += 1

        end_time = time.time()
        spend_time = end_time - start_time

                #统计数据
        print(
            f'epoch: {epoch + 1}: '
            f'total_loss: {running_total_loss/count:.4f}, '
            f'class_loss: {running_cls_loss/count:.4f}, '
            f'bbox_loss: {running_bbox_loss/count:.4f}, '
            f'time: {spend_time:.2f}s'
        )



if __name__ == '__main__':
    train(epoch = 20,
          )
    torch.save(net.state_dict(), 'net.pth')
    print(f'训练完成, 权重保存在: net.pth')
