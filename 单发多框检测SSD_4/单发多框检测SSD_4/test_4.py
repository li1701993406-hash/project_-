import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms

from get_data_4 import get_device
from necessary_function_4 import ResNet, anchor_generator


# 1. 逆偏移量函数：把网络预测的 dx, dy, dw, dh 还原为真实的坐标
def offset_inverse(anchors, bbox_preds, esp=1e-6):
    """
    anchors: [N, 4] (x1, y1, x2, y2)
    bbox_preds: [N, 4] (dx, dy, dw, dh) 网络预测值
    """
    bx = (anchors[:, 0] + anchors[:, 2]) * 0.5
    by = (anchors[:, 1] + anchors[:, 3]) * 0.5
    bw = (anchors[:, 2] - anchors[:, 0]).clamp(min=esp)
    bh = (anchors[:, 3] - anchors[:, 1]).clamp(min=esp)

    dx = bbox_preds[:, 0]
    dy = bbox_preds[:, 1]
    dw = bbox_preds[:, 2]
    dh = bbox_preds[:, 3]

    # 根据训练时的 encoder 公式反向推导
    gx = dx * bw + bx
    gy = dy * bh + by
    gw = torch.exp(dw) * bw
    gh = torch.exp(dh) * bh

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    return torch.stack([x1, y1, x2, y2], dim=1)


def predict(image_path, model_path, score_thresh=0.5, nms_thresh=0.3):
    device = get_device()

    # 1. 加载训练好的模型参数
    net = ResNet(class_num=1, anchor_size_num=2).to(device)
    # 注意：确保你在 train_4.py 训练结束时用 torch.save(net.state_dict(), 'ssd_banana.pth') 保存了权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 切换到测试模式（会关闭 BatchNorm 的动态更新）

    # 2. 图像预处理
    original_img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(original_img).unsqueeze(0).to(device)  # 增加 batch 维度 [1, 3, H, W]

    with torch.no_grad():  # 推理阶段不需要计算梯度
        # 前向传播
        cls_preds, bbox_preds, features = net(img_tensor)
        cls_prediction, bbox_prediction = net.prediction_flatten(
            cls_preds, bbox_preds
        )

        # 因为 batch_size=1，直接去掉 batch 维度
        cls_prediction = cls_prediction.squeeze(0)  # [Num_Anchors, 2]
        bbox_prediction = bbox_prediction.squeeze(0)  # [Num_Anchors, 4]

        # 3. 生成和训练一模一样的 Anchors
        anchors = []
        for feature in features:
            anchor = (
                anchor_generator(featuremap=feature, anchor_size=[32, 64])
                .create()
                .to(device)
            )
            anchors.append(anchor)
        anchors = torch.cat(anchors, dim=0)  # [Num_Anchors, 4]

        # 4. 计算香蕉类别的置信度 (假设背景是0，香蕉是1)
        prob = F.softmax(cls_prediction, dim=-1)
        scores = prob[:, 1]  # 获取每个 Anchor 属于香蕉的概率

        # 5. 解码预测框
        decoded_boxes = offset_inverse(anchors, bbox_prediction)

        # 6. 阈值过滤：过滤掉置信度低于 score_thresh 的框
        keep_idx = scores > score_thresh
        filtered_boxes = decoded_boxes[keep_idx]
        filtered_scores = scores[keep_idx]

        if filtered_boxes.shape[0] == 0:
            print("未检测到任何香蕉！")
            return original_img

        # 7. NMS（非极大值抑制）：消除重叠的冗余框
        # torchvision 自带的 nms 运行在 GPU 上非常快
        nms_idx = torchvision.ops.nms(
            filtered_boxes, filtered_scores, iou_threshold=nms_thresh
        )

        final_boxes = filtered_boxes[nms_idx]
        final_scores = filtered_scores[nms_idx]

    # 8. 可视化画图
    draw = ImageDraw.Draw(original_img)
    for box, score in zip(final_boxes, final_scores):
        box = box.cpu().numpy()
        # 画出边界框
        draw.rectangle(
            [box[0], box[1], box[2], box[3]], outline="red", width=3
        )
        # 写上置信度
        draw.text((box[0], box[1] - 10), f"Banana: {score:.2f}", fill="red")

    original_img.show()  # 弹出窗口查看图片
    original_img.save("result.jpg")  # 保存图片
    return original_img


if __name__ == "__main__":
    # 使用示范
    predict(
        image_path=r"C:\Users\li170\Desktop\banana-detection\bananas_train\images\0.png",  # 换成你想测试的图片路径
        model_path="net.pth",  # 换成你训练保存的模型权重路径
        score_thresh=0.6,  # 置信度阈值，调高可以减少误报
        nms_thresh=0.3,  # NMS 重叠度阈值，调低可以减少重叠框
    )