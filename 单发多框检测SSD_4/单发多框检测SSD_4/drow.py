import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn.functional as F
import torchvision
import random
from PIL import Image, ImageTk, ImageDraw

# 导入你自己的模块
from get_data_4 import get_device, get_test_dataloader
from necessary_function_4 import ResNet, anchor_generator


# 复用之前的逆偏移量函数
def offset_inverse(anchors, bbox_preds, esp=1e-6):
    bx = (anchors[:, 0] + anchors[:, 2]) * 0.5
    by = (anchors[:, 1] + anchors[:, 3]) * 0.5
    bw = (anchors[:, 2] - anchors[:, 0]).clamp(min=esp)
    bh = (anchors[:, 3] - anchors[:, 1]).clamp(min=esp)

    dx, dy, dw, dh = bbox_preds[:, 0], bbox_preds[:, 1], bbox_preds[:, 2], bbox_preds[:, 3]

    gx = dx * bw + bx
    gy = dy * bh + by
    gw = torch.exp(dw) * bw
    gh = torch.exp(dh) * bh

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


class BananaDetectionGUI:
    def __init__(self, root, model_path='ssd_banana.pth'):
        self.root = root
        self.root.title("香蕉目标检测测试工具")
        self.root.geometry("900x550")

        # 初始化设备和模型
        self.device = get_device()
        self.net = ResNet(class_num=1, anchor_size_num=2).to(self.device)
        try:
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            print("成功加载模型权重！")
        except FileNotFoundError:
            messagebox.showerror("错误", f"找不到模型权重文件: {model_path}\n请先确保模型已训练并保存。")
        self.net.eval()

        # 加载测试集 DataLoader (这里 batch_size=1 方便抽取单张)
        # TODO: 请确认你的测试集路径是否正确
        self.test_dataloader = get_test_dataloader(
            root_dir=r"C:\Users\li170\Desktop\banana-detection\bananas_train",
            batch_size=1,
            shuffle=True,
            num_workers=0  # GUI应用中建议设为0避免多进程冲突
        )
        self.dataset = self.test_dataloader.dataset

        # 缓存当前抽取的图像变量
        self.current_pil_img = None
        self.predicted_pil_img = None

        # --- 界面布局 ---
        # 顶部按钮区域
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=15)

        self.btn_random = tk.Button(btn_frame, text="🎲 随机抽取测试集图片", font=("微软雅黑", 12),
                                    command=self.load_random_image)
        self.btn_random.pack(side=tk.LEFT, padx=50)

        self.btn_predict = tk.Button(btn_frame, text="🔍 绘制预测框", font=("微软雅黑", 12), state=tk.DISABLED,
                                     command=self.show_prediction)
        self.btn_predict.pack(side=tk.RIGHT, padx=50)

        # 中间显示图片的区域
        img_frame = tk.Frame(root)
        img_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)

        # 左边显示原图
        self.label_orig = tk.Label(img_frame, text="等待抽取图片...", font=("微软雅黑", 12), bg="gray90")
        self.label_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 右边显示预测图
        self.label_pred = tk.Label(img_frame, text="等待点击预测...", font=("微软雅黑", 12), bg="gray90")
        self.label_pred.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_random_image(self):
        # 1. 随机从数据集中选一个索引
        idx = random.randint(0, len(self.dataset) - 1)

        # 2. 获取原始 PIL 图片路径以便无损显示
        row = self.dataset.annotations.iloc[idx]
        img_name = row['img_name']
        img_path = os.path.join(self.dataset.images_dir, img_name)

        self.current_pil_img = Image.open(img_path).convert('RGB')

        # 3. 将图片格式化并展示在左边
        # 统一缩放到合适大小展示 (例如 350x350)
        img_resize = self.current_pil_img.resize((350, 350))
        img_tk = ImageTk.PhotoImage(img_resize)
        self.label_orig.config(image=img_tk, text="")
        self.label_orig.image = img_tk  # 保持引用防止被垃圾回收

        # 重置右边的预测框区域
        self.label_pred.config(image='', text="等待点击预测...")
        self.label_pred.image = None

        # 激活预测按钮
        self.btn_predict.config(state=tk.NORMAL)

        # 4. 后台直接用模型跑出这张图的 Tensor 结果缓存起来
        images_tensor, _ = self.dataset[idx]  # 拿到经过 transform 后的 tensor
        self.img_tensor_input = images_tensor.unsqueeze(0).to(self.device)

    def show_prediction(self, score_thresh=0.2, nms_thresh=0.3):
        if self.current_pil_img is None:
            return

        with torch.no_grad():
            # 前向传播
            cls_preds, bbox_preds, features = self.net(self.img_tensor_input)
            cls_prediction, bbox_prediction = self.net.prediction_flatten(cls_preds, bbox_preds)

            cls_prediction = cls_prediction.squeeze(0)
            bbox_prediction = bbox_prediction.squeeze(0)

            # 生成 Anchors (注意你的真实函数名，这里我用你在 necessary_function_4 里实际定义的)
            anchors = []
            for feature in features:
                # 兼容你代码里的 .crate() 拼写
                if hasattr(anchor_generator, 'crate'):
                    anchor = anchor_generator(featuremap=feature, anchor_size=[32, 64]).crate().to(self.device)
                else:
                    anchor = anchor_generator(featuremap=feature, anchor_size=[32, 64]).create().to(self.device)
                anchors.append(anchor)
            anchors = torch.cat(anchors, dim=0)

            # 置信度与解码
            prob = F.softmax(cls_prediction, dim=-1)
            scores = prob[:, 1]  # 属于香蕉的概率
            decoded_boxes = offset_inverse(anchors, bbox_prediction)

            # 阈值过滤
            # 在 keep_idx = scores > score_thresh 这行前面加上：
            print("=== 调试信息 ===")
            print("所有 Anchor 中最高的置信度:", torch.max(scores).item())
            print("置信度大于 0.1 的 Anchor 数量:", torch.sum(scores > 0.1).item())
            keep_idx = scores > score_thresh
            filtered_boxes = decoded_boxes[keep_idx]
            filtered_scores = scores[keep_idx]

            # 复制一份原图用来画框
            self.predicted_pil_img = self.current_pil_img.copy()
            draw = ImageDraw.Draw(self.predicted_pil_img)

            if filtered_boxes.shape[0] > 0:
                # NMS 抑制
                nms_idx = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=nms_thresh)
                final_boxes = filtered_boxes[nms_idx]
                final_scores = filtered_scores[nms_idx]

                # 开始画框
                for box, score in zip(final_boxes, final_scores):
                    box = box.cpu().numpy()
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)
                    draw.text((box[0], max(0, box[1] - 12)), f"Banana: {score:.2f}", fill="red")
            else:
                draw.text((10, 10), "No Banana Detected", fill="yellow")

        # 5. 更新右侧标签显示预测图
        pred_resize = self.predicted_pil_img.resize((350, 350))
        pred_tk = ImageTk.PhotoImage(pred_resize)
        self.label_pred.config(image=pred_tk, text="")
        self.label_pred.image = pred_tk


if __name__ == "__main__":
    import os

    root = tk.Tk()
    app = BananaDetectionGUI(root, model_path='net.pth')
    root.mainloop()