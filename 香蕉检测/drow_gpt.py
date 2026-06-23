import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageTk

from get_data_4 import get_test_dataloader, get_device
from necessary_function_4 import ResNet, anchor_generator


# =========================
# decode: offset -> box
# =========================
def decode_bbox(anchors: torch.Tensor, bbox_preds: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """anchors: [N,4] xyxy, bbox_preds: [N,4] dx dy dw dh"""
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=eps)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=eps)

    dx = bbox_preds[:, 0]
    dy = bbox_preds[:, 1]
    dw = bbox_preds[:, 2]
    dh = bbox_preds[:, 3]

    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw).clamp(max=1e4) * aw
    ph = torch.exp(dh).clamp(max=1e4) * ah

    x1 = px - pw * 0.5
    y1 = py - ph * 0.5
    x2 = px + pw * 0.5
    y2 = py + ph * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


# =========================
# drawing helpers
# =========================
def clamp_boxes_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, h - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, w - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, h - 1)
    return boxes


def draw_boxes_pil(img: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, color="red") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 12)), f"banana: {float(score):.2f}", fill=color)
    return out


def draw_anchors_pil(img: Image.Image, anchors: torch.Tensor, max_draw: int = 250, color="blue") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)

    if anchors.shape[0] == 0:
        return out

    # 均匀采样一些anchor，避免画面太乱
    num = min(max_draw, anchors.shape[0])
    idx = torch.linspace(0, anchors.shape[0] - 1, steps=num).long()
    sample = anchors[idx]

    for box in sample:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
    return out


# =========================
# main GUI
# =========================
class BananaSSDViewer:
    def __init__(self, root, dataloader, weights_path="net.pth", input_size=256):
        self.root = root
        self.root.title("香蕉 SSD 推理可视化 Demo")
        self.root.geometry("1280x760")

        self.device = get_device()
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.weights_path = weights_path
        self.input_size = input_size

        self.net = ResNet(class_num=1, anchor_size_num=2).to(self.device)
        self._load_weights()
        self.net.eval()

        self.current_idx = 0
        self.current_pil = None
        self.current_tensor = None
        self.current_pred_pil = None
        self.current_anchor_pil = None

        self.show_anchors = tk.BooleanVar(value=True)
        self.score_thresh = tk.DoubleVar(value=0.10)
        self.nms_thresh = tk.DoubleVar(value=0.30)

        self._build_ui()

    def _load_weights(self):
        if not os.path.exists(self.weights_path):
            messagebox.showerror("错误", f"找不到权重文件: {self.weights_path}")
            return
        state = torch.load(self.weights_path, map_location=self.device)
        self.net.load_state_dict(state)
        print(f"成功加载模型权重: {self.weights_path}")

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text=f"Device: {self.device}").pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="随机抽一张", command=self.load_random_image).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="重新加载权重", command=self.reload_weights).pack(side=tk.LEFT, padx=12)

        ttk.Checkbutton(top, text="显示锚框", variable=self.show_anchors, command=self.refresh_view).pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="score").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Scale(top, from_=0.01, to=0.95, variable=self.score_thresh, orient=tk.HORIZONTAL, length=130, command=lambda e: self.refresh_view()).pack(side=tk.LEFT)

        ttk.Label(top, text="nms").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Scale(top, from_=0.1, to=0.9, variable=self.nms_thresh, orient=tk.HORIZONTAL, length=130, command=lambda e: self.refresh_view()).pack(side=tk.LEFT)

        self.info = ttk.Label(self.root, text="先点随机抽一张", padding=8)
        self.info.pack(side=tk.TOP, fill=tk.X)

        body = ttk.Frame(self.root, padding=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(body, text="原图", padding=8)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        mid_frame = ttk.LabelFrame(body, text="锚框", padding=8)
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        right_frame = ttk.LabelFrame(body, text="预测框", padding=8)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.label_orig = tk.Label(left_frame, text="等待图片", bg="gray90")
        self.label_orig.pack(fill=tk.BOTH, expand=True)

        self.label_anchor = tk.Label(mid_frame, text="等待图片", bg="gray90")
        self.label_anchor.pack(fill=tk.BOTH, expand=True)

        self.label_pred = tk.Label(right_frame, text="等待图片", bg="gray90")
        self.label_pred.pack(fill=tk.BOTH, expand=True)

    def reload_weights(self):
        self._load_weights()
        self.refresh_view()

    def _get_random_index(self):
        return random.randint(0, len(self.dataset) - 1)

    def load_random_image(self):
        self.current_idx = self._get_random_index()
        self._load_image_by_idx(self.current_idx)
        self.refresh_view()

    def prev_image(self):
        if len(self.dataset) == 0:
            return
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self._load_image_by_idx(self.current_idx)
        self.refresh_view()

    def next_image(self):
        if len(self.dataset) == 0:
            return
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self._load_image_by_idx(self.current_idx)
        self.refresh_view()

    def _load_image_by_idx(self, idx):
        # 用 dataset 里的原图路径，保证你看到的是原始图片
        row = self.dataset.annotations.iloc[idx]
        img_name = row["img_name"]
        img_path = os.path.join(self.dataset.images_dir, img_name)

        self.current_pil = Image.open(img_path).convert("RGB")
        self.current_tensor, _ = self.dataset[idx]  # 与训练时完全同一套 transform
        self.current_tensor = self.current_tensor.unsqueeze(0).to(self.device)

        self.info.config(text=f"当前图片: {img_name} | idx={idx} | size={self.current_pil.size}")

    @torch.no_grad()
    def refresh_view(self):
        if self.current_pil is None or self.current_tensor is None:
            return

        # 1) 前向
        cls_preds, bbox_preds, features = self.net(self.current_tensor)
        cls_prediction, bbox_prediction = self.net.prediction_flatten(cls_preds, bbox_preds)
        cls_prediction = cls_prediction.squeeze(0)   # [N, 2]
        bbox_prediction = bbox_prediction.squeeze(0)  # [N, 4]

        # 2) 生成 anchors（严格按你训练时的配置）
        anchor_sizes = [[32, 64], [64, 128], [128, 256]]
        strides = [4, 16, 64]
        anchors = []
        for feature, sizes, stride in zip(features, anchor_sizes, strides):
            a = anchor_generator(featuremap=feature, anchor_size=sizes, stride=stride).create().to(self.device)
            anchors.append(a.float())
        anchors = torch.cat(anchors, dim=0)  # [N, 4]

        # 3) decode + score
        probs = F.softmax(cls_prediction, dim=-1)
        fg_scores = probs[:, 1]
        decoded_boxes = decode_bbox(anchors, bbox_prediction)

        w, h = self.current_pil.size
        decoded_boxes = clamp_boxes_xyxy(decoded_boxes, w=w, h=h)

        # 4) filter + nms
        keep = fg_scores > float(self.score_thresh.get())
        filtered_boxes = decoded_boxes[keep]
        filtered_scores = fg_scores[keep]

        if filtered_boxes.numel() > 0:
            keep_idx = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=float(self.nms_thresh.get()))
            final_boxes = filtered_boxes[keep_idx]
            final_scores = filtered_scores[keep_idx]
        else:
            final_boxes = torch.empty((0, 4), device=self.device)
            final_scores = torch.empty((0,), device=self.device)

        # 5) debug: 看看正样本/背景分数
        topk_scores, _ = fg_scores.topk(k=min(10, fg_scores.numel()))
        print("Top10 fg scores:", topk_scores.detach().cpu().tolist())

        # 6) 画图
        orig_show = self.current_pil.copy()
        pred_show = draw_boxes_pil(orig_show.copy(), final_boxes.detach().cpu(), final_scores.detach().cpu(), color="red")

        if self.show_anchors.get():
            anchor_show = draw_anchors_pil(orig_show.copy(), anchors.detach().cpu(), max_draw=250, color="blue")
        else:
            anchor_show = orig_show.copy()

        # 7) 缩放到窗口大小
        orig_show = orig_show.resize((360, 360))
        anchor_show = anchor_show.resize((360, 360))
        pred_show = pred_show.resize((360, 360))

        # 8) 更新显示
        tk_orig = ImageTk.PhotoImage(orig_show)
        tk_anchor = ImageTk.PhotoImage(anchor_show)
        tk_pred = ImageTk.PhotoImage(pred_show)

        self.label_orig.config(image=tk_orig, text="")
        self.label_anchor.config(image=tk_anchor, text="")
        self.label_pred.config(image=tk_pred, text="")

        self.label_orig.image = tk_orig
        self.label_anchor.image = tk_anchor
        self.label_pred.image = tk_pred


if __name__ == "__main__":
    # 你也可以直接换成自己的 train_dataloader
    dataloader = get_test_dataloader(
        root_dir=r"C:\Users\li170\Desktop\banana-detection\bananas_train",
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    root = tk.Tk()
    app = BananaSSDViewer(root, dataloader=dataloader, weights_path="net.pth", input_size=256)
    root.mainloop()
