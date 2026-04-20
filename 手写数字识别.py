import tkinter as tk
from PIL import Image
import torch
from torchvision import transforms
import io
import torch.nn as nn

    #设定GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

    #定义模型
model = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 64), nn.ReLU(),
    nn.Dropout(0.3), nn.Linear(64, 10)).to(device)

    #加载模型数据
state_dict = torch.load('MLP.pth',map_location = device)
model.load_state_dict(state_dict)

#初始化鼠标x,y坐标
last_x, last_y = None, None
    # 鼠标按下：记录起点
def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

    # 鼠标拖动：画线
def last_draw(event):
    global last_x, last_y           #边界检测
    if event.x > 460 and event.x < 740 and event.y > 220 and event.y < 500 and last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, event.x, event.y,
                           width=10,
                           fill='black',
                           capstyle='round',
                           smooth=True,
                           tags = 'draw')
    last_x, last_y = event.x, event.y

    # 松开鼠标
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

    #清除画布函数
def delete_canvas():
    print('已清空画布')
    canvas.delete('draw')

    #把画布(img)转成(1,784)格式的tensor
def canvas_to_tensor(canvas):
    roi_x, roi_y = 460, 220
    roi_w, roi_h = 280, 280
    ps = canvas.postscript(colormode = 'color',
                           x = roi_x,
                           y = roi_y,
                           width = roi_w,
                           height = roi_h,)
    img = Image.open(io.BytesIO(ps.encode('utf8')))
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = 1 - img
    img = img.unsqueeze(0)
    print('画布已转为张量')
    return img

    #封装canvas_to_tensor函数
def predict():
    print('开始预测')
    img_tensor = canvas_to_tensor(canvas).to(device)
        #开始把数据传入模型开始预测
    model.eval()
    with torch.no_grad():
        y_pred = model(img_tensor)
        pred1 = torch.softmax(y_pred,dim = 1)
        pred2 = torch.argmax(y_pred, dim = 1).item()

    print(f'预测结果:{pred2},可信度:{pred1*10}')

    #创建画布
win = tk.Tk()
win.title('手写数字识别')
canvas = tk.Canvas(win, width=1200, height=720,)
canvas.create_rectangle(460,220,740,500,outline = 'black')
canvas.create_text(600,600,text = '请在方框内写字',font = ("Arial", 20))
canvas.pack()

    #定义预测按钮
button = tk.Button(win,text = '预测',command = predict,width = 8,height = 1)
button.pack(pady = 10)
    #定义清除按钮
button = tk.Button(win, text = '清除', command = delete_canvas, width = 8, height = 1)
button.pack()
    #绑定鼠标事件
canvas.bind('<Button-1>',start_draw)
canvas.bind('<B1-Motion>',last_draw)
canvas.bind('<ButtonRelease-1>',reset)

win.mainloop()


