import torch
import tkinter as tk
from torchvision import transforms
import io
from PIL import Image
from CNN_LeNet import Model
import numpy as np



    #设定GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

    #初始化参数
model_choose = None     #初始化模型类型
last_x,last_y = None,None   #初始化鼠标坐标
image_Tensor = None     #初始化图片张量
canvas_image = None    #初始化图片
canvas_size = 448
img_size = 28
img_array = np.zeros((img_size,img_size),dtype = np.float32)

# 实例化CNN模型
model_cnn = Model().to(device)
state_dict = torch.load('CNN1.pth')
model_cnn.load_state_dict(state_dict)  # 导入已保存的模型参数
print(f'已实例化CNN')

# # 实例化MLP模型
# model_mlp = Model().to(device)
# state_dict = torch.load('MLP.pth')
# model_mlp.load_state_dict(state_dict)  # 导入已保存的模型参数
# print(f'已实例化MLP')


def button_root_start():
    """开始按钮,切换到选择模型页面"""
    print(f'开始按钮测试正常')
    # root.withdraw() #隐藏首页

    page1 = tk.Tk() #创建新窗口
    page1.geometry('960x576')
    label_text = tk.Label(page1,text = '选择参数',font = ('Arial', 20))
    button_cnn = tk.Button(page1,text = 'CNN',font = ('Arial', 20),command = model_choose_CNN)
    button_mlp = tk.Button(page1,text = 'MLP',font = ('Arial', 20),command = model_choose_MLP)
    button_sure = tk.Button(page1,text = '下一步',command = model_choose_sure,font = ('Arial', 17))

    label_text.place(relx = 0,rely = -0.3,relwidth = 1,relheight=1)
    button_cnn.place(relx = 0.375,rely= 0.45,width = 80,height = 60)
    button_mlp.place(relx = 0.55,rely= 0.45,width = 80,height = 60)
    button_sure.place(relx = 0.75,rely= 0.45,width = 80,height = 60)
    print(f'窗口切换功能正常')
    pass


def button_root_quit():
    """退出按钮"""
    root.destroy()
    print(f'退出成功')
    pass


# def canvas_to_tensor(image):
#     """把画布转为张量"""
#     img_tensor = torch.tensor(img_array)
#     img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
#     global image_Tensor
#     image_Tensor = img_tensor
#     return img_tensor



def model_choose_CNN():
    """选择CNN模型"""
    print(f'已选择CNN模型')

    global model_choose
    model_choose = 'cnn'


def model_choose_MLP():
    """选择MLP模型"""
    print(f'已选择MLP模型')

    global model_choose
    model_choose = 'mlp'


def model_choose_sure():
    """确认模型的选择并跳转到识别页面"""
    if model_choose == None:
        return

    page2 = tk.Tk()
    page2.geometry('960x576')
    page2.title('识别')
    page2.resizable(False,False)


        #清除画布
    def button_canvas_clear():
        """清除画布"""
        canvas.delete('draw')
        global img_array
        img_array.fill(0)
        print(f'已清空画布')
        pass
    canvas = tk.Canvas(page2,width = 960,height = 576)
    button_inference = tk.Button(page2,text = '预测',command = button_canvas_inference,font = ('Arial', 20))
    button_clear = tk.Button(page2,text = '清除',command = button_canvas_clear,font = ('Arial', 20))
    canvas.create_text(700,30,text='正在使用cnn(卷积神经网络)' if model_choose == 'cnn' else '正在使用mlp(多层感知机)',
                       font=('Arial', 15),
                       fill='black'
                       )
    canvas.create_rectangle(470, 68, 918, 516,outline='black', width=2)     #绘制正方形取景框
    canvas.pack()
    button_inference.place(relx = 0.1,rely= 0.8,width = 60,height = 40)
    button_clear.place(relx = 0.3,rely= 0.8,width = 60,height = 40)


    def get(event):
        global last_x, last_y
        last_x, last_y = event.x, event.y


    def start_draw(event):
        """绘画"""
        global img_array
        if event.x>470 and event.x<918 and event.y>68 and event.y<516:  #限制绘画边界
            left, top = 470, 68
            x,y = event.x - left,event.y - top
            scale = canvas_size // img_size
            i = int(y // scale)
            j = int(x // scale)
            if 0 <= i < img_size and 0 <= j < img_size:
                img_array[i,j] = 1.0


            global last_x, last_y
            canvas.create_line(last_x,last_y,event.x,event.y,fill = 'black',width = 5,tags = 'draw')
            last_x, last_y = event.x, event.y

        #绑定事件
    canvas.bind('<Button-1>',get)    #记录拖动时的坐标
    canvas.bind('<B1-Motion>',start_draw)      #记录鼠标按下时坐标


def button_canvas_inference():      #将转化成张量的图片传入模型
    """把画布转为张量"""
    print(f'正在将图片转为张量')
    img_tensor = torch.tensor(img_array).to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    global image_Tensor
    image_Tensor = img_tensor.to(device)

    """接入模型并开始推理"""
    print(f'开始预测')
    if model_choose == 'cnn':
            #以cnn模型推理
        with torch.no_grad():
            model_cnn.eval()    #切换推理模式
            y_hat = model_cnn(image_Tensor).to(device)
            y_hat = y_hat.argmax(dim = 1).item()
            print(f'CNN模型的预测结果为:{y_hat}')
            # print(f'{image_Tensor}')


    #     #以mlp推理
    # elif model_choose == 'mlp':
    #     with torch.no_grad():
    #         model_mlp.eval()    #切换推理模式
    #         y_hat = model_mlp(image_Tensor).to(device)
    #         y_hat = y_hat.argmax(dim = 1)
    #         print(f'MLP模型的预测结果为:{y_hat}')



def button_canvas_back():
    """返回上级页面重新选择模型"""
    pass


def weights_train_choose():
    """选择是否自定义参数训练模型"""
    pass


        #创建首页窗口
root = tk.Tk()
root.geometry('960x576')   #窗口尺寸
root.title('手写数字识别')
label = tk.Label(root,text = '手写数字识别',font = ('微软雅黑', 45))
label.place(relx = 0,rely = -0.2,relwidth = 1,relheight = 1)
        #创建按键
button1 = tk.Button(text = '开始',command = button_root_start,font = ('Arial', 20))
button2 = tk.Button(text = '退出',command = button_root_quit,font = ('Arial', 20))
button1.place(relx = 0.375,rely= 0.5,width = 80,height = 60)
button2.place(relx = 0.55,rely= 0.5,width = 80,height = 60)

    #窗口循环
root.mainloop()


