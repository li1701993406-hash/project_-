import torch
from torch import nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

batch_size = 512
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

model = Model().to(device)    #实例化模型
loss = nn.CrossEntropyLoss()    #定义交叉熵损失函数
updater = torch.optim.SGD(model.parameters(),lr = 0.1,weight_decay = 0.001)  #权重衰退和SGD

if __name__ == '__main__':
    #训练
    epoch_num = 50
    train_loss_list = []
    train_acc_list = []
    for epoch in range(epoch_num):
        total_loss = 0
        total_acc = 0
        num_loss = 0
        num_acc = 0

        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)
            updater.zero_grad()
            y_hat = model(x).to(device)
            l = loss(y_hat,y)
            l.backward()
            updater.step()
        # print(f'\r第{epoch + 1}轮训练已完成\n{epoch + 1}/50',end = '')

            #参数统计
            num_loss += 1 #loss
            total_loss += l.item()

            total_acc += ((y_hat.argmax(dim = 1) == y)  #acc
                          .float().sum().item())
            num_acc += y.numel()
        train_loss = total_loss / num_loss
        train_acc = total_acc/num_acc*100
            #添加到列表
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
            #打印参数
        print(f'epoch:{epoch+1},loss:{train_loss:.4f},'
              f'accuracy:{train_acc:.2f}%')

        epochs = range(len(train_loss_list))

        # plt.plot(epochs,train_loss_list,label = 'train_loss')
        # plt.xlabel('epochs')
        # plt.ylabel('train_loss')

        plt.plot(epochs,train_acc_list,label = 'train_acc')
        plt.xlabel('epochs')
        plt.ylabel('train_acc')
    plt.show()

        #测试
    total_acc = 0
    num_acc = 0
    accuracy = []
    model.eval()
    with torch.no_grad():
        for x,y in test_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).to(device)

            #参数统计
            total_acc += ((y_hat.argmax(dim = 1) == y)  #acc
                          .float().sum().item())
            num_acc += y.numel()
            acc = total_acc / num_acc * 100
            accuracy.append(acc)
        x = range(len(accuracy))
        plt.plot(x,accuracy,label = 'accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
            # print(f'accuracy:{total_acc/num_acc*100:.2f}%')
        plt.show()
    # def clear():
    #     root.destroy()

    torch.save(model.state_dict(),'CNN1.pth')
    # root = tk.Tk()
    # root.title('训练完成确认窗口')
    # root.geometry('500x200')
    # button = tk.Button(root,text = '确认',command = clear)
    # label = tk.Label(root,text = '请确认参数已保存',font = ('Arial',20))
    # label.pack()
    # button.pack()
    # root.mainloop()

    print(f'训练已完成')
    print(f'模型参数已保存至CNN1.pth')