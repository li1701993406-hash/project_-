import torch
import torch.nn as nn
from d2l import torch as d2l

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

        #载入数据集
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    batch_size = 64
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #定义模型
    model = nn.Sequential(
        nn.Flatten(),nn.Linear(784,64),nn.ReLU(),
        nn.Dropout(0.3),nn.Linear(64,10)).to(device)
        #SGD
    updater = torch.optim.SGD(model.parameters(),lr = 0.1,weight_decay = 0.01)
        #交叉熵损失函数
    loss = nn.CrossEntropyLoss()

        #训练
    num_epochs = 10

    for epoch in range(num_epochs):
        num_loss = 0
        num_acc = 0
        total_loss = 0
        total_acc = 0
        for x,y in train_iter:
            #数据放入GPU
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            updater.zero_grad()
            l = loss(y_hat,y)
            l.backward()
            updater.step()

            #每epoch的平均loss
            num_loss += 1
            total_loss += l.item()

            #每epoch的平均acc
            y_hat = torch.argmax(y_hat,dim = 1)
            num_acc += (y_hat == y).float().sum().item()
            total_acc += y.numel()
        print(f'epoch:{epoch+1},loss:{total_loss/num_loss:.4f},acc:{num_acc/total_acc*100:.4f}%')

        #测试
    model.eval()
    with torch.no_grad():
        num_acc = 0
        total_acc = 0

        for x,y in test_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)

            y_hat = torch.argmax(y_hat,dim = 1)
            num_acc += (y_hat == y).float().sum().item()
            total_acc += y.numel()
        print(f'测试集平均acc:{num_acc/total_acc*100:.4f}%')

    #保存模型数据
    torch.save(model.state_dict(),'MLP.pth')
    print(f'模型已保存')

