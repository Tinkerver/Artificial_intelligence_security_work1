import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

num_epoch = 10
BATCH_SIZE = 50

transform = transforms.Compose(
    [transforms.ToTensor()
    ]
)

#数据集加载
train_dataset = datasets.CIFAR10(
    root= '/home/cxm-irene/PyTorch/data/cifar10',
    train= True,
    transform= transform,
    download= True
)
train_loader = data.DataLoader(
    dataset= train_dataset,
    batch_size= BATCH_SIZE,
    shuffle= True
)
test_dataset = datasets.CIFAR10(
    root= '/home/cxm-irene/PyTorch/data/cifar10',
    train= False,
    transform= transform,
    download= False
)
test_loader = data.DataLoader(
    dataset= test_dataset,
    batch_size= BATCH_SIZE,
    shuffle= True
)

#定义网络
net = torchvision.models.resnet50(pretrained=True).cuda()
print(net)

#进行优化
#optimizer = torch.optim.RMSprop(net_cifar10.parameters(), lr = 0.005, alpha= 0.9)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, betas= (0.9, 0.99))
loss_function = nn.CrossEntropyLoss()

for epoch in range(num_epoch):
    print('epoch = %d' % epoch)
    for i, (image, label) in enumerate(train_loader):
        image, label = image.cuda(), label.cuda()
        x = net(image)
        loss = loss_function(x, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('loss = %.5f' % loss)

torch.save(net.state_dict(),'final.pth')

#对测试集的评估
total = 0
correct = 0
net.eval()

for image, label in test_loader:
    image, label=image.cuda(),label.cuda()
    x = net(image)
    _, prediction = torch.max(x, 1)
    total += label.size(0)
    correct += (prediction == label).sum()

print('There are ' + str(correct.item()) + ' correct pictures.')
print('Accuracy=%.2f' % (100.00 * correct.item() / total))
