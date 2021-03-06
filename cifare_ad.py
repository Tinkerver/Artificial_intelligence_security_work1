import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
num_epoch = 100
BATCH_SIZE = 250

def attack(image, model, target, epsilon=1e-3):
    input=image
    input.requires_grad = True
    pred = model(input)
    loss = nn.CrossEntropyLoss()(pred, target)
    loss.backward()
    output = input - epsilon * input.grad.sign()
    #print(input.grad.sign())
    del input
    return output.detach()

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
    batch_size= 1,
    shuffle= True
)

#定义网络
net = torchvision.models.resnet50(pretrained=True).cuda()
print(net)

#进行优化
#optimizer = torch.optim.RMSprop(net_cifar10.parameters(), lr = 0.005, alpha= 0.9)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, betas= (0.9, 0.99))
loss_function = nn.CrossEntropyLoss()

# for epoch in range(num_epoch):
#     print('epoch = %d' % epoch)
#     for i, (image, label) in enumerate(train_loader):
#         image, label = image.cuda(), label.cuda()
#         x = net(image)
#         loss = loss_function(x, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if i % 100 == 0:
#             print('loss = %.5f' % loss)

net.load_state_dict(torch.load('final.pth'))

#对测试集的评估
# total = 0
# correct = 0
net.eval()

# for image, label in test_loader:
#     image, label=image.cuda(),label.cuda()
#     x = net(image)
#     _, prediction = torch.max(x, 1)
#     total += label.size(0)
#     correct += (prediction == label).sum()
#
# print('There are ' + str(correct.item()) + ' correct pictures.')
# print('Accuracy=%.2f' % (100.00 * correct.item() / total))
count_ac=0      #本来预测正确
count_ag=0      #对抗样本识别错误
count_ag_l=0    #对抗样本识别为生成的样本
count_aga=0     #缩放后正确识别
for image, label in test_dataset:
    image, label=image.cuda(),label
    image_to_attack = image.clone()
    image_to_attack=image_to_attack.cuda()
    pred = net(image.unsqueeze(0))
    image=transforms.ToPILImage()(image.squeeze(0))
    #image.show()
    #print(label)
    pred=F.softmax(pred,dim=-1)[0]
    #print(pred)
    prob, clss = torch.max(pred, 0)
    prob2, clss2 = torch.min(pred[:10], 0)
    #print(prob,clss)
    #print(prob2, clss2)
    clss_pre=clss
    clss_ag=clss2
    if clss_pre==label:
        count_ac+=1
    target=clss2.unsqueeze(0)

    image_to_attack = image_to_attack.unsqueeze(0)
    for _ in range(30):
        image_to_attack = attack(image_to_attack, net, target)

    pred = net(image_to_attack)
    image_to_attack=transforms.ToPILImage()(image_to_attack.squeeze(0))
    #image_to_attack.show()
    pred = F.softmax(pred, dim=-1)[0]
    prob, clss = torch.max(pred, 0)
    #print(prob, clss)
    clss_ag_p=clss
    if clss_pre==label and clss_ag_p!=label:
        count_ag+=1

    if clss_pre==label and clss_ag_p==clss_ag:
        count_ag_l+=1


    image_to_attack= image_to_attack.resize((8, 8))
    image_to_attack = image_to_attack.resize((32, 32))
    #image_to_attack.show()
    image_to_attack=transforms.ToTensor()(image_to_attack).unsqueeze(0).cuda()
    pred = net(image_to_attack)
    pred = F.softmax(pred, dim=-1)[0]
    prob, clss = torch.max(pred, 0)
    if clss_pre == label and clss_ag_p != label and clss==label:
        count_aga+=1
    print(count_ac, count_ag, count_ag_l, count_aga)

print(count_ac,count_ag,count_ag_l,count_aga)
