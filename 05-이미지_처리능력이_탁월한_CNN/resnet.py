#!/usr/bin/env python
# coding: utf-8

# # 신경망 깊게 쌓아 컬러 데이터셋에 적용하기 
# Convolutional Neural Network (CNN) 을 쌓아올려 딥한 러닝을 해봅시다. 
- 여러단계의 신경망을 거치며 최초입력 이미지에 대한 정보 소실 -> 작은블록인 Residual블록으로 나누어, 출력에 전전층의 입력을 더하는 ResNet으로 해결 
- Residual 블록 = BasicBlock 파이토치 모듈로 정의하여 사용 (nn.Module이용하여 모듈 위에 모듈 쌓음)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# ## 하이퍼파라미터 
EPOCHS     = 300
BATCH_SIZE = 128

# ## 데이터셋 불러오기
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',            // CIFAR-10데이터셋 사용
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                   train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])),
    batch_size=BATCH_SIZE, shuffle=True)


# ## ResNet 모델 만들기
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()        //BasicBlock = Residual블록 의미
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)        //배치정규화 수행 : 학습률이 너무높으면 기울기 소실되는것 예방하여 안정화
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(    //nn.Sequential : nn.Module을 하나의 모듈로 묶는역할
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):                    //x를 컨볼루션, 배치정규화, 활성화함수 거치고 입력x를 self.shortcut을 거쳐 크기를 같게하고 활성화함수를 더함
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):                    //모델 정의 : 이미지를받아 컨볼루션, 배치정규화를 거친후 / 여러 BasicBlock층을 통과 / 풀링, 신경망 거쳐 예측 출력 
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))    //##BasicBlock함수 : self.make_layer()함수를 통해 ResNet모델의 주요층을 이룸.
            self.in_planes = planes                                      //self.make_layer()는 nn.Sequential도구로, 여러 BasicBlock을 모듈하나로 묶어줌.
        return nn.Sequential(*layers)

    def forward(self, x):                        //입력이 들어오면 일반적인 방식대로 컨볼루션, 배치정규화, 활성화함수를 통과하고 / 
        out = F.relu(self.bn1(self.conv1(x)))    //사전에 정의해둔 BasicBlock층을 가진 layer1,2,3을 통과함 ( 각 레이어는 2개의 Residual블록을 가짐)
        out = self.layer1(out)                   //그후 평균풀링을 거치고 마지막 신경망을 거쳐 분류결과를 출력한다.
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ## 준비
model = ResNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


print(model)


# ## 학습하기
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


# ## 테스트하기
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!
for epoch in range(1, EPOCHS + 1):
    scheduler.step()
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))




