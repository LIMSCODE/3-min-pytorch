#!/usr/bin/env python
# coding: utf-8

# # 뉴럴넷으로 패션 아이템 구분하기
# Fashion MNIST 데이터셋과 앞서 배운 인공신경망을 이용하여 패션아이템을 구분해봅니다.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 30
BATCH_SIZE = 64  //한번에처리하는 데이터갯수 - 반복마다 이미지를 64개씩 읽어줌


# ## 데이터셋 불러오기
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.FashionMNIST(   //FashionMNIST 데이터셋을 가져온다 ###
    root      = './.data/', 
    train     = True,        //학습용
    download  = True,
    transform = transform    //이미지를 파이토치 텐서로 변환함
)
testset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = False,      //성능평가용
    download  = True,
    transform = transform
)

train_loader = torch.utils.data.DataLoader( //데이터로더에 앞서 불러온 데이터셋을 넣어주고, 배치크기를 지정함 ###
    dataset     = trainset,         
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)
test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)


# ## 뉴럴넷으로 Fashion MNIST 학습하기
# 입력 `x` 는 `[배치크기, 색, 높이, 넓이]`로 이루어져 있습니다.
# `x.size()`를 해보면 `[64, 1, 28, 28]`이라고 표시되는 것을 보실 수 있습니다.
# Fashion MNIST에서 이미지의 크기는 28 x 28, 색은 흑백으로 1 가지 입니다.
# 그러므로 입력 x의 총 특성값 갯수는 28 x 28 x 1, 즉 784개 입니다.
# 우리가 사용할 모델은 3개의 레이어를 가진 인공신경망 입니다. 
### 이미지분류 신경망 구현
입력x, 레이블y를 받아 학습한후 새로운 x가왔을때 어떤패션아이템인지 예측한다 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)  //nn.Linear은 선형결합을 수행하는 객체를 만듬. fc1은 픽셀값 784개를 입력받아 가중치를 행렬곱하고 편향을 더해 256개출력
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)   //출력값 10개 각각은 클래스를 나타내며, 10개중 값이 가장 큰 클래스가 이 모델의 예측값이 됨. ###

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x                        //모델의 반환값(output)은 10개의 클래스이다. ###


# ## 모델 준비하기
# `to()` 함수는 모델의 파라미터들을 지정한 곳으로 보내는 역할을 합니다.
# 일반적으로 CPU 1개만 사용할 경우 필요는 없지만,
# GPU를 사용하고자 하는 경우 `to("cuda")`로 지정하여 GPU로 보내야 합니다.
# 지정하지 않을 경우 계속 CPU에 남아 있게 되며 빠른 훈련의 이점을 누리실 수 없습니다.
//#모델을 위의 구현한 인공신경망으로 지정함
model        = Net().to(DEVICE)               //모델의 파라미터를 지정한 장치메모리로 보냄


# 최적화 알고리즘으로 파이토치에 내장되어 있는 `optim.SGD`를 사용하겠습니다.
optimizer    = optim.SGD(model.parameters(), lr=0.01)
# ## 학습하기
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): 
        data, target = data.to(DEVICE), target.to(DEVICE)  //학습 데이터를 DEVICE의 메모리로 보냄
        optimizer.zero_grad()                              //기울기 새로계산
        output = model(data)                               //학습데이터를 모델에 입력한결과 ##
        loss = F.cross_entropy(output, target)             //오차
        loss.backward()                                    //오차의기울기 계산
        optimizer.step()                                   //가중치수정


# ##테스트
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)                        //학습값
            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()
            # 가장 큰 값을 가진 클래스가 모델의 예측입니다.
            # 예측과 정답을 비교하여 일치할 경우 correct에 1을 더합니다.
            pred = output.max(1, keepdim=True)[1]        //모델의반환값(output)중 가장큰값을 가진클래스가 모델의 예측임 ###
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!
for epoch in range(1, EPOCHS + 1):   //이폭마다 학습,검증 반복
    train(model, train_loader, optimizer)                        //학습시키고
    test_loss, test_accuracy = evaluate(model, test_loader)      //검증하여 오차,정확도 가져옴
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(    //오차는 줄어들고, 정확도올라가는것 확인
          epoch, test_loss, test_accuracy))

