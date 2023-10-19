#!/usr/bin/env python
# coding: utf-8

# # 파이토치로 구현하는 신경망
# ## 신경망 모델 구현하기

### 원리
입력층, 은닉층, 출력층 각층 노드(한단위의 인공뉴런)에서 
입력된자극을 가중치(입력값이 출력에주는 영향을 계산)에 행렬곱시키고 
편향(각노드의 데이터에 민감한지)을 더해주는 연산을 수행

행렬곱의 결과는 활성화함수(입력값을 출력신호로 변환, 입력신호합이 활성화일으키는지 결정)를 거쳐 
인공뉴런의 결과값을 산출함

인공지능 출력층이 낸 결과값과 정답을 비교해 오차를 계산하고, 
오차기반 신경망 학습시키려면 출력층의 가중치부터 입력층의 가중치까지 모두 경사하강법을 이용해 변경시킨다.
겹겹이 쌓인 가중치를 뒤에서부터 차례로 조정, 최적화 하는것이 역전파 알고리즘.

 
import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

### 데이터 준비 (넘파이, 싸이킷런, 맷플롯립) 
싸이킷런의 make_blob()함수 : 
데이터를 2차원벡터형태로 만들고, 만들어진 레이블 데이터는 
각 데이터 하나하나가 몇번째 클러스터에 속하는지 알려주는 인덱스임 (0,1,2,3 으로 인덱싱됨)
n_dim = 2
x_train, y_train = make_blobs(n_samples=80, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
x_test, y_test = make_blobs(n_samples=20, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)

### label.map()으로 0,1번 레이블로 바꿔줌
def label_map(y_, from_, to_):
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y

y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)

### 맷플롯립으로 시각화 (레이블 0이면 빨간점, 1이면 십자가)
def vis_data(x,y = None, c = 'r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x,y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*',markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c+'o' if y_ == 0 else c+'+')

plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()

### 넘파이벡터형식의 데이터를 파이토치 텐서로 바꿈
x_train = torch.FloatTensor(x_train)
print(x_train.shape)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

### 신경망모델 구현
class NeuralNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size):  //input_size = 신경망에 입력되는 데이터차원
            super(NeuralNet, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size) //torch.nn.Linear : 행렬곱, 편향 연산 객체반환
            self.relu = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, input_tensor):
            linear1 = self.linear_1(input_tensor)
            relu = self.relu(linear1)
            linear2 = self.linear_2(relu)
            output = self.sigmoid(linear2)
            return output


model = NeuralNet(2, 5)
learning_rate = 0.03
criterion = torch.nn.BCELoss()  //오차함수 : 이진교차엔트로피
epochs = 2000                   //학습데이터를 몇번 입력할지 (충분히학습)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)  //최적화알고리즘 - 확률적경사하강법(SGD) 선택 ###
                                                                     //새가중치 = 가중치-학습률x가중치에대한기울기
                                                                     //optimizer는 step()함수를 부를때마다 가중치를 학습률만큼 갱신

model.eval()
test_loss_before = criterion(model(x_test).squeeze(), y_test)        //오차구하기 
                                                                     //모델의 결괏값과 레이블값의 차원을 맞추기위해
                                                                     //squeeze함수를 호출후 오차구함

print('Before Training, test loss is {}'.format(test_loss_before.item()))
# 오차값이 0.73 이 나왔습니다. 이정도의 오차를 가진 모델은 사실상 분류하는 능력이 없다고 봐도 무방합니다.


# 자, 이제 드디어 인공신경망을 학습시켜 퍼포먼스를 향상시켜 보겠습니다.
for epoch in range(epochs):
    model.train()                                                    //학습모드로 변경
    optimizer.zero_grad() 
    train_output = model(x_train)                                    //모델에 학습데이터를 입력해 결과값 계산
    train_loss = criterion(train_output.squeeze(), y_train)          //오차계산
    if epoch % 100 == 0:                                             //100이폭마다 오차를출력해 학습이잘되는지 확인
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))
    train_loss.backward()                                            //역전파 (오차함수를 가중치로 미분하여 오차최소화방향을 구하고,
    optimizer.step()                                                 //그 방향으로 모델을 학습률만큼 이동시킴)
                                                                     //즉, 역전파알고리즘으로 확률적경사하강법(SGD)를 사용하여 
                                                                     //오차최소화방향으로 가중치를 학습률만큼 갱신한다. ###
 

model.eval()
test_loss = criterion(torch.squeeze(model(x_test)), y_test)
print('After Training, test loss is {}'.format(test_loss.item()))


# 학습을 하기 전과 비교했을때 현저하게 줄어든 오차값을 확인 하실 수 있습니다. ( 오차값 0.73 -> 0.09 )
# 지금까지 인공신경망을 구현하고 학습시켜 보았습니다.
# 이제 학습된 모델을 .pt 파일로 저장해 보겠습니다.
torch.save(model.state_dict(), './model.pt')
print('state_dict format of the model: {}'.format(model.state_dict()))
# `save()` 를 실행하고 나면 학습된 신경망의 가중치를 내포하는 model.pt 라는 파일이 생성됩니다. 
아래 코드처럼 새로운 신경망 객체에 model.pt 속의 가중치값을 입력시키는 것 또한 가능합니다.

new_model = NeuralNet(2, 5)                                       //저장된 가중치를 가져와 새모델에 적용
new_model.load_state_dict(torch.load('./model.pt'))
new_model.eval()
print('벡터 [-1, 1]이 레이블 1을 가질 확률은 {}'.format(new_model(torch.FloatTensor([-1,1])).item()))

