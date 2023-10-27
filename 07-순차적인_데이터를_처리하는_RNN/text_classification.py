#!/usr/bin/env python
# coding: utf-8

# # RNN
- 정적인 데이터가 아닌 순차적데이터, 시계열 데이터의 정보를 받아 전체내용을 학습하는 RNN - 텍스트, 자연어 처리에 사용됨. LSTM, GRU 등의 응용RUNN 개발됨
- 텍스트형식의 데이터셋인 IMDB는 영화리뷰 5만건으로 이뤄짐, 긍정 리뷰2, 부정리뷰 1로 레이블링됨. 
- 영화리뷰텍스트를 RNN에 입력해 영화평 전체내용을 압축하고 압축된 리뷰가 긍정인지 부정인지 판단하는 분류모델 만듬
- 토크나이징, 워드임베딩 : 문장을 단어별로 토크나이징하고, 2번째단어 토큰을 벡터로 변환함. (워드임베딩)
- 자연어전처리 : 토치텍스트의 전처리도구와 파이토치의 nn.Embedding 

# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets

# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)

# 데이터 로딩하기
print("데이터 로딩중...")
TEXT = data.Field(sequential=True, batch_first=True, lower=True)    //텍스트형태의 영화리뷰와 그에 해당하는 레이블을 텐서로 바꿔줌
LABEL = data.Field(sequential=False, batch_first=True)              //Sequential : 데이터셋이 순차적인 데이터셋인지 명시 (레이블은 단순 클래스를 나타내는 숫자이므로 순차적이지않음) 
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)               //splits함수로 모델에 입력되는 데이터셋을 만들어줌. (transet, testset)
TEXT.build_vocab(trainset, min_freq=5)                              //워드임베딩에 필요한 단어사전을 만듬
LABEL.build_vocab(trainset)

# 학습용 데이터를 학습셋 80% 검증셋 20% 로 나누기
trainset, valset = trainset.split(split_ratio=0.8)                //IMDB데이터셋에서는 검증셋이 존재하지않으므로 학습셋을 쪼개어 사용
train_iter, val_iter, test_iter = data.BucketIterator.splits(        //배치단위로 쪼개어 학습해야함
        (trainset, valset, testset), batch_size=BATCH_SIZE,        //trainset, valset, testset에서 반복시 배치생성하는 iterator 만듬
        shuffle=True, repeat=False)

vocab_size = len(TEXT.vocab)                                        //사전속 단어갯수, 레이블수를 정해주는 변수를 만듬
n_classes = 2
print("[학습셋]: %d [검증셋]: %d [테스트셋]: %d [단어수]: %d [클래스] %d"
      % (len(trainset),len(valset), len(testset), vocab_size, n_classes))

//RNN을 포함하는 신경망모델 
class BasicGRU(nn.Module):        //nn.Module 상속받음
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):        //self.n_layer = n_layers 은닉벡터의 층
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)          //n_vocab : 사전형태나열시 등재된단어수, embed_dim: 텐서가 가진 차원값
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)                   //은닉벡터의 차원값과 드롭아웃 정의
        self.gru = nn.GRU(embed_dim, self.hidden_dim,          //RNN은 입력길어질시 기울기폭발, 소실 되어 앞부분의정보를 담지못함 -> GRU가 기울기 적정유지하여 앞부분정보를 끝까지 전달해줌
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)        //압축된 텐서를 신경망에통과시켜 클래스에대한 예측을 출력

    def forward(self, x):                               //모델에 입력된 텍스트가 어떤전처리과정을 거치고 신경망에 입력되는지 정의 (데이터x : 한배치속에 있는 모든영화평)
        x = self.embed(x)                                //영화평을 워드임베딩 -> 시계열데이터로 변환
        h_0 = self._init_state(batch_size=x.size(0))        //init_state로 첫번째 은닉벡터 정의 
        x, _ = self.gru(x, h_0)  # [i, b, h]                //self.gru()가 반환한 텐서를 ;,-1,;로 인덱싱하면 (batch_size, 1, hidden_dim)모양의 텐서를 추출할수잇음
        h_t = x[:,-1,:]                                   //h_t가 영화리뷰를 압축한 은닉벡터                
        self.dropout(h_t)                                //리뷰속 모든내용을 압축한 h_t를 self.out신경망에 입력해 결과 출력함
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit
    
    def _init_state(self, batch_size=1):       
        weight = next(self.parameters()).data          // nn.GRU모듈의 첫번째 가중치텐서를 추출
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()        //이 텐서는 모델의 가중치텐서와 같은 데이터타입임.
                                                                                    //new - 모델의 가중치와 같은모양으로 변환, zero-0으로 초기화
            
def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)        //x 영화평데이터, y 그에상응하는 레이블에 접근
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환                //1,2값인 레이블을 0,1로 만듬
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


best_val_loss = None
for e in range(1, EPOCHS+1):                        //학습, 평가 수행 
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[이폭: %d] 검증 오차:%5.2f | 검증 정확도:%5.2f" % (e, val_loss, val_accuracy))        //검증오차 줄어듬, 검증정확도 증가함 확인
    
    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss


model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))

