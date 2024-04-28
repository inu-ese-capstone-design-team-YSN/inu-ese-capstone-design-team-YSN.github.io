---
layout: single
title:  "Weekly Diary 8주차(2024.04.22 ~ 2024.04.28)"
excerpt: "8주차 개발 기록 정리"
categories: WeeklyㅤDiary
# tag: [python, blog, food]
toc: true
toc_sticky: true
typora-root-url: ../
auto_profile: false
# sidebar:
#    nav: "docs"
# search: false
---



# 1. 주간 활동 정리

## **1) 활동 기록**

**[주간 정규 회의 및 활동 기록]**  
2024-04-24 (수) **[17:00 ~ 18:00]**, 18:00 ~ 22:00  
2024-04-28 (일) 13:00 ~ 18:00  
총 활동 시간: 10시간 0분  
**<u>대괄호로 작성된 부분은 정규 회의 시간임.</u>*
{: .notice--danger .text-center}

<br>

## **2) 개발 기록**

- 추가 신청한 H/W를 수령 후 탑재
- 팀별 추후 방향성에 대한 설계

<br>

------

# 2. 공통 개발

## 1) 정규 회의

> 정규 회의를 통해 개발의 전체적인 방향성을 고민하였다.

![image-20240428212316202](/images/Untitled/image-20240428212316202.png){: img-default-setting}

앞으로의 개발 FLOW는 다음과 같다.

1. **색조 보정**

2. **BWAC**(Brightness Weighted Adaptive Convolution)

   > Adaptive Convolution Filter를 사용하여 이미지 축소 및 조도 보정한다.

3. **Brightness Correction**

4. **DBSCAN을 이용한 색상 클러스터링**

   > 이 과정에서 색상 클러스터를 구분한다.

5. **CNN을 활용한 대표 색상 추론**

   > 이 과정을 **Color Inference**라고 한다.

<br>

## 2) 자석 및 멀티탭 탑재

> 시스템 H/W에 자석을 부착하여 안정성을 높이고 멀티탭을 탑재해 배선을 정리했다.

![image-20240428212352663](/images/Untitled/image-20240428212352663.png)

⇒ 전방부 문에 자석 2개를 부착하였다.

![image-20240428212434276](/images/Untitled/image-20240428212434276.png){: .img-default-setting}

⇒ 측면부 문에 자석  2개를 부착하였다.

> 그 외에도 H/W의 윗부분인 모듈부 천장에 자석 1개를 부착하고  
> 모듈부 전면에 프레임을 추가하여 천장이 기우는 현상을 제어했다.

<br>

------

# 3. CI 팀(Color Inference)

> 최종적으로 대표색상을 추출하는 방법으로 CNN을 테스트 해보기 위해서 이미지 데이터셋을 구해서 기초적인 CNN모델을 구현한 뒤 적용 가능성과 성능을 확인해보았다.

<br>

## 1) CNN모델 테스트

- **CNN의 구조**

합성곱 신경망 CNN은 데이터의 특징을 추출하여 특징들의 패턴을 파악하는 구조이다. CNN은 convolution과정과 pooling과정을 통해 진행한다.

![image-20240428212451517](/images/Untitled/image-20240428212451517.png)

이를 구현한 코드는 다음과 같다.

학습에 사용된 데이터셋은  깃허브에서 오픈소스로 제공된 데이터셋을 사용했다. 또한 원단의 색상이 레이블링된 데이터셋이 존재하지 않기때문에 임시로 전체 이미지의 색상 평균을 레이블이라고 가정한 상태에서 학습을 진행했다.

```python
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        
        # 이미지를 RGB 형식으로 변환
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        # 이미지의 평균 RGB 값 계산
        label = np.mean(image, axis=(0, 1))  
        
        # 0~1 사이로 정규화
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32) / 255.0  
        
        return image, label
        
# 데이터셋 및 데이터로더 생성
dataset = CustomDataset(root_dir='train_set')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.fc1 = nn.Linear(32 * 25 * 25, 32 * 25)
        self.fc2 = nn.Linear(32 * 25, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
model = SimpleCNN()

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(10):
    running_loss = 0.0
    for images, labels in data_loader:
        images = images.float()
        images = images.permute(0, 3, 1, 2)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/10, Loss: {running_loss/len(data_loader)}')
```

학습 결과와 Epoch당 Loss는 다음과 같다

> Epoch 1/10, Loss: 0.20399277233511384  
> Epoch 2/10, Loss: 0.05952263447894438  
> Epoch 3/10, Loss: 0.0016590737891365847  
> Epoch 4/10, Loss: 0.0018240681229175237  
> Epoch 5/10, Loss: 0.001504460798653895  
> Epoch 6/10, Loss: 0.001212007130261128  
> Epoch 7/10, Loss: 0.0008855674576162542  
> Epoch 8/10, Loss: 0.0012020306051513784  
> Epoch 9/10, Loss: 0.0008853987460817336  
> Epoch 10/10, Loss: 0.0009359312001896569

<br>

- 레이블링이 된 TPG 이미지로 테스트해 보았다.

  ![image-20240428212539319](/images/Untitled/image-20240428212539319.png)

  *⇒ 팬톤 17-3014를 촬영한 이미지*

<br>

```python
# 테스트할 이미지
test_image_path = "test_set/test2.jpg"
test_image = Image.open(test_image_path)

# 이미지를 모델이 요구하는 형식으로 전처리
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # 이미지 크기를 모델의 입력 크기에 맞게 조정
    transforms.ToTensor(),           # 이미지를 텐서로 변환
    #transforms.Lambda(lambda x: x / 255.0)  # 이미지를 [0, 1]로 정규화
])

# 배치 차원을 추가
test_image_tensor = transform(test_image).unsqueeze(0)  

# 모델을 평가 모드로 설정
model.eval()

# 테스트 이미지를 모델에 전달하여 예측을 수행
with torch.no_grad():  # 그래디언트 계산을 비활성화
    output = model(test_image_tensor)

# 예측 결과 출력
#print("Output:", output)

#정규화된 rgb 실수화
rgb_array = np.array(output) * 255

# RGB 순서로 배열
rgb_array = rgb_array.reshape(3)

#반올림
rgb_array = np.round(rgb_array)
print(rgb_array)  # 변환된 실수 배열 출력
```

테스트 결과는 다음과 같다.

CNN이 추출한 대표값: [130 101 101] TPG 레이블 값: [169,114,157]

아주 큰 오차가 있는것을 확인 할 수 있는데 원인을 분석하기 위해서 train set의 이미지를 분석해보았다.

![image-20240428212555659](/images/Untitled/image-20240428212555659.png)

![image-20240428212606760](/images/Untitled/image-20240428212606760.png)

학습한 데이터셋은 면 원단을 촬영한 이미지이므로 딥러닝에서 원단과 컬러코팅 종이는 매우 차이가 크다고 판단했고 추가로 대부분 단색 원단임을 확인하고 데이터셋으로 사용했지만 4000장이 넘는 원본 데이터셋에서 학습 속도를 위해 800장 가량으로 간추리는중 다색상 이미지가 의도보다 많이 포함된것을 확인했다. 여건상 색상의 평균값을 레이블링값으로 상정했기 때문에 다색상 이미지가 포함될수록 성능이 떨어지거나 아예 이상한 결과가 나올 수 있다.

그렇기 때문에 원단과 코팅 종이의 재질 차이라도 문제 해결이 되는지 확인하기 위해서 다른 샘플 스와치를 촬영한 이미지로 테스트를 계속해 보았다.

![image-20240428212616678](/images/Untitled/image-20240428212616678.png)

*⇒ 사용한 샘플*

CNN 추출 결과값: [149. 169. 170.]

> 아직 팬톤 TCX가 없는 환경이기 때문에 현재 사용하는 스와치는 레이블 값이 존재하지 않는다. 그래서 단순하게 그림판으로 비교해서 어느정도의 색상 차이가 있는지 비교해보았다.

![image-20240428212625798](/images/Untitled/image-20240428212625798.png)

중앙 부분의 단색이 CNN이 추론한 대표색상인데 육안으로 보았을때 전체 스와치의 색상을 평균낸 색상과 비슷한 톤의 색상을 뽑아낸것을 확인했다.

- **최종 정리**

  이번 테스트에는 다음과 같은 부족한점이 있었다.

  1. 우리팀이 촬영한 레이블링된 데이터셋 부재
  2. 학습 속도로 인한 시간 사정상 다양한 모듈 구조 테스트 불가
  3. 사용한 데이터셋의 낮은 적합성

  그렇기 때문에 우선적으로 CNN 대표색상 추출의 가능성을 확인하는 의미의 테스트였고 추가로 보유한 TPG와 추가로 LINC 3.0 지원을 이용해 구매할 TCX를 촬영해 데이터셋을 만들고 모델을 최적의 방향으로 튜닝해 나간다면 좋은 결과를 얻을 수 있을것으로 보인다.

<br>

# 4. BWAC 팀

> 우선적으로 해결해야 하는 부분이 무엇인지에 치중하고 앞으로의 LINC 3.0에서 구매할 TCX 색상 선정 기준을 마련하였다.

<br>

## 1) LINC 3.0 지원 사업

> LINC 3.0 지원 사업에서 예산 70만원을 배정받았기 때문에, 해당 지원금으로 어떤 품목을 구매할지 고려하였다. 팬톤 TCX의 색 선정 기준을 간단한 TPG 촬영을 통해 추가적으로 마련한 후, 상세 품목을 결정하기로 한다.

현재의 대략적인 예산 배분은 다음과 같이 진행할 예정이다.

- 팬톤 TPG → 약 350,000원
- LCD 디스플레이 → 약 50,000원
- 팬톤 TCX 낱개 15장 → 315,000원

팬톤 TPG를 구매하기로 결정하였기 때문에, 기존에 계획하던 TCX의 구매 가능한 수량이 현저히 줄어들었다. 따라서 팬톤 TCX 14장을 선정하는 기준을 마련하기로 한다.

기존의 자체적인 실험을 통해 색조(Saturation)는 조도 차이에 의한 색차에 미미한 영향을 준다는 결론을 내렸다. 따라서, 색조는 보색관계를 갖는 두 가지 색상으로만 분류하고, 채도와 명도에 대해서 배치 선형 계획법을 통해 색상을 선정한다.

![image-20240428212638382](/images/Untitled/image-20240428212638382.png)

⇒ 색조 선정 기준

<br>

색조를 2가지로 분류할 경우, 명도와 채도를 고려하여 하나의 색조당 7개의 종류를 뽑아낼 수 있다.  이를 배치 선형 계획에 의거하여 특정 명도와 채도에 대해서 RGB색차의 구간별 차이가 가장 큰 구간들을 위주로 선정할 수 있도록 한다.

<br>

**1) 배치 선형 계획법**

**배치 선형 계획법**을 통해 **채도(Hue)간 RGB색차**와 **명도(Value)**를 기준으로 **채도의 RGB색차 변화량**이 큰 구간들을 선별할 수 있다. 색차 변화량이 큰 구간들을 선별하는 이유는 TCX로 조도 보정값을 세밀화 할 때, 조도 변화량이 큰 구간들을 위주로 최대한 고르게 보정하기 위함이다. 이 방법을 퉁해 두 가지 색조에서 특정 명도와 채도 구간을 동시에 고려하여 최적의 원단 색상을 선정할 수 있다.

![image-20240428212650082](/images/Untitled/image-20240428212650082.png)

- 위에 표는 명도의 채도별 구간 색차
- 아래 표는 명도의 채도별 구간 색차의 크기 계산해서 max 선별

<br>

**의사결정변수**

Xij; 채도 i(i=1,2,3,4)가 명도 j(j=1,2,3)에 배치되는 경우 원단 개수

Pij: 채도 i(i=1,2,3,4)가 명도 j(j=1,2,3)에 배치되는 경우 구간별 색차 크기

**목적함수**

![image-20240428212711980](/images/Untitled/image-20240428212711980.png)

**제약조건**

- 원단 제약조건: 모든 의사결정변수의 합은 7이다.

![image-20240428212724612](/images/Untitled/image-20240428212724612.png)

- 명도 제약조건: 각 명도당 하나 이상의 원단이 포함되어야 한다.

![image-20240428212746883](/images/Untitled/image-20240428212746883.png)

- 정수 계획법: 원단의 개수는 반드시 정수의 형태를 지닌다.

- 최대 원단 개수 제약조건: 각 명도에 해당하는 채도 구간은 하나 이하의 원단 개수를 가진다.

TPG를 촬영 후 각 명도에 대한 채도 구간별 RGB색차를 구해 TCX 색상을 선정한다.

<br>

**2) 기존 스와치와 유사한 색상들을 선별**

시연할 때 유사한 색상의 스와치가 있어야 유사성 검출을 직관화할 수 있다.

<br>

------

# 5. 향후 계획

## 1) CI 팀

> TPG 촬영을 우선적으로 해서 데이터셋 마련에 집중한다.

- [ ]  TPG 데이터셋 제작
- [ ]  CNN 모델 튜닝 또는 기존 모델 조사 후 테스트

<br>

## 2)  BWAC 팀

> LINC 주문은 5월 1일 이전에 반드시 해야한다.

- [ ]  LINC 3.0 주문 품목 정리 및 주문
- [ ]  BWAC 알고리즘 설계

<br>