---
layout: single
title:  "Weekly Diary 10주차(2024.05.06 ~ 2024.05.12)"
excerpt: "10주차 개발 기록 정리"
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

<br>

# 1. 주간 활동 정리

## **1) 활동 기록**

**[주간 정규 회의 및 활동 기록]**  
2024-05-08 (수) 15:00 ~ 17:00, [**17:00 ~ 18:00**], 19:00 ~ 22:00  
2024-05-09 (목) 19:00 ~ 05:30  
총 활동 시간: 16시간 30분  
**<u>대괄호로 작성된 부분은 정규 회의 시간임.</u>*
{: .notice--danger .text-center}

<br>

# 2. 공통 활동

## 1) TPG 코드 작성

> TPG 2626장의 코드 번호를 뒷면에 작성한다. 이는 TPG를 촬영 규격에 맞게 커팅할 때, 데이터를 구분하기 위한 과정이다.

![image-20240512222545895](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222545895.png){: .img-default-setting}

![image-20240512222553475](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222553475.png){: .img-default-setting}

<br>

## 2) 캡스턴디자인 중간발표 PPT 제작

> 캡스턴디자인 중간발표를 위해 PPT를 제작하였다.

[입샛노랑 중간발표 자료 다운로드 링크](https://github.com/inu-ese-capstone-design-team-YSN/inu-ese-capstone-design-team-YSN.github.io/blob/master/_posts/%EC%9E%85%EC%83%9B%EB%85%B8%EB%9E%91%20%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9C%20%EC%9E%90%EB%A3%8C.pptx)

<br>

발표 이후 조교님들과 교수님의 피드백의 핵심은 다음과 같다.

- 이미지 축소 과정에서 loss가 없이 잘 되는지 확인할 필요가 있음.
- 클러스터링 과정에서 군집화되면서 대표색상을 찾는다면 이도 색상에 대한 loss 발생안하는지 확인할 필요가 있음.
- 시간 분배를 잘 하여 더 빠르게 완성도를 높일 필요가 있음.

<br>

=> <u>피드백을 잘 반영하여 개발의 속도를 높이고 알고리즘의 정확도를 잘 확인하는 데에 중점을 두어야 한다.</u>

<br>

------

# 3. CI 팀

## 1) CNN 모델 개선

> MobileNet 모델을 통한 성능 개선이 있었기 때문에 방향성을 잡고 계속 발전시키고자 하였다.

<br>

**MobileNet 이해**

> 이전 Torchvision내장 MobileNet 모델을 사용했을때 들쭉날쭉한 성능을 보였기 때문에 MobileNet 구조를 이해한 뒤 코드로 구현해 테스트해보고자 한다.

<br>

- **Depthwise Separable Convolution(\*DSConv\*)**

  MobileNet 구조의 핵심으로 표준 convolution을**Deptwise convolution(*dwConv*)**과 **Pointwise convolution(*pwConv*)**으로 분리한것이다. **dwConv**는 각 입력 채널당 1개의 filter를 적용하고, **pwConv**는 **dwConv**의 결과를 합치기 위해 1×1conv를 적용한다. Depthwise separable convolution은 이를 2개의 layer로 나누어 구성되고 이를 통해 모델 크기를 줄일 수 있다.

  ![image-20240512222656171](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222656171.png){: .img-default-setting}

  ![image-20240512222705738](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222705738.png){: .img-default-setting}

  ![image-20240512222724571](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222724571.png){: .img-default-setting}

  <br>

  **연산량 비교**

  Dk: 입력값 크기  
  M: 입력 채널 수  
  N: 출력 채널 수  
  DF: 피쳐맵 크기

  <br>

  *dwConv의 연산량*

  ![image-20240512222756069](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222756069.png){: .img-width-large}

  *pwConv의 연산량*

  ![image-20240512222801579](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222801579.png){: .img-width-large}

  *DSConv의 총합 연산량*

  ![image-20240512222807544](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222807544.png){: .img-width-large}

  *StdConv의 연산량*

  ![image-20240512222813167](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222813167.png){: .img-width-large}

<br>

- **MobileNet의 구조**

  ![image-20240512222825352](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222825352.png){: .img-default-setting}

<br>

- **MobileNet구조를 사용한 모델**

  ```python
  class MobileNetLike(nn.Module):
      def __init__(self):
          super(MobileNetLike, self).__init__()
          self.features = nn.Sequential(
              nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32),
              nn.Conv2d(64, 128, kernel_size=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128),
              nn.Conv2d(128, 256, kernel_size=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),
              nn.Conv2d(256, 512, kernel_size=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512),
              nn.Conv2d(512, 1024, kernel_size=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, groups=1024),
              nn.Conv2d(1024, 1024, kernel_size=1),
              nn.ReLU(inplace=True),
              nn.AdaptiveAvgPool2d(1)
          )
          self.classifier = nn.Sequential(
              nn.Dropout(0.2),
              nn.Linear(1024, 3)
          )
  
      def forward(self, x):
          x = self.features(x)
          x = x.view(x.size(0), -1)
          x = self.classifier(x)
          return x
  ```

  제일 처음 Std Conv layer가 하나 있고 이후로 계속 dwConv과 pwConv이 반복되는 구조를 가지고 있다.

<br>

**성능 비교**

> 성능 비교를 위해 DSConv layer를 적용한 모델과 일반 Conv layer를 적용한 모델의 학습속도와 모델 크기를 비교해 보았다. 정확도에 대한 비교는 임시 데이터셋을 사용했기 때문에 아직 정확도를 판단하기엔 이르다고 생각해 이후 TPG데이터셋이 완성되면 비교하기로 했다.

<br>

1. 학습 속도 비교

   *(100x100이미지 800장 10epoch기준)*

   |          | Std Conv | DSConv |
   | -------- | -------- | ------ |
   | 학습시간 | 8m24s    | 3m 8s  |

2. 모델 크기 비교

   *Std Conv*

   ![image-20240512222843322](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222843322.png){: .img-default-setting}

   *DSConv*

   ![image-20240512222851533](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222851533.png){: .img-default-setting}

   일반 Convolution방식에 비해 Depthwise Separable Convolution 방식을 사용해 모델의 추정 크기를 반 이하로 줄였고 학습 속도 역시 크게 단축 된 것을 확인 할 수 있다.

<br>

- **전체 코드**

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
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
dataset = CustomDataset(root_dir='new_train_set')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

class MobileNetLike(nn.Module):
    def __init__(self):
        super(MobileNetLike, self).__init__()
        self.features = nn.Sequential(
		        # Convolutional 레이어 정의
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, groups=1024),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            # Global Average Pooling 레이어
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        # 특성 추출
        x = self.features(x)
        # 특성 맵을 벡터로 변환
        x = x.view(x.size(0), -1)
        # 분류기에 통과하여 예측
        x = self.classifier(x)
        return x
        
# 모델 구조 확인
model = MobileNetLike()

summary(model, input_size = (1, 3, 100, 100), device = "cpu")

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(10):
    running_loss = 0.0
    for images, labels in data_loader:
        images = images.float()
        # PyTorch에서는 채널이 먼저였으므로 순서 변경
        images = images.permute(0, 3, 1, 2)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/10, Loss: {running_loss/len(data_loader)}')
    
# 테스트할 이미지
test_image_path = "test_set/test2.jpg"
test_image = Image.open(test_image_path)

# 이미지를 모델이 요구하는 형식으로 전처리
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # 이미지 크기를 모델의 입력 크기에 맞게 조정
    transforms.ToTensor(),          # 이미지를 텐서로 변환
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

# 테스트 이미지와 예측된 색상을 함께 시각화하는 함수 정의
def plot_color_with_image(rgb_code, image_path):
    # 주어진 RGB 코드를 [0, 1] 범위로 변환
    color = np.array(rgb_code) / 255.0

    # 이미지 불러오기
    img = mpimg.imread(image_path)

    # 플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 색상 플롯
    ax1.imshow([[color]])
    ax1.set_title(f'Output RGB: {rgb_code}')
    ax1.axis('off')

    # 이미지 플롯
    ax2.imshow(img)
    ax2.set_title('Input Image')
    ax2.axis('off')

    plt.show()
    
plot_color_with_image(rgb_array, test_image_path)
```

추후 TPG 데이터셋이 완성되면 학습을 진행하여 본격적인 성능을 테스트해보고 추가적으로 튜닝을 하거나 다른 방법을 사용해 모델을 개선한다.

<br>

## 2) DBSCAN for CNN 문제점 해결

CNN을 활용한 대표 색상 추론을 위해 클러스터 별로 픽셀(MCU)을 떼어내어 생성한 새로운 단색 원단 이미지를 사용하는데 각각의 단색 이미지의 **픽셀의 개수가 모두 다르기 때문에 CNN에 적용하는 이미지의 크기가 같지 않다는 문제점**이 있었다. 이에 대해 생각해 본 여러 해결 방법 중 우선적으로 간단하게 **이미지 크기 통일화 방법을** 적용하여 확인해 보았다.

아래의 출력 값들은 가장 적은 수의 픽셀을 가진 이미지 크기로 모든 단색 이미지 크기를 맞춰 조정한 이후 CNN을 활용한 대표 색상 추론에 적용해본 결과이다.

![image-20240512222924014](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222924014.png){: .img-width-large}



![image-20240512222931787](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222931787.png){: .img-width-large}

![image-20240512222941113](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222941113.png){: .img-width-large}

![image-20240512222947672](/images/2024-05-12-Weekly Diary(10주차)/image-20240512222947672.png){: .img-width-large}

***⇒***

**cluster 0 RGB**

- **DBSCAN : [ 0, 68, 141]**
- **DBSCAN for CNN : [16, 90, 153]**

**cluster 1 RGB**

- **DBSCAN : [ 11, 31, 61]**
- **DBSCAN for CNN : [31, 52, 79]**

**cluster 2 RGB**

- **DBSCAN : [16, 44, 67]**
- **DBSCAN for CNN : [33, 55, 81]**

**cluster 3 RGB**

- **DBSCAN : [26, 39, 57]**
- **DBSCAN for CNN : [41, 53, 72]**

**DBSCAN클러스터링 알고리즘 만을 사용한 대표 색상 추론 값과 클러스터 별 픽셀(MCU)을 떼어내어 생성한 단색 원단 이미지를 적용한 CNN 대표 색상 추론 값에서 모든 RGB에서 10~20 의 RGB값 차이가 존재함을 확인할 수 있었다.**

<br>

현재는 충분한 TPG 데이터셋이 아닌 임의의 원단 데이터셋을 통해서만 학습한 결과임에도 시각적으로 생각보다 정확한 결과 값을 얻을 수 있었으며 추후 TPG 데이터셋 마련 이후 더 정확한 값을 얻을 수 있을 것이라고 기대 할 수 있었다. 또한 픽셀의 개수에 따른 CNN 적용 이미지 크기 차이에 대한  문제점에 대해서 다른 해결 방법도 진행하면서 추가적인 개선 사항 마련에 대해 확인해보기로 하였다.

<br>

------

# 4. AC 팀

## 1) 개발 사항

> TPG 촬영 환경의 편의성과 white balancing의 정확도를 높이기 위해 기존 코드를 개선하였다.

<br>

- **tpg2sw (shortcut)**

  ```python
  #!/bin/bash
  
  # 2024.05.03 kwc
  # tpg2sw 1 9 하면 1부터 8개씩, 9번의 마지막 8개까지
  # 동작하도록 수정
  
  # 인자 검사: 두 개가 아닐 경우 오류 메시지 출력 후 종료
  if [ "$#" -ne 2 ]; then
      echo "Invalid number of arguments provided."
      echo "Usage: $0 [start_num] [end_num]"
      exit 1
  fi
  
  # 변수 할당:
  # start_num - 스크립트에 전달된 첫 번째 인자, 처리 시작 번호
  # end_num - 스크립트에 전달된 두 번째 인자, 처리 종료 번호
  start_num=$1
  end_num=$2
  
  # 시작 번호와 끝 번호 출력
  echo "Start number: $start_num"
  echo "End number: $end_num"
  
  # 파이썬 스크립트 실행
  python /home/pi/project/utility/tpg2sw.py --start $start_num --end $end_num
  ```

  기존에는 인자 하나를 입력 받아 tpg2sw.py를 동작하였다.

  ex) tpg2sw 1 → 1.jpg ~ 8.jpg를 crop하고 combine한다.

  인자 두 개를 입력 받아 여러 장의 이미지를 한 번의 명령으로 동작할 수 있도록 수정한다.

  ex) tpg2sw 1 90 → 1.jpg ~ 8.jpg + 9.jpg ~ 16.jpg + … + 90.jpg ~ 98.jpg

  첫 번째 인자는 시작 이미지의 number를 의미한다. 해당 숫자의 이미지부터 8장씩 동작한다. 이 과정은 두 번째 인자값에 도달하기 전까지 순차적으로 수행한다. 두 번째 인자는 마지막 이미지의 number를 의미한다. 마지막 이미지의 number부터 8장의 이미지를 동작한 후 프로그램을 종료한다.

<br>

- **tpg2sw.py**

  ```python
  import subprocess
  import argparse
  
  from image_utility import ImageUtility
  from path_finder import PathFinder
  
  '''
  
  2024.05.03 kwc
  tpg2sw로 입력 인자 두 개를 입력 받아 combined 및 히트맵 이미지까지 생성되도록 변경
  두 번째 입력값까지만 동작
  
  '''
  
  # 이미지 처리와 경로 탐색을 위한 유틸리티 클래스 인스턴스 생성
  util = ImageUtility()
  pf = PathFinder()
  
  # 명령줄 인수 파싱을 설정
  parser = argparse.ArgumentParser()
  parser.add_argument("--start", type=int, required=True, help="Starting number")
  parser.add_argument("--end", type=int, required=True, help="Ending number")
  
  args = parser.parse_args()
  
  extension=".jpg" # 파일 확장자 지정
  
  comb_type_vetically="v" # 세로 결합 유형 지정
  comb_type_horizontally="h" # 가로 결합 유형 지정
  
  # Left and Right
  lr_1 = 700
  lr_2 = 1950
  lr_3 = 3200
  
  # Upper and Lower
  ul_1 = 260
  ul_2 = 885
  ul_3 = 1510
  ul_4 = 2135
  ul_5 = 2760
  
  # TPG를 크롭할 사이즈 지정
  # TPG의 순서는 다음과 같이 되어 있다고 상정한다.
  # 1 5
  # 2 6
  # 3 7
  # 4 8
  crop_size = [
      (lr_1, ul_1, lr_2, ul_2),
      (lr_1, ul_2, lr_2, ul_3),
      (lr_1, ul_3, lr_2, ul_4),
      (lr_1, ul_4, lr_2, ul_5),
  
      (lr_2, ul_1, lr_3, ul_2),
      (lr_2, ul_2, lr_3, ul_3),
      (lr_2, ul_3, lr_3, ul_4),
      (lr_2, ul_4, lr_3, ul_5)
  ]
  
  # 시작 번호 지정
  if args.start:
      base_num = args.start
  
  # 시작 번호부터 종료 번호까지 8개씩 처리하고, 그 다음 8개씩 처리하는 작업 반복
  while base_num <= args.end:
      # TPG Image Crop
      for tpg_index in range(0, 8):
          file_name = f"{base_num+tpg_index}{extension}"
          
          left = crop_size[tpg_index][0]
          upper = crop_size[tpg_index][1]
          right = crop_size[tpg_index][2]
          lower = crop_size[tpg_index][3]
  
          print(f"crop {file_name}")
          result = subprocess.run(['imcrop', '-s', f'{left}', f'{upper}', f'{right}', f'{lower}', '-n', file_name], capture_output=True, text=True)
          print(file_name + " cropping done")
  
      print("\\n\\n")
  
      # TPG Image Combine
      # Combine Vertically - 1
      for tpg_index in range(0, 8, 2):
          upper_image_num = f"{base_num+tpg_index}"
          lower_image_num = f"{base_num+tpg_index+1}"
  
          upper_image_name = f"{base_num+tpg_index}_cropped{extension}"
          lower_image_name = f"{base_num+tpg_index+1}_cropped{extension}"
  
          combined_image_file_name = f"{base_num}_combined_{tpg_index}{extension}"
  
          print(f"combine {upper_image_name} and {lower_image_name} vertically")
          result = subprocess.run(['imcomb', '-u', upper_image_name, '-l', lower_image_name, '-n', combined_image_file_name, '-t', comb_type_vetically], capture_output=True, text=True)
          print(f"{upper_image_name} and {lower_image_name} combining done")
          
          # Image가 combined에 저장되므로, mv가 필요하다.
          # combined_dir에 있는 이미지를 cropped_dir로 이동한다.
          result = subprocess.run(['mv', f'{pf.swatch_combined_dir_path}/{combined_image_file_name}', f'{pf.swatch_cropped_dir_path}/'], capture_output=True, text=True)
  
      print("\\n\\n")
  
      # Combine Vertically - 2
      for tpg_index in range(0, 8, 4):
          upper_image_name = f"{base_num}_combined_{tpg_index}{extension}"
          lower_image_name = f"{base_num}_combined_{tpg_index+2}{extension}"
  
          # 위치에 따라 combined_image_file_name을 지정
          if tpg_index == 0:
              combined_image_file_name = f"{base_num}_left{extension}"
          elif tpg_index == 4:
              combined_image_file_name = f"{base_num}_right{extension}"
  
          print(f"combine {upper_image_name} and {lower_image_name} vertically")
          result = subprocess.run(['imcomb', '-u', upper_image_name, '-l', lower_image_name, '-n', combined_image_file_name, '-t', comb_type_vetically], capture_output=True, text=True)
          print(f"{upper_image_name} and {lower_image_name} combining done")
          
          # Image가 combined에 저장되므로, mv가 필요하다.
          # combined_dir에 있는 이미지를 cropped_dir로 이동한다.
          result = subprocess.run(['mv', f'{pf.swatch_combined_dir_path}/{combined_image_file_name}', f'{pf.swatch_cropped_dir_path}/'], capture_output=True, text=True)
  
      print("\\n\\n")
  
      # Combine Horizontally - 3
      left_image_name = f"{base_num}_left{extension}"
      right_image_name = f"{base_num}_right{extension}"
  
      combined_image_file_name = f"{base_num}_combined{extension}"
  
      print(f"combine {left_image_name} and {right_image_name} horizontally")
      result = subprocess.run(['imcomb', '-u', left_image_name, '-l', right_image_name, '-n', combined_image_file_name, '-t', comb_type_horizontally], capture_output=True, text=True)
      print(f"{left_image_name} and {right_image_name} combining done")
      # 최종적으로 만들어진 이미지가 combined 디렉터리에 저장되었음.
  
      print("\\n\\n")
  
      # convert combined image to heatmap
      combined_image_file_name = f"{base_num}_combined{extension}"
  
      print(f"{combined_image_file_name}")
      result = subprocess.run(['im2hmap', '-o', combined_image_file_name])
      print(f"{combined_image_file_name} heatmap converting done")
  
      print("\\n\\n")
  
      # 시작 번호 업데이트
      base_num += 8
  ```

  인자 두 개를 입력받아 여러 장의 tpg를 동시에 작업할 수 있도록 수정한다.

<br>

- **calcOuterSimilarity.py**

  ```python
  import numpy as np
  from skimage.color import rgb2lab, deltaE_ciede2000
  from image_utility import ImageUtility
  import json
  import os
  
  '''
  2024.05.03 kwc
  흰색 부분 CIE LAB으로 유사도 비교하도록 제작
  '''
  
  # 이미지 유틸리티 클래스 인스턴스 생성
  image_utility = ImageUtility()
  
  # JSON 파일 위치 확인 및 로드
  json_file_path = 'image_processing_results.json'
  
  # JSON 파일 존재 여부 확인
  if not os.path.exists(json_file_path):
      raise FileNotFoundError(f"No such file or directory: '{json_file_path}'")
  else:
      with open(json_file_path, 'r') as f:
          results = json.load(f)  # JSON 파일을 파이썬 객체로 로드
  
  # 각 이미지에 대한 유사도 및 거리 계산을 위한 반복 처리
  for image_id, data in results.items():
      target_rgb = tuple(data['Target Area Value'])  # 대상 영역의 RGB 값
      corrected_rgb = tuple(data['Corrected Target Area Value'])  # 보정된 대상 영역의 RGB 값
      label_rgb = tuple(data['Target Label'])  # 레이블의 RGB 값
  
      # Target Area Value와 Label RGB 비교
      similarity_target = image_utility.calculateLABSimilarity(target_rgb, label_rgb)  # LAB 색상 공간에서의 유사도 계산
      similarity_target_percentage = 100 - similarity_target  # 유사도를 퍼센테이지로 변환
      distance_target = image_utility.calculateRGBDistance(target_rgb, label_rgb)  # RGB 색상 공간에서의 거리 계산
      channel_distance_target = image_utility.calculateChannelIndependentRGBDistance(target_rgb, label_rgb)  # RGB 각 채널 별 독립적 거리 계산
  
      # Corrected Target Area Value와 Label RGB 비교
      similarity_corrected = image_utility.calculateLABSimilarity(corrected_rgb, label_rgb)  # 보정된 값과 레이블 사이의 LAB 유사도 계산
      similarity_corrected_percentage = 100 - similarity_corrected  # 유사도를 퍼센테이지로 변환
      distance_corrected = image_utility.calculateRGBDistance(corrected_rgb, label_rgb)  # RGB 색상 공간에서의 보정된 값과 레이블 사이의 거리 계산
      channel_distance_corrected = image_utility.calculateChannelIndependentRGBDistance(corrected_rgb, label_rgb)  # RGB 각 채널 별 독립적 거리 계산
  
      # 결과 출력
      print(f"Image {image_id}:")
      print(f"  Target Area Value vs Label RGB - LAB 색상 유사도 (Delta E): {similarity_target}")
      print(f"  Target Area Value vs Label RGB - 유사도 퍼센테이지: {similarity_target_percentage}%")
      print(f"  Target Area Value vs Label RGB - RGB 거리: {distance_target}")
      print(f"  Target Area Value vs Label RGB - 채널별 RGB 거리: {channel_distance_target}")
      print(f"  Corrected Target Area Value vs Label RGB - LAB 색상 유사도 (Delta E): {similarity_corrected}")
      print(f"  Corrected Target Area Value vs Label RGB - 유사도 퍼센테이지: {similarity_corrected_percentage}%")
      print(f"  Corrected Target Area Value vs Label RGB - RGB 거리: {distance_corrected}")
      print(f"  Corrected Target Area Value vs Label RGB - 채널별 RGB 거리: {channel_distance_corrected}")
  ```

  흰색 아크릴 부분에 해당하는 Outer Area의 색상을 비교하는 코드이다. TPG 없이 찍은 Target Area의 Value, TPG가 있을 때 찍은 Target Area Value, 색보정 이후의 Target Area Value에 대한 유사도 비교를 다양한 방식으로 진행한다. 기존 코드에서 출력 결과를 다음과 같이 수정하였다.

  - arget Area Value vs Label RGB - LAB 색상 유사도 (Delta E)
  - Target Area Value vs Label RGB - 유사도 퍼센테이지
  - Target Area Value vs Label RGB - RGB 거리
  - Target Area Value vs Label RGB - 채널별 RGB 거리
  - Corrected Target Area Value vs Label RGB - LAB 색상 유사도 (Delta E)
  - Corrected Target Area Value vs Label RGB - 유사도 퍼센테이지
  - Corrected Target Area Value vs Label RGB - RGB 거리
  - Corrected Target Area Value vs Label RGB - 채널별 RGB 거리

- **capture.py**

  ```python
  import subprocess # 외부 프로세스를 실행하고 그 결과를 다루기 위해 사용
  import os # 운영체제와 상호작용을 위한 모듈, 파일 및 디렉토리 관리에 사용
  import sys # 시스템 관련 파라미터와 함수를 다루기 위해 사용
  import curses # 터미널 핸들링을 위해 사용, 사용자와 대화형으로 상호작용하는 텍스트 인터페이스 구성에 활용
  from google.cloud import storage   # Google Cloud Storage 서비스를 사용하기 위한 클라이언트 라이브러리
  
  # 프로젝트의 data_class 디렉토리를 모듈 검색 경로에 추가하여 해당 디렉토리의 모듈을 사용 가능하게 함
  
  from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
  from cloud_controller import CloudController  # 클라우드 관련 작업을 관리하는 클래스
  
  # ----------------------------------------------------------------------------- #
  
  '''
  2024.04.17, kwc
  
  SWATCH, TPG, TCX별로 촬영 모드를 선택하여
  촬영 시 이미지 이름 설정, 이미지 로그 기록,
  GCS 자동 업로드 기능 구현을 동작하는 총체적인 프로그램 구현
  '''
  
  '''
  변수 설명:
  
  - preview_command: 카메라 미리보기를 시작하는 외부 명령어
  - capture_command: 이미지를 캡처하는 외부 명령어로, 실행 시 파일명이 동적으로 설정
  
  - preview_stop_key: 미리보기를 중단하는데 사용되는 키
  - user_input_exit: 프로그램을 종료하는데 사용되는 사용자 입력 키
  - user_input_capture: 이미지 캡처를 실행하는데 사용되는 사용자 입력 키
  - user_input_preview: 미리보기 모드를 시작하는데 사용되는 사용자 입력 키
  
  - image_file_extension: 캡처된 이미지 파일의 확장자를 정의하는 문자열, 기본적으로 '.jpg'로 설정
  
  - stdscr: curses 라이브러리에서 사용되는 표준 화면 객체로, 모든 출력과 입력은 이 객체를 통해 화면에 반영
  
  - image_type: 사용자가 선택한 이미지 유형('1'은 Swatch, '2'는 TPG, '3'은 TCX)을 저장하는 변수
  - cmd_input: 메인 루프에서 사용자로부터 받은 명령을 저장하는 변수
  - directory: 선택된 이미지 유형에 따라 이미지가 저장될 경로를 저장하는 변수
  - image_name: 캡처된 이미지의 이름을 저장하는 변수
  - upload_path: 클라우드에 업로드할 때 사용될 경로를 저장하는 변수
  - file_name: 캡처 명령을 실행할 때 사용되는 전체 파일 경로를 저장하는 변수
  '''
  
  ''' 클래스 인스턴스 생성 및 사용 '''
  path_finder = PathFinder() # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
  path_finder.ensureDirectoriesExist() # 
  cloud_controller = CloudController(path_finder) # 클라우드 관련 작업을 관리하는 클래스
  
  ''' 카메라 커맨드 지정 '''
  # 카메라 미리보기 커맨드 설정, '-t 0'은 타이머 없음을 의미
  preview_command = ['rpicam-hello', '-t', '0'] 
  # 이미지 캡처 커맨드, '-o'는 출력 파일 경로, '-t 1000'은 1000ms 동안 실행
  capture_command = ['libcamera-still', '-o', '', '--awb', 'daylight', '--shutter', '10000', '--gain', '1', '--contrast', '1', '--saturation', '1', '--nopreview'] 
  # libcamera-still -o output3.jpg --awb daylight --shutter 14000 --gain 1 --contrast 1 --saturation 1 --nopreview
  
  ''' 사용자 입력에 대한 처리를 위한 키 설정 '''
  preview_stop_key = 'q'  # 미리보기 종료 키
  user_input_exit = '0'  # 사용자 입력: 종료
  user_input_capture = '2'  # 사용자 입력: 이미지 캡처
  user_input_preview = '1'  # 사용자 입력: 미리보기 시작
  user_input_delete = '5'  # 사용자 입력: 이미지 삭제
  
  # 저장될 이미지의 파일 확장자 설정
  image_file_extension = '.jpg'  # 이미지 파일 확장자로 '.jpg'를 기본값으로 설정
  
  # ----------------------------------------------------------------------------- #
  
  def readImageInfo():
      """
      이미지 번호 파일에서 마지막 번호를 읽어오는 함수
      파일이 존재하지 않을 경우 초기 값을 설정하여 반환
      """
      try:
          # 이미지 번호 파일을 읽기 모드로 열어 처리
          with open(path_finder.image_number_file_path, 'r') as file:
              data = file.readline().strip()
              if data:
                  return data.split(',')
              else:
                  return ['1', '', '']  # Swatch는 숫자를, TPG와 TCX는 코드를 사용
          # 파일이 존재하지 않을 경우 예외 처리
      except FileNotFoundError:
          with open(path_finder.image_number_file_path, 'w') as file:
              file.write('1,,')
          return ['1', '', '']
  
  def saveImageInfo(swatch_number, tpg_code, tcx_code):
      """
      캡처된 이미지의 번호를 파일에 저장하는 함수
      캡처마다 번호를 증가시켜 파일에 저장
      """
      
      # 이미지 번호 파일을 쓰기 모드로 열어 처리
      with open(path_finder.image_number_file_path, 'w') as file:
          file.write(f"{swatch_number},{tpg_code},{tcx_code}\\n") 
          # 전달받은 이미지 번호를 문자열로 변환하여 파일에 기록
  
  def validate_and_format_image_name(stdscr, image_type, last_tpg_code, tpg_count):
  
      """
      사용자로부터 이미지 이름을 입력 받고, 적절한 형식으로 변환하는 함수
      입력 형식은 '00-0000' 이며, curses 라이브러리를 사용하여 터미널에서 입력을 받음
      """
      curses.noecho()  # 사용자 입력을 화면에 바로 표시하지 않도록 설정
      stdscr.clear() # 화면을 초기화
  
      if image_type == '2' and tpg_count % 8 != 1:
          formatted_name = f"{last_tpg_code}_{tpg_count}"
      else:
          stdscr.addstr("Enter the image file name (format '00-0000'):\\n")
          # 입력 형식 초기 설정, 하이픈 포함
          formatted_input = [" ", " ", "-", " ", " ", " ", " "] 
          stdscr.addstr("".join(formatted_input) + "\\r") # 초기 형식 화면에 표시
          stdscr.refresh()
  
          # 하이픈을 제외한 실제 입력 받을 위치
          cursor_positions = [0, 1, 3, 4, 5, 6]  # 하이픈을 제외한 입력 위치
          position_index = 0  # cursor_positions에서의 위치 인덱스
  
          while position_index < 6:  # 총 6개의 숫자 입력
              ch = stdscr.getch() # 키 입력 받기
              # 백스페이스 처리
              if ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
                  if position_index > 0:  # 입력된 위치가 있을 경우에만 백스페이스 처리
                      # 커서 위치를 하나 뒤로 이동
                      position_index -= 1  
                      # 커서 위치가 하이픈 바로 다음이면 하이픈 위치로 추가 조정
                      if cursor_positions[position_index] == 3:
                          position_index -= 1
                      # 이동한 위치의 문자를 공백으로 설정
                      formatted_input[cursor_positions[position_index]] = ' ' 
                      stdscr.addstr(1, 0, "".join(formatted_input) + "\\r")  # 화면 갱신
                      stdscr.move(1, cursor_positions[position_index])  # 커서 위치 조정
              # 숫자 입력 처리
              elif ch >= ord('0') and ch <= ord('9'):
                  # 입력 위치에 숫자 저장
                  formatted_input[cursor_positions[position_index]] = chr(ch)
                  position_index += 1
                  # 화면에 현재 형식 다시 표시
                  stdscr.addstr(1, 0, "".join(formatted_input) + "\\r")
                  # 다음 입력 위치로 커서 이동
                  if position_index < 6:
                      stdscr.move(1, cursor_positions[position_index])
  
              if position_index >= 6:  # 모든 숫자 입력 완료
                  break
  
          stdscr.refresh() # 화면 최종 갱신
          # 최종적으로 형성된 문자열 추출
          formatted_name = "".join(formatted_input).strip() 
          # 최종 입력된 이름 출력
          stdscr.addstr(3, 0, "Formatted input: " + formatted_name + "\\n")
          formatted_name = "".join(formatted_input).strip() + "_1"
          
      # 처리된 이름 반환
      return formatted_name
  
  def delete_last_image(code, directory, image_type):
      """
      가장 최근에 캡처된 이미지 파일을 삭제하는 함수
      """
      index = int(image_type) - 1
      last_image_code = code[index]
      if last_image_code:
          last_image_path = os.path.join(directory, f"{last_image_code}{image_file_extension}")
          if os.path.exists(last_image_path):
              os.remove(last_image_path)
              code[index] = ''  # 해당 코드 리셋
              saveImageInfo(*code)
              return True, f"Deleted image: {last_image_code}"
          else:
              return False, "Image file does not exist."
      return False, "No recorded image code to delete."
  
  def main_loop(stdscr):
      """
          메인 루프를 구성하여 사용자 입력에 따라 다양한 기능을 수행
      """
      code = readImageInfo()  # 현재 이미지 정보 읽기
      tpg_count = 1  # TPG 이미지 카운터
      
      while True:
          stdscr.clear()
          # 이미지 유형 선택 안내 메시지 출력
          stdscr.addstr("What kind of image will you take?\\n")
          # 각 키에 대한 설명 출력
          stdscr.addstr("Enter '1' for Swatch\\nEnter '2' for TPG\\nEnter '3' for TCX\\nEnter '0' to exit the program\\n")
          stdscr.refresh()
          
          # 사용자로부터 입력 받음
          image_type = stdscr.getkey()
  
          # 프로그램 종료 조건
          if image_type == '0':
              stdscr.addstr("Exiting program.\\n")
              stdscr.refresh()
              break
          
          directory = path_finder.get_directory(image_type)
          
          # 내부 루프 시작
          while True:
              stdscr.clear()
              stdscr.addstr("Enter '1' to preview\\nEnter '2' to capture an image\\nEnter '9' to reselect the image type\\nEnter '0' to quit\\n")
              stdscr.refresh()
  
              # 사용자 입력 받기
              cmd_input = stdscr.getkey()
              # 입력받은 이미지 유형에 따라 저장 경로 설정
  
              if cmd_input == '1':
                  # 미리보기 시작
                  process = subprocess.Popen(preview_command)
                  stdscr.addstr("Preview started. Press 'q' to stop.\\n")
                  stdscr.refresh()
                  # 미리보기 중지 대기 루프
                  while True:
                      stop = stdscr.getkey()
                      # 중지 키 입력 처리
                      if stop == preview_stop_key:
                          # 미리보기 프로세스 종료
                          process.terminate()
                          # 프로세스 종료 대기
                          process.wait()
                          # 중지 안내 메시지 출력
                          stdscr.addstr("Preview stopped.\\n")
                          stdscr.refresh()
                          break
  
              # 이미지 캡처 기능 선택
              elif cmd_input == '2':
                  if image_type == '2':
                      last_tpg_code = code[1].split('_')[0] if code[1] else "" 
                      image_name = validate_and_format_image_name(stdscr, image_type, last_tpg_code, tpg_count)
                      file_name = os.path.join(directory, f"{image_name}{image_file_extension}")
                      capture_command[2] = file_name
                      subprocess.run(capture_command)
                      code[1] = image_name
                      tpg_count = tpg_count + 1 if tpg_count < 8 else 1
                  else:
                      image_name = validate_and_format_image_name(stdscr)
                      file_name = os.path.join(directory, f"{image_name}{image_file_extension}")
                      capture_command[2] = file_name
                      subprocess.run(capture_command)
                      if image_type == '1':
                          code[0] = image_name
                      elif image_type == '2':
                          code[1] = image_name
                      elif image_type == '3':
                          code[2] = image_name
                      saveImageInfo(*code)
                  
                 
                 # 클라우드에 이미지 업로드
                  cloud_controller.upload_file(file_name, os.path.basename(file_name), image_type)
                
                  
              elif cmd_input == '5':
                  success, message = delete_last_image(code, directory, image_type)
                  stdscr.addstr(message + "\\n")
                  stdscr.refresh()
                  
              elif cmd_input == '9':
                  break  # 다시 유형 선택으로 돌아가기
  
              elif cmd_input == '0':
                  stdscr.addstr("Exiting program.\\n")
                  stdscr.refresh()
                  return  # 프로그램 종료
  
  if __name__ == "__main__":
      curses.wrapper(main_loop)  # curses를 사용하여 메인 루프 실행
  
  # ----------------------------------------------------------------------------- #
  ```

  추가한 기능은 다음과 같다.

  1. TPG 촬영 모드 시 8컷당 코드 입력란 한 번만 나오도록 수정

     TPG를 촬영할 때 하나의 코드 당 8개의 이미지를 촬영하기 때문에 코드를 입력하면 그 후 8번의 촬영에 대해 코드를 입력하지 않도록 한다. (특정 코드를 촬영 중 8장을 모두 촬영하지 않고 프로그램을 종료할 경우, 다시 8장 모두 재촬영해야 한다.)

  2. TPG 촬영 이미지에서 코드 당 추가 번호 부여

     TPG 코드를 입력하면, 8번째 이미지까지 구분할 수 있도록 추가번호가 자동으로 붙는다.

     ex) 코드 12-3456을 입력한 경우.

     첫 번째 이미지: 12-3456_1.jpg

     두 번째 이미지: 12-3456_2.jpg …

  3. 이전 촬영한 이미지 삭제 옵션

     촬영 실수에 대한 빠른 피드백을 위해 이전 촬영 이미지 삭제 기능을 추가한다. 직전 촬영 이미지만 삭제할 수 있다.

<br>

------

# 5. 향후 계획

## 1) CI 팀

> TPG 데이터셋이 완성되면 학습을 진행해 추가적으로 모델을 튜닝하고 Clustering과 Color Inference를 이미지 전치리를 통해 연결한다.

<br>

- [ ]  TPG 데이터셋 학습 및 튜닝
- [ ]  Clustering을 통해 색상 군집으로 분류된 cluster를 이미지 형식으로 변환
- [ ] Clustering 모델과 Color Inference CNN모델을 모듈화를 통한 연결

<br>

## 2) AC 팀

> 기존 Custum White Balance 코드를 기반으로 세부적인 색온도 보정 방안을 마련한다. MCU 생성 과정에서 loss 혹은 그 외의 문제 상황에 대해 적절히 처리할 수 있도록 알고리즘을 개선한다.

<br>

- [ ]  원단 색조에 따른 색온도 변화 조절(White Balancing)
- [ ]  (*BW*)AC 알고리즘 설계
- [ ] 조도 차이 보정(RGB)

<br>

## 3) 공통 계획

- [ ]  TPG 커팅
- [ ]  TPG 촬영

<br>