---
layout: single
title:  "Weekly Diary 9주차(2024.04.29 ~ 2024.05.05)"
excerpt: "9주차 개발 기록 정리"
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

## **활동 기록**

**[주간 정규 회의 및 활동 기록]**  
2024-04-29 (월) 12:30 ~> 13:30, 13:30 ~> 17:00  
2024-04-30 (화) 16:00 ~> 21:00  
2024-05-02 (목) **[11:00 ~> 12:00]**, 12:00 ~> 20:00  
총 활동 시간: 14시간 30분  
**대괄호로 작성된 부분은 정규 회의 시간임.*
{: .notice--alert .text-center}

<br>

------

# 2. CI 팀

## 1) CNN 모델 튜닝

> 지난 CNN테스트에서 인터넷에서 가져온 Dataset이 적합하지 않았다. 때문에 다른 적합한 Dataset이 있는지 조사해보았지만 조건에 맞는 Dataset이 존재하지 않았고 때문에 기존 아쉬웠던 4000장 가량의 이미지 Dataset을 눈으로 분석해서 사용하기에 적합한 이미지들만 분류해서 다시 CNN을 테스트해보려 한다.

기존 Dataset을 분석 결과 적합성이 너무 떨어져서 사용하기 적절치 않다고 판단했다.

그렇기 때문에 직접 촬영한 이미지 중 팬톤 포스트 카드 이미지가 50장 가량 있었고 그 이미지가 4056x3040 사이즈이기 때문에 이 사진을 중앙 부분 400x400으로 crop해서 최대한 조도가 비슷한 부분을 선택하고 한 장마다 다시 100x100 이미지로 나누어서 50장의 이미지를 800장으로 만들어서Dataset으로 사용했다.

<br>

**Test**

지난 테스트와 마찬가지로 labeling이 되어있는 많은 데이터가 없기 때문에 테스트 이미지 위에 cnn이 추출한 색상을 띄워 단순 육안으로 비교해보았다.

![image-20240505234315884](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234315884.png){: .img-default-setting}

**⇒ 기존 Dataset과의 비교 이미지**

![image-20240505234325920](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234325920.png){: .img-default-setting}

**⇒ Dataset 개선 후 비교 이미지**

<br>

- ***Dataset 개선 이후 추론된 색상과 원본 색상을 비교했을 때 더욱 비슷한 색으로 가까워진 결과를 확인했다.***

보다 나은 결과를 얻었기 때문에 지난번 큰 오차가 있었던 TPG에서도 더 좋은 결과를 얻을 것인지 확인을 위해 같은 방법으로 색상 추출을 해보았다.

![image-20240505234333617](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234333617.png){: .img-default-setting}

***⇒ 팬톤 17-3014를 촬영한 이미지***

- **팬톤 카드 레이블 값: [169,114,157]**
- **데이터 개선 이전 CNN이 추출한 대표 값: [130 101 101]**
- **데이터 개선 이후 CNN이 추출한 대표 값: [139  63 128]**

수치만 본다면 여전히 큰 차이가 있어 보이지만 색상으로 비교해본다면 다음과 같은 결과를 확인할 수 있다.

<br>

![image-20240505234344657](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234344657.png){: .img-default-setting}

*배경: TPG이미지, 우 하단 : 개선 이전 추론, 우 상단 : 개선 이후 추론*

![image-20240505234353551](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234353551.png){: .img-default-setting}

결과는 보이는 바와 같이 개선 이후에는 밝기의 차이만 존재할 뿐 색조는 일치하는 결과를 보였다. 하지만 추후 추가로 확인이 필요한 사항이 있었다. 레이블 값과 추론 값의 RGB차이가 고르지 않다는 점이었는데

<br>

**R:169-139=30, G:114-63=51, B:157-128=29**

다음과 같이 예상해볼 수 있다.  

**1) G값이 특수한 성질을 가지고 있다.**

**2) 보라색과 초록색(G)이 서로 보색 관계여서 그렇다.**

![image-20240505234401845](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234401845.png){: .img-default-setting}

두 경우 모두 확인을 위해서 TPG TestSet이 더욱 필요한 상황이지만 촬영된 이미지가 부족해서 역시 LNIC3.0을 통해 구매한 TPG북이 도착하면 이어서 진행하기로 하였다.

추가적으로 데이터 셋 임시 labeling 방식을 평균+밝기 가중치로 코드 수정해봤는데 모델의 학습이 이상하게 되는 문제 발생 밝기를 이전 과정에서 처리했다고 가정해서 하던지 아니면 코드를 추가 수정할 필요가 있을 것으로 보인다.

<br>

또한 800장 가량의 100x100 이미지를 10 Epoch 학습 시키는데 pc cpu기준 15분 정도(laptop에서는 1시간 정도)가 소요 되는 것을 확인하였고 지금도 모델 구조 튜닝이나 수정 후 재 학습에 20~25분(*laptop환경 고려 보수적 수치*)이 소요될 것을 고려하면 지금 당장의 비효율성이나 추후 데이터 셋의 규모가 더 커지거나 이미지의 크기가 커질 때 문제가 발생할 수 있기 때문에 moblienet 모델 적용을 확인해보아야 할 듯하다.

> 이 방법 역시 데이터의 수가 부족하기 때문에 추후 TPG를 촬영해서 사용하면 해결될 문제일 것으로 보인다.

<br>

## 2) CNN 모델 학습 속도 개선

> 지난 간단한 CNN 모델도 100x100 사이즈의 800장  DataSet을 학습하는데 pc성능 기준 10 epoch 학습에서도 15분 이상 소요되었다. 때문에 모델 튜닝 등에 너무 많은 시간이 소요되는 것을 방지하고자 **경량화 시스템에 적합한 MobileNet 모델 등 여러 모델들을 활용하여 학습 시간을 단축하고자 한다.**

<br>

**MobileNet**

MobileNet은 Google에서 개발한 경령화된 심층 신경망 아키텍쳐로, 모바일 및 임베디드 장치에서도 효율적으로 사용할 수 있도록 설계되어있는 모델이다. SImpleCNN에 비하여 작은 모델 크기와 적은 파라미터를 가지고 있어 저장 공간 및 메모리 사용량이 적어 임베디드 시스템에서도 빠르고 효율적인 추론이 가능하다.

<br>

> **torchvision의 mobilenet_V2 를 사용하여 테스트를 진행하였다.**

```python
from torchvision.models import mobilenet_v2

# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.mean(image, axis=(0, 1))
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32) / 255.0
        # 이미지의 채널을 3으로 변경 (그레이스케일 이미지에서 RGB로 변경)
        image = image.permute(2, 0, 1)  # [height, width, channels] -> [channels, height, width]
        return image, label

# 데이터로더 설정
dataset = CustomDataset(root_dir='train_set')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# MobileNetV2 기반의 모델 정의
class CustomMobileNet(nn.Module):
    def __init__(self):
        super(CustomMobileNet, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.mobilenet(x)

# 모델 및 손실 함수, 옵티마이저 설정
model = CustomMobileNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 모델 학습

for epoch in range(10):
    running_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/10, Loss: {running_loss/len(data_loader)}')
```

<br>

***⇒***

learing rate = 0.001

Epoch 1/10, Loss: 0.0777027748976252   
Epoch 2/10, Loss: 0.071246074722294  
Epoch 3/10, Loss: 0.06878780077007832  
Epoch 4/10, Loss: 0.06726623870548792  
Epoch 5/10, Loss: 0.06651387445628643  
Epoch 6/10, Loss: 0.06625950472312979  
Epoch 7/10, Loss: 0.06575135017803405  
Epoch 8/10, Loss: 0.06559909309085925  
Epoch 9/10, Loss: 0.06557650647591799  
Epoch 10/10, Loss: 0.066031929369492

![image-20240505234415230](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234415230.png){: .img-default-setting}

![image-20240505234431084](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234431084.png){: .img-default-setting}

<br>

***⇒***

learning rate = 0.0005

Epoch 1/10, Loss: 0.07629290627184673  
Epoch 2/10, Loss: 0.06960117996903137  
Epoch 3/10, Loss: 0.06704106572287856  
Epoch 4/10, Loss: 0.06489625137706753  
Epoch 5/10, Loss: 0.06630172903882339  
Epoch 6/10, Loss: 0.06514749591588043  
Epoch 7/10, Loss: 0.06434292380465195  
Epoch 8/10, Loss: 0.06463699442625512  
Epoch 9/10, Loss: 0.06450260689714923  
Epoch 10/10, Loss: 0.06493636634317226

![image-20240505234517597](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234517597.png){: .img-default-setting}

![image-20240505234524287](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234524287.png){: .img-default-setting}

<br>

MobileNet 모델로 진행한 결과 색 일치에 대한 정확성에 너무 낮은 결과가 발생하였다. 기존의 학습률을 조정을 진행하여도 마찬가지로 정확하지 못한 결과가 발생하여 MobileNet 모델 사용에 대해서는 더 많은 조사가 필요하다고 생각했다.

<br>

**ShuffleNet**

ShuffleNet은 또 다른 경량화 신경망 아키텍처로, 모바일 및 임베디드 환경에서의 효율적인 딥러닝 모델을 구축하기 위해 설계되었다. ShuffleNet은 효율적인 네트워크 구조와 채널 셔플링 메커니즘을 사용하여 작은 모델 크기와 적은 계산이 사용된다는 장점을 가지고 있다.

> Torchvision의 ShuffleNet을 사용하여 테스트를 진행하였다.

```python
#ShuffleNet CNN
from torchvision.models import shufflenet_v2_x1_0

# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.mean(image, axis=(0, 1))
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32) / 255.0
        # 이미지의 채널을 3으로 변경 (그레이스케일 이미지에서 RGB로 변경)
        image = image.permute(2, 0, 1)  # [height, width, channels] -> [channels, height, width]
        return image, label

# 데이터로더 설정
dataset = CustomDataset(root_dir='train_set')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# MobileNetV2 기반의 모델 정의

class CustomShuffleNet(nn.Module):
    def __init__(self):
        super(CustomShuffleNet, self).__init__()
        self.shufflenet = shufflenet_v2_x1_0(pretrained=True)
        for param in self.shufflenet.parameters():
            param.requires_grad = False
        num_ftrs = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.shufflenet(x)

# 모델 및 손실 함수, 옵티마이저 설정
model = CustomShuffleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(10):
    running_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/10, Loss: {running_loss/len(data_loader)}')
```

<br>

*⇒*

Epoch 1/10, Loss: 0.06898841288755647  
Epoch 2/10, Loss: 0.06363247000437695  
Epoch 3/10, Loss: 0.06043517287849681  
Epoch 4/10, Loss: 0.057936912499717434  
Epoch 5/10, Loss: 0.0551334055273037  
Epoch 6/10, Loss: 0.05287036096153315  
Epoch 7/10, Loss: 0.04943786088129855  
Epoch 8/10, Loss: 0.046692197056254374  
Epoch 9/10, Loss: 0.04532296615809173  
Epoch 10/10, Loss: 0.044707377956365240

![image-20240505234555023](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234555023.png){: .img-default-setting}

![image-20240505234601721](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234601721.png){: .img-default-setting}

| 기존 DataSet 기준 (m) | PC   | Laptop | Colab |
| --------------------- | ---- | ------ | ----- |
| Simple_CNN            | 15   | 29     | 59    |
| MobileNet_v2          | 3    | 6      | 10    |
| ShuffleNet            | 5    | 13     | 20    |

v2는 빠른 학습 시간을 보여주었지만 성능이 좋지 않았고 v3는 이론적으로는 v2보다 빠르며 성능이 좋아야 하지만 오히려 느리면서 성능 역시 들쭉날쭉한 결과를 보였다.

ShuffleNet에서도 MobileNet과 마찬가지로 색 일치 정확성이 낮은 결과가 발생하였다. 두 모델 모두 정확성이 떨어지는 결과를 보였기 때문에 Torchvision에서 불러오는 Mobilenet이 아니라 자체적으로 구현된 코드를 조건에 맞춰 튜닝 하거나 기존 simple_CNN을 개선하는 방향으로 진행하고자 한다.

<br>

## 3) DBSCAN for CNN

> 기존의 DBSCAN 알고리즘을 적용하여 대표 색상 클러스터를 추출한 후 각 색상 클러스터 별 실제 이미지에서의 해당 픽셀들을 떼어낸 후 각각의 새로운 단색 이미지들을 생성하고 대표 색상을 추론하는 CNN 과정을 연계하여 진행하고자 한다.

테스트에서는 임의의 tpg 촬영 이미지를 DBSCAN을 이용하여 4개의 색상으로 클러스터링 하였고 각각 클러스터링된 부분에 대해 실제 이미지의 해당 위치 픽셀들을 1개씩 나란히 배열한 후 가로 세로의 길이가 같은 형태의 1개의 이미지로 만들어 보았다.

```python
# 대표 색상 추출
cluster_centers = np.array([np.mean(color_tbl[labels == label], axis=0) for label in set(labels)])

# 클러스터된 색상 이미지 출력
cluster_image = cluster_centers[labels].reshape(img.shape).astype(int)

# 대표 클러스터 색의 RGB 값 출력 및 텍스트로 이미지 위에 표시
for label, color in enumerate(cluster_centers):
    if label == -1:
        continue
    print(f"Cluster {label} RGB: {color.astype(int)}")
    
    # 클러스터된 색상 영역 분리
    color_mask = np.all(cluster_image == color.astype(int), axis=-1)
    segmented_image = np.zeros_like(img)
    segmented_image[color_mask] = img[color_mask]
    
    # 색상 영역 이미지 출력
    plt.figure(figsize=(6, 6))
    plt.imshow(segmented_image)
    plt.title(f"Cluster {label} RGB: {color.astype(int)}")
    plt.axis('off')
    plt.show()

    # 클러스터 영역의 픽셀 위치 찾기
    cluster_pixels = np.argwhere(color_mask)
    
    # 클러스터 영역에서의 실제 픽셀 값들 추출하여 리스트에 저장
    cluster_pixels_values = [img[pixel[0], pixel[1]] for pixel in cluster_pixels]
    
    # 픽셀 값들을 정사각형 모양의 이미지에 출력
    num_pixels = len(cluster_pixels_values)
    num_cols = int(np.ceil(np.sqrt(num_pixels)))  # 이미지에 출력할 열의 개수
    num_rows = int(np.ceil(num_pixels / num_cols))  # 이미지에 출력할 행의 개수

    # 이미지를 그리드 형태로 나열
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for i, pixel_value in enumerate(cluster_pixels_values):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.imshow([[pixel_value]])
        ax.axis('off')

    # 빈 이미지 슬롯 채우기
    for i in range(num_pixels, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')

    plt.suptitle(f"Cluster {label} Pixels", fontsize=16)
    plt.tight_layout(pad=0)  # 간격 조정
    plt.show()
```

![image-20240505234625146](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234625146.png){: .img-default-setting}

![image-20240505234633964](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234633964.png){: .img-default-setting}



![image-20240505234642995](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234642995.png){: .img-default-setting}

![image-20240505234647847](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234647847.png){: .img-default-setting}

CNN을 적용하는데 있어서 이상이 없을 정도의 깔끔한 형태의 이미지들이 나왔다. 다만 **각각의 이미지의 픽셀의 개수가 모두 다르기 때문에 모든 이미지의 크기가 같지 않다는 문제점이 생길 수 있었다**. 따라서 위 문제에 대한 해결 방법에 대해 생각해보고 하나씩 적용하면서 최적의 알고리즘을 찾아보기로 하였다.

<br>

1. **이미지 크기 통일화**: 모든 클러스터에 대해 이미지 크기를 동일하게 조정한다.
2. **이미지 패딩**: 이미지의 크기를 조절하여 모든 이미지의 크기를 동일하게 만들 때 패딩을 사용한다.
3. **이미지 데이터 확장**: 이미지 데이터를 확장하여 모든 이미지의 크기를 통일화한다.
4. **이미지 잘라내기**: 모든 이미지를 동일한 크기로 자르거나 잘라내어 픽셀 수를 조정한다.
5. **CNN 모델의 입력 처리**: CNN 모델을 사용할 때 입력 이미지의 크기가 동일해야 한다. 따라서 모든 이미지를 동일한 크기로 조정하거나, 모델에 맞게 입력 이미지를 처리한다.

<br>

------

# 3. BWAC 팀

## 1) LINC 3.0 주문

<br>

- **TPG**

LINC 3.0 지원을 통해 추가 예산을 확보하였으며, 이를 통해 시스템 학습을 위한 데이터셋을 마련하기로 하였다. 이전 주차 활동에 이어 **배치 선형 계획법**을 중심으로 구매할 TCX의 Code를 정하기 위해 지속적으로 논의를 진행하였다.

배치 선형 계획법은 원단에 발생하는 지역적 조도 차이는 색조와 관련되지 않고 오로지 원단의 밝기에만 관련이 있다고 가정한 채 설계되었는데, 문제는 이러한 설계가 정말로 시스템의 정확도를 측정하기에 적합한 방식인지 확신할 수가 없다는 것이다. 이에 따라 기존에 학습 데이터셋으로 활용하기로 했던 TPG를 먼저 마련하여 배치 선형 계획법의 가정을 실험적으로 확인하고자 하며, 따라서 TCX의 구매는 이후로 미뤘고, LINC 3.0을 통해 TPG를 구매하였다.

<br>

- **LCD**

시스템 동작 및 결과 제시를 위한 LCD를 구매하였다. 이를 통해 사용자가 시스템과 interaction 할 수 있도록 한다.

> 현재 TPG와 LCD가 도착하였으며, 다음 주 활동에서 TPG를 통해 데이터셋을 생성하고 분석하여 최종적으로 TCX를 주문하도록 한다. 또한 시스템 H/W에 LCD를 부착하여 interaction 할 수 있도록 프로그램을 제작한다.

<br>

## 2) WB(White Balancing) 알고리즘 설계

> 현재 시스템에서 사용하는 이미지 센서의 특성 상, 특정 물체를 촬영할 때 색조 변화 현상이 발생하는 것을 확인했으며, 라이브러리 함수를 통해 Auto White Balancing을 Manual하게 지정하였음에도 이러한 현상을 통제할 수가 없었다. 이런 현상을 통제하기 위해 별도의 White Balancing 알고리즘을 설계하고 실험을 진행하였다.

### 1. 분석 코드 및 설명

**이전 7주차 보고서에서 White Balancing 알고리즘으로 Outer Area의 Mean Value를 바탕으로 한 Custom Algorithm을 제시했었다.** 이 알고리즘을 구현하고, 현재 가지고 있는 간단한 데이터셋에 적용하여 전반적인 성능을 확인해보았다.

분석을 위해 생성한 코드는 기존에 작성한 코드와 특성이 아주 다르다고 판단되어, 별도의 디렉터리를 생성한 후 보관하였다. (*/home/pi/project/analysis*)

<br>

- **코드 동작 원리 설명**

  - **calcStandardRGB.py**: Swatch Area 위에 아무 것도 올리지 않은 상태로 이미지를 촬영한 다음, Left Outer Area와 Right Outer Area의 모든 픽셀에 대한 RGB 평균을 구한다. ⇒ *이때, Left Mean, Right Mean을 먼저 구하며, 최종적으로 10개의 바닥 이미지를 촬영한 다음 Standard RGB값을 생성했다.*
  - **calcOuterRGB.py**: start_num부터 end_num까지 반복하며 각 이미지의 Outer Area RGB를 구하고 출력한다. 해당 값들은 아래 Pantone Post Card에 대한 Label에 정리되어 있다.
  - **calcOuterDiff.py**: start_num부터 end_num까지 반복하며 각 이미지에 대해 Outer Area의 RGB와 Standard RGB 값과의 차이(***Diff***)를 구하고, 이를 Target Area의 평균값에 적용해 Label과의 차이를 구한다. 이후에는 ***Diff*** 만큼 전체 픽셀에서 뺀 이미지를 새롭게 저장한다. ⇒ *hue_corrected 디렉터리에 저장한다.*
  - **calcOuterSilmilarity.py:** Target Label과 Target Area Value, Target Label과 Corrected Target Area Value를 각각 유사도 비교 및 출력

  > 본 코드가 시스템 구현에 매우 중요한 코드는 아니며, 특별히 설명한 사항도 없기 때문에 9주차 보고서에 첨부하지는 않고 깃허브에 업로드 하였음.

<br>

### 2. 분석 데이터의 정보 및 Label

> WB 알고리즘에 적용할 데이터의 정보이다. xx-xxxx는 색상 코드이며, 뒤에 나오는 숫자는 RGB이다.

- 18-2326: 168, 62, 108
- 19-3438: 121, 67, 132
- 14-1419: 255, 178, 165
- 19-3935: 64, 68, 102
- 19-6110: 55, 65, 58
- 17-5722: 67, 125, 109
- 18-5621: 59, 114, 95
- 14-0852: 243, 193, 44
- 18-4525: 0, 129, 157
- 13-1107: 219, 203, 190
- 15-0543: 181, 182, 68
- 17-1564: 221, 65, 50
- 19-2025: 124, 41, 70
- 19-3921: 43, 48, 66
- 17-1456: 226, 88, 62
- 16-1439: 195, 124, 84
- 13-5304: 207, 200, 189
- 18-4530: 0, 99, 128
- 17-1436: 185, 113, 79
- 12-5204: 207, 223, 219
- 19-1250: 141, 63, 45
- 18-4330: 0, 126, 177
- 18-3025: 148, 78, 135
- 18-1248: 181 90 48

<br>

### 3. 분석 결과

- **1차 Standard Value 생성**

> 지난 정규회의에서, 원단 이미지를 촬영하기 위해 사용하는 [Capture.py](http://Capture.py) 프로그램에서 원단이 다소 어둡게 촬영된다는 Feedback이 있었음. 또한 rpicam documentation을 통해 카메라가 여러가지 설정을 자동적으로 바꿀 가능성이 있다고 판단, ISO, 대조, 채도를 항상 기본값으로 고정하고 Shutter Speed를 기본값인 10,000에서 14,000으로 변경하여 이미지를 밝게 촬영하도록 하였음.

**Standard Value**: 235 233 238

⇒ 이 값은 Swatch Area 위에 아무 것도 올리지 않은 상태로 촬영한 10개의 이미지에서, Left Outer Area와 Right Outer Area의 RGB 평균을 구한 값이다. 우리 시스템은 통제된 환경이기 때문에 촬영되는 환경이 변화되지 않는다. 따라서 이와 같이 WB을 위한 기준값을 설정한 후, 색조 변화가 발생했을 때 Outer Area에서 색이 바뀌는 정도를 측정하여 Manual하게 White Balancing을 시도하고자 한다.

<br>

- **1차 분석 결과에 제시된 값의 의미**
  - **Total Value**: Swatch를 올린 상태에서 얻어낸 Outer Area의 평균 RGB
  - **Diff with Standard**: 위에서 구한 Standard Value(235 233 238)와 Total Value의 차이
  - **Target Label**: Swatch의 실제 색상 Label (RGB)
  - **Target Area Value**: Swatch를 촬영했을 때 Target Area의 RGB를 단순평균 낸 값
  - **Corrected Target Area Value**: Target Area의 모든 픽셀에 Diff를 뺀 후 다시 Target Area Value를 구한 값(White Balancing 된 값)
  - **Diff with Label**: Target Label과 Target Area Value의 차이
  - **Diff with Label after Correction**: Target Label과 Corrected Target Area Value의 차이

<br>

- **1차 분석 대표 이미지**

> 1차 분석은 이미지가 23장으로 매우 많기 때문에 대표 이미지 몇 장만 첨부하고, 그 이외의 이미지는 1차 분석 결과로 대체하도록 한다.

<br>

**6번 이미지**

![image-20240505234801883](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234801883.png){: .img-default-setting}

<br>

**18번 이미지**

![image-20240505234815205](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234815205.png){: .img-default-setting}

<br>

- **1차 분석 결과**

![image-20240505234922385](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234922385.png)

![image-20240505234932344](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234932344.png)

![image-20240505234940816](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234940816.png)

![image-20240505234950718](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234950718.png)

![image-20240505234959992](/images/2024-05-05-Weekly Diary(9주차)/image-20240505234959992.png)

![image-20240505235007319](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235007319.png)

<br>

1차 분석 결과, WB 적용 후의 이미지는 대체적으로 색조 변화가 통제된 모습을 보였음.

아직 모든 색상에 대해서 촬영해본 것이 아니기 때문에 좀 더 확인해 보아야겠지만, 색조 변화는 일부 색조(Hue)에 대해서만 나타나는 것으로 보이며, **대체로 R값이 매우 강해지거나 매우 약해지는 현상이 두드러지게 나타나는 것으로 확인**했다. 이러한 결과를 기억해 두었다가 이후에 TPG를 통해 모든 색조에 대해서 분석한 후, 어떤 색조의 원단에 대해서 발생하는 것인지 일반화 된 결론을 내고자 한다.

추가적으로, (수치화까지는 하지 못했지만) Label과 값이 너무 다르게 촬영되는 원단들이 있는데, 이것은 대체로 녹색 계열의 원단인 것으로 추정된다. 이 부분도 TPG를 분석을 통해 자세히 알아보아야 할 것이다.

그런데 현재 일부 원단에서 원단의 색상이 과도하게 밝게 촬영되고, 이 때문에 Label과 RGB값이 매우 다르게 나타나는 현상을 보이고 있다. 이는 Shutter Speed를 10,000에서 14,000으로 상승시킨 것이 원인인 것으로 추정되어 Shutter Speed를 기본값인 10,000으로 되돌린 후 2차 분석을 실시해 보았다.

<br>

- **2차 분석 결과**

**Standard Value: 214 211 217**

> Shutter Speed를 감소시켜 이전보다 Standard Value의 값이 감소하였다.

![image-20240505235029686](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235029686.png)

![image-20240505235038137](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235038137.png)

![image-20240505235045585](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235045585.png)

![image-20240505235058208](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235058208.png)

![image-20240505235106566](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235106566.png)

![image-20240505235114038](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235114038.png)

<br>

2차 분석 결과, Shutter Speed를 증가시켜 이전보다 이미지가 밝게 찍히는 정도는 감소하였다. 이에 따라 1차 분석보다 Label과의 차이가 대체적으로 감소한 것으로 보인다. 그러나 현재 단계에서는 어떤 방식이 좋은지 결론을 내리기는 이른 것으로 보이며, 결론적으로 어떤 방법을 선택할 것인지 확정하기 위해서는 **White Balancing의 성능을 측정해 보아야 할 것으로 생각된다.** 따라서 다음 주차에 WB 성능 측정을 위한 지표를 마련하고, TPG를 촬영한 후 모든 이미지에 대해서 적용해본 뒤에 최종적으로 방법론을 결정하고자 한다.

<br>

### 4. 면적 가중 WB

시스템 설계 과정에서 미처 Left Outer Area와 Right Outer Area의 면적을 동일하게 맞추지 못했는데, 이 때문에 Standard Value를 결정하는 과정에서 면적에 비례한 가중치를 반영하는 것이 더 좋은 결과를 낼 수 있을 것이라는 의견이 제시되었다. 이에 따라 Standard Value를 생성할 때 Left Outer Area와 Right Outer Area의 픽셀 수에 대비하여 1:2의 비율을 반영하고 새롭게 분석을 시도해 보았다.

<br>

- **면적 가중 WB 분석 결과**

그러나 기대와는 다르게 유의미한 차이는 존재하지 않았다. 2차 분석과 비교할 때 R, G, B 값에서 1 내지 2 정도의 값의 차이는 존재하였으나, 그것이 유의미한 차이를 보인다고 판단할 수준은 아니었기 때문에 우선 면적 가중 방식은 배제하기로 하였다. 그러나 자세한 사항은 TPG 데이터를 생성한 후, 정확한 성능 지표를 만든 후에 적용해 보아야 알 수 있을 것이다.

<br>

## 3) 유사도 비교 알고리즘 설계를 위한 분석

> WB의 성능지표를 만들기 위해서는 촬영된 이미지의 RGB가 Label RGB와 얼마나 가까워지는 지를 체크하는 것이 가장 좋다고 생각한다. 이에 따라 유사도 비교 알고리즘을 설계하기 위해 색상의 유사도를 나타내는 방법에 대해서 분석해 보았다.

주제 신청 보고서를 작성할 당시 조사한 내용에 따르면, 현 시점까지 인간의 시각적 인지를 반영하는 가장 적절한 색공간 시스템은 CIELAB2000이라고 평가된다. CIELAB2000에서 유사도 값에 대한 기준은 일반적으로 다음과 같다고 평가된다.

<br>

- **유사도 값(ΔE)에 대한 기준**
  - 0~1: 색이 매우 유사 (인간의 눈으로는 거의 감지할 수 없음.)
  - 1~2: 전문가가 감지할 수 있는 매우 미세한 차이
  - 2~10: 일반적인 관찰자도 감지할 수 있는 색상차이
  - 10 이상: 명확하게 구별되는 색상차이

> CIELAB2000의 최솟값은 0(색이 동일함)이고, 최댓값은 약 100이다. (0,0,0과 255,255,255의 차이는 100.00000017602524이다. 사실상 100이라고 봐도 될 것으로 생각된다.

<br>

색의 유사도(색차)를 측정하는 방법에 대해서 이해하기 위해, CIELAB2000, Euclidean Distance 방식으로 색상 비교 실험을 진행해 보았다. 왼쪽 이미지는 120, 100, 80으로 원본 이미지이고, 오른쪽 이미지는 원본 이미지를 약간씩 변형시킨 이미지이다.

<br>

- **1) 비율을 유지한 채로 RGB 변화**

![image-20240505235147467](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235147467.png){: .img-default-setting}

색은 RGB의 비율로 정해지며, 비율이 같으면서 RGB의 크기가 달라질 경우 밝기가 달라지는 것이라고 판단된다. RGB의 비율을 유지한 채로 밝기만 감소시켰다.

1. ΔE(*CIELAB2000*): 1.9755588150789245
2. L2(*Euclidean Distance*): 8.774964387392123

⇒ 색조는 동일하지만, ΔE값이 2.23인 것으로 보아 차이가 두 색상이 시각적으로 차이가 큰 것으로 보인다. 육안으로도 구분이 가능하기 때문에 차이가 분명히 있다고 판단된다.

<br>

- **2) RGB에 대해 동일한 임의의 값을 보정**

![image-20240505235200580](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235200580.png){: .img-default-setting}

1. ΔE(*CIELAB2000*): 1.9068574249080552
2. L2(*Euclidean Distance*): 8.660254037844387

⇒ 이전과 비교할 때, 오른쪽 이미지의 RGB 값이 크게 달라지지는 않았으며, L2 값 또한 거의 유사하지만 소폭 감소하였고, CIELAB2000 값 또한 소폭 감소하였다.

<br>

- **3) RGB에 대해 값을 소폭 보정**

![image-20240505235211886](/images/2024-05-05-Weekly Diary(9주차)/image-20240505235211886.png){: .img-default-setting}

1. ΔE(*CIELAB2000*): 2.837034591810601
2. L2(*Euclidean Distance*): 3.0

⇒ 이전과 비교할 때, 오른쪽 이미지의 변화량이 작아서 L2 값이 매우 작게 나왔지만, CIELAB2000의 값은 매우 크게 도출되었다.

<br>

위의 결과를 바탕으로, 팀원들에게 RGB 값을 제공하지 않고 관능평가(*육안을 통한 구분 실험*)를 실시하여 **왼쪽 이미지와 색상이 가장 유사해 보이는 것으로 판단되는 이미지를 순서대로 매겨보았다.** 결과는 다음과 같다.

- **참가자 1**: 1), 2), 3)
- **참가자 2**: 2), 1), 3)
- **참가자 3**: 2), 1), 3)
- **참가자 4**: 2), 1), 3)

간단한 관능평가 실험을 진행해 본 결과, 많은 인원이 실제로 CIELAB2000의 값이 낮은 순서대로 유사한 색깔을 지정한 것을 확인했다. 참가자 1번의 경우 1), 2), 3)으로 순서를 매기긴 하였으나, 두 값의 차이가 0.07 수준으로 매우 미미하기 때문에 주관에 의해 발생할 수 있는 차이로 보인다.

<br>

본 실험 결과가 완전히 객관적인 결과라고 확신할 수는 없겠지만, 값의 차이가 가장 적었던 것이 항상 사람의 눈에 유사해 보이는 것은 아니며, 이러한 측면에서 **Euclidean Distance가 색차에 대해 정확한 결과를 내기는 어렵다는 결론을 잠정적으로 내릴 수 있었다.** 이는 색이 정해지는 원리가 RGB의 비율에 기인하기 때문인 것으로 생각되며, RGB의 절댓값의 차이가 큰 것보다 RGB의 비율이 다른 것이, 사람의 눈에는 더 다른 색으로 인식될 수 있는 것으로 생각된다. (*이후에 좀 더 정밀한 실험을 통해 더 정확한 결론을 내릴 것이다.*) 또한, **본 실험의 결과를 바탕으로 향후에 TPG 데이터에 대한 알고리즘의 성능 평가에 CIELAB2000을 중점적으로 활용하고자 한다.**

<br>

------

# 4. 향후 계획

## 1) CI 팀

> 아직 진행하지 못한 TPG DataSet 제작을 우선적으로 진행하며 다른 모델들의 추가 테스트가 아닌SImple_CNN에 대한 최적화 진행을 중점적으로 진행하고자 한다. 또한 원활하고 정확한 CNN 적용을 위한 DBSCAN 이후의 과정에 대한 알고리즘 설계에 대해 진행하고자 한다.

- [ ]  SimpleCNN 최적화
- [ ] DBSCAN for CNN 알고리즘 개선 및 추가 설계

<br>

## 2)  BWAC 팀

> TPG DataSet을 바탕으로 가장 적절한 WB 알고리즘과 BWAC 알고리즘을 설계한다.

- [ ]  WB 알고리즘 설계 및 성능 평가
- [ ] BWAC 알고리즘 설계 및 성능 평가

<br>

## 3) 공통 계획

- [ ] TPG DataSet 생성
- [ ] LCD GUI 코딩

<br>

