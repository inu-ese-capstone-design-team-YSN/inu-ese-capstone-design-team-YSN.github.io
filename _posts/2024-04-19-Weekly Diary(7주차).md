---
layout: single
title:  "Weekly Diary 7주차(2024.04.15 ~ 2024.04.21)"
excerpt: "7주차 개발 기록 정리"
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

# **1. 주간 활동 정리**

## **1) 활동 기록**

**[주간 정규 회의 및 활동 기록]**  
2024-04-15 (월) **[12:30 ~ 13:30]**, 13:30 ~ 20:30      
2024-04-16 (화) 14:00 ~ 22:00  
2024-04-17 (수) 15:00 ~ 22:00   
2024-04-11 (금) 10:00 ~ 18:00(*Calibration 팀*)   
총 활동 시간: 31시간    
**<u>대괄호로 작성된 부분은 정규 회의 시간임.</u>*  
{: .notice--danger .text-center}

목요일 정규 회의는 논의할 사항이 많지 않다고 판단하여 진행하지 않았음. 이에 따라 각 팀 별로 활동을 진행함.

<br>

## 2) 개발 기록

- **Clustering 팀**: DBSCAN에서의 문제점 해결 및 속도 개선
- **Calibration 팀**: 프로젝트 기반 프로그램 작성, 데이터 촬영과 분석, Calibration 알고리즘 설계

<br>

# 2. Clustering 팀

## 1) DBSCAN에서의 문제점 해결

> 지난 6주차에 구성된 H/W환경에서의  촬영된 이미지를 통하여 Colab에서 DBSCAN 테스트를 실행했을때  시스템 RAM을 초과 사용해 세션이 종료되는 문제가 있었다.  실제 라즈베리파이 환경에서도 적용해 본 결과 ***마찬가지로 CPU 사용률과 메모리 초과로 인하여 세션이 종료되는 현상***이 발생하였다.

먼저 DBSCAN 테스트를 통해 확인된 문제 해결을 위해 고려해보았던 아래의 5가지 방법 중 진행하였던 2가지 방법을 정리하여 보면 다음과 같다.

1. 이미지 crop을 통한 저해상도 이미지에서의 DBSCAN

   ***⇒ 각 클러스터의 대표값으로 나타나게 했을때 클러스터링이 잘 되지 않은 문제점을 확인했다. 여러 다른 색상으로 진행했을 때도 비슷한 결과를 얻었다. 확실하게 구분이 가능한 다색상에서만 어느정도 준수한 결과를 확인할 수 있었다.***

2. K-means 알고리즘만 사용

   ***⇒ 원사 사이사이의 부분을 DBSCAN보다 확연하게 보일 정도로 잘 구분하는 결과를 얻었으며 또한 앞서 언급된 문제였던 고해상도 이미지의 DBSCAN적용의 어려움을 K-means 알고리즘을 사용하면서 해결 할 수 있었다. 하지만 여전히 클러스터의 개수를 미리 지정 해야만 구현 가능하다는 문제점이 있었다.***

   따라서 이번 주차에서는 아래의 3번째 방법을 구현해보고 나머지 2가지 방법에 대해서도 확인해 보기로 하였다.

3. **저해상도 DBSCAN으로 클러스터의 개수만 추출하고 K-means 알고리즘을 사용**

> 이 방법은 DBSCAN을 통하여 구분된 클러스터의 개수를  확인하고 K-means 알고리즘에 인자로 넘겨주는 방식이기 때문에 2번의 방식에서 클러스터의 개수를 정하지 않고 DBSCAN이 정해준다는 차이점이 있다.

<br>

- **라이브러리 import**

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans
from skimage.transform import resize
```

<br>

- 고해상도의 이미지를 불러온 후 **OpenCV의 Resize함수를 통한 저해상도 변환**

```python
# 파일이 위치한 경로 설정
file_path = os.path.join("Color", "color.png")
# png 이미지 불러오기
img = plt.imread(file_path)
# 이미지 크기 출력
print("Original Image Size:", img.shape)
# 이미지를 200x200으로 리사이징
resized_img = resize(img, (200, 200))
# matplotlib으로 리사이징된 이미지 시각화
plt.imshow(resized_img)
plt.show()
# 리사이징된 이미지의 크기 출력
print("Resized Image Size:", resized_img.shape)
```

*⇒* `Original Image Size: (984, 678, 3)`*⇒*  `Resized Image Size: (200, 200, 3)`

![image-20240420233524695](/images/2024-04-19-Weekly Diary(7주차)/image-20240420233524695.png){: .img-default-setting}

- **RGB값 추출**

```python
# 이미지 픽셀 값을 0에서 255 사이의 정수로 변환
resized_img = (resized_img * 255).astype(int)
# 이미지 배열을 2차원 배열로 변환
color_tbl = resized_img.reshape(-1, 3)
# 변환된 배열을 Pandas의 데이터프레임으로 변환되며 각 열은 RGB 각각의 값으로 레이블링
df = pd.DataFrame(color_tbl, columns=["Red", "Green", "Blue"])
print(df)
```

*⇒*

```python
       Red  Green  Blue
0      138     57    45
1      150     63    50
2      159     66    51
3      167     70    54
4      166     65    53
...    ...    ...   ...
39995  145     51    32
39996  149     54    35
39997  158     63    43
39998  158     63    43
39999  153     58    40

[40000 rows x 3 columns]
```

<br>

- **DBSCAN**을 통한 클러스터링 개수 **K 값 추출**

```python
# DBSCAN 모델 생성
# eps는 클러스터의 반경을 결정하고 min_samples는 클러스터를 형성하기 위한 최소 데이터 포인트의 수다.
dbscan = DBSCAN(eps=3, min_samples=50)
# 모델 fitting
dbscan.fit(color_tbl)
# 클러스터 레이블
labels = dbscan.labels_
# 대표 색상 추출
unique_labels = set(labels)
# 클러스터링된 개수 K 출력
K = len(unique_labels)
print("클러스터링된 개수:", K)
# 대표 색상에 해당하는 색상 팔레트 생성
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
```

⇒  클러스터 개수: 2

<br>

- **3D 그래프 구현**

```python
from mpl_toolkits.mplot3d import Axes3D

#3D 그래프를 생성하기 위한 figure와 Axes 생성
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# 각 클러스터에 대한 색상과 데이터 포인트를 3D 그래프에 추가
for k, col in zip(unique_labels, colors):
    # Noise인 경우 제외하고 진행
    if k == -1:
        continue  # Noise는 제외
    # 현재 클러스터에 속한 데이터 포인트를 마스킹하여 선택
    class_member_mask = (labels == k)
    xy = color_tbl[class_member_mask]
    
    # 각 클러스터에 속한 색상들의 평균 값 계산
    representative_color = np.mean(xy, axis=0)
    # 클러스터에 속한 데이터 포인트를 3D 그래프에 scatter plot으로 추가
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=tuple(col), marker='o', s=30)
    ax.scatter(representative_color[0], representative_color[1], representative_color[2], c=tuple(col), marker='o', s=200,
               edgecolor='k', label=f'Cluster {k}')  # 테두리 색을 검정색으로 설정
    
    # 대표 클러스터 색의 RGB 값 출력
    print(f"Cluster {k} RGB: {representative_color.astype(int)}")

# 축 레이블과 그래프 제목 설정
ax.set_xlabel('L')
ax.set_ylabel('a')
ax.set_zlabel('b')
plt.title('Cluster Centers in Lab Color Space')

# 범례 표시
ax.legend()
# 그래프 출력
plt.show()
```

<br>

- **K-means 알고리즘에 DBSCAN을 통해 구한 K값 적용 및 이미지 시각화**

```python
# KMeans 객체 생성 및 군집화 수행
n_clusters = K
km = KMeans(n_clusters=n_clusters)
km.fit(color_tbl)
# 클러스터 중심점 추출
km.cluster_centers
# 클러스터 중심점 시각화
plt.imshow(km.cluster_centers_.reshape(K,1,3).astype(int))
# 클러스터 중심점 가져오기
cluster_centers = km.cluster_centers_
# 클러스터 중심점 시각화
plt.figure(figsize=(10, 5))
for i in range(len(cluster_centers)):
    plt.subplot(1, len(cluster_centers), i+1)
    plt.imshow(cluster_centers[i].reshape(1, -1, 3).astype(int))
    plt.axis('off')
plt.show()
# 각 픽셀에 할당된 클러스터 중심점을 사용하여 이미지 재구성
color8 = km.cluster_centers_.astype(int)[km.labels_].reshape(resized_img.shape)
# 재구성된 이미지 시각화
plt.figure(figsize=(100, 100))  # 이미지 크기 조절
plt.imshow(color8)
```

⇒

![image-20240420233551295](/images/2024-04-19-Weekly Diary(7주차)/image-20240420233551295.png){: .img-width-large}

<br>

- **기존 이미지에 클러스터링된 색 매핑**

```python
color8 = km.cluster_centers_.astype(int)[km.labels_].reshape(resized_img.shape)
plt.figure(figsize=(100, 100))  # 이미지 크기 조절
plt.imshow(color8)
```

*⇒*

![image-20240420233606527](/images/2024-04-19-Weekly Diary(7주차)/image-20240420233606527.png){: .img-default-setting}

위 코드를 구현하면서 DBSCAN에서의 고해상도의 이미지에서 저해상도 이미지로 변환하는 방법에 대해 생각하다 우선적으로 간단하게 테스트 해 볼 수 있는 ***OpenCV의 Resize함수***를 사용해 보았다.

Resize 함수는 주어진 이미지를 새로운 크기로 변경하는데, ***보간(interpolation) 방법***을 사용하여 이미지의 픽셀 값을 조정한다. 이 과정은 주어진 이미지의 픽셀 값을 새로운 이미지의 픽셀 값으로 매핑하는 단순한 크기 변환 과정으로 이미지의 원본 픽셀 값들을 직접적으로 조작하지 않지만 보간 과정에서 ***픽셀 값들을 추정하고 조합하는 방식 때문에 원본 이미지의 세부 정보가 손실될 수 있다는 문제점***을 확인할 수 있었다.

따라서 추가적인 대안에 대해 찾아 보다가 ***컨볼루션(Convolution) 방법*** 을 사용하기로 하였다

컨볼루션은 이미지 필터링, 특징 추출 등 다양한 이미지 처리 작업에 사용되는 기술로 주어진 이미지와 필터(커널) 사이의 합성곱 연산을 의미한다. 필터는 이미지의 패턴이나 특징을 감지하는데 사용되며 ***각 픽셀에 필터를 적용하여 새로운 픽셀 값을 생성하는 방식을 통하여 이미지의 세부 정보를 보다 정확하게 보존하고, 특정한 목적을 위해 필요한 정보를 추출하는 데에 더 효과적*** 이라는 사실을 알 수 있었다.

<br>

> 추가적인 테스트로 원격 접속을 통해 실제 pi 환경에서 DBSCAN을 동작시키며, 어느 정도의 저해상도 이미지가 적절한 지에 대한 간단한 테스트도 진행해 보았다.

- **단색상 흰색**으로 테스트

![image-20240420233623151](/images/2024-04-19-Weekly Diary(7주차)/image-20240420233623151.png){: .img-default-setting}

*⇒  400x300: 커널이 종료됨.*

<br>![image-20240420233651655](/images/2024-04-19-Weekly Diary(7주차)/image-20240420233651655.png){: .img-default-setting}

*⇒ 300x200: 정상적으로 실행되는것 확인*

<br>

실행 시간은 평균 6초이며 클러스터링 결과 부분적 어두운 부분을 잘 분리하지 못하였는데 아마 저화질 이미지이긴 하나 단순 확대만 된 이미지이기 때문에 이런 결과가 나온 것으로 예상된다. 추가적으로 다양한 해상도에서의 테스트를 통하여 최적의 값을 확인해 봐야 한다는 사실을 알 수 있었다.

<br>

## 2) 클러스터링 속도 개선

**DBSCAN**은 고해상도의 이미지에서 엄청나게 느리게 작동하고 리소스 소모가 크다. K-means 알고리즘이 훨씬 더 빠르고 좋은 결과를 얻었다.

> 이유는 K-means 알고리즘의 **시간 복잡도**는 일반적으로 ***O(n * k * i * d)***이며, 주로 **데이터 포인트와 클러스터 중심 간의 거리 계산**과 클러스터 업데이트에 따라 선형적으로 증가하지만 DBSCAN 알고리즘의 **시간 복잡도**는 ***O(n^2)에서 O(nlogn)*** 사이이며, **데이터의 이웃을 탐색하고 클러스터를 형성하는 과정**에서 시간이 소요되어 데이터가 밀집된 지역과 데이터의 크기와 차원에 따라 추가로 계산량이 증가할 수 있기 때문이다.

DBSCAN에서의 문제점 해결에서 고려한 5번째 방법인 [***자체적으로  DBSCAN 알고리즘을 pytorch로 구현***] 을 클러스터링 속도 개선과 연관 지어서 진행해 보았다.

PyTorch에 대해 간단하게 설명하면 PyTorch는 GPU 가속과 PyTorch의 최적화된 구현을 통해 이루어진다. GPU를 활용하여 병렬 처리를 수행하고, PyTorch의 최적화된 텐서 연산과 Autograd 기능을 활용하여 연산 속도를 향상 시킬 수 있다.

<br>

> *아래는 torch를 이용하여 테스트 해 본 코드이다.*

```python
import torch
import matplotlib.pyplot as plt
import os

# jpg 이미지 불러오기
from google.colab import drive
drive.mount('/content/drive')
file_path = os.path.join("TBD.jpg")

# 3D 유클리드 거리 계산 함수
def euclidean_distance_3d(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))

# 3D DBSCAN 알고리즘 구현
def dbscan_3d(X, eps, min_samples):

    n_samples = X.shape[0]
    labels = torch.zeros(n_samples, dtype=torch.int)

    # 클러스터 레이블 및 방문 여부 초기화
    cluster_label = 0
    visited = torch.zeros(n_samples, dtype=torch.bool)

    # 각 점에 대해 반복
    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        # 이웃 찾기
        neighbors = torch.nonzero(euclidean_distance_3d(X[i], X) < eps).squeeze()

        if neighbors.shape[0] < min_samples:
            # 잡음으로 레이블 지정
            labels[i] = 0
        else:
            # 클러스터 확장
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster_3d(X, labels, visited, neighbors, cluster_label, eps, min_samples)

    return labels

# 클러스터 확장 함수
def expand_cluster_3d(X, labels, visited, neighbors, cluster_label, eps, min_samples):
    i = 0
    while i < neighbors.shape[0]:
        neighbor_index = neighbors[i].item()
        if not visited[neighbor_index]:
            visited[neighbor_index] = True
            neighbor_neighbors = torch.nonzero(euclidean_distance_3d(X[neighbor_index], X) < eps).squeeze()
            if neighbor_neighbors.shape[0] >= min_samples:
                neighbors = torch.cat((neighbors, neighbor_neighbors))
        if labels[neighbor_index] == 0:
            labels[neighbor_index] = cluster_label
        i += 1

# DBSCAN 파라미터 설정
eps = 3
min_samples = 10

# 클러스터링 수행
labels = dbscan_3d(X, eps, min_samples)

# 클러스터 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
ax.set_title('3D DBSCAN Clustering')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)
plt.show()
```

테스트를 통해 **colab CPU런타임 유형 에서는 17초가 소요**되며 **colab GPU런타임 유형 에서는 7초가 소요**되는 점을 확인함을 통해 클러스터링 속도 개선 방향으로 PyTorch도 긍정적으로 고려해 볼 수 있다는 점을 알 수 있었다.

> 추가적으로 CUDA를 사용한 환경에서 다른 이미지에 대해서 실험을 진행한 결과, colab 환경에서는 약 22~25초가 걸리고 팀원의 로컬 컴퓨터에서는 1분 4초가 소요되었다. 팀원의 로컬 컴퓨터에서 CUDA 가속을 사용한 환경이 단순 GPU 런타임 유형보다 더 많은 리소스를 요구하거나, 코드의 최적화 수준이 낮거나, 다른 환경 설정이나 하드웨어의 차이 등의 문제가 발생 시 시간 지연이 있을 수 있다고 생각 해 볼 수 있었다.

<br>

---

# 3. Calibration 팀

## 1) LINC 3.0 신청 및 통과

> LINC 3.0 지원사업을 통해 LCD 디스플레이와 학습을 위한 데이터인  
> 팬톤 TCX를 확보하기 위해 과제 신청서를 작성하고 제출하였다. 

![image-20240421003317013](/images/2024-04-19-Weekly Diary(7주차)/image-20240421003317013.png){: .img-default-setting}

![image-20240421003331941](/images/2024-04-19-Weekly Diary(7주차)/image-20240421003331941.png){: .img-default-setting}

![image-20240421003516637](/images/2024-04-19-Weekly Diary(7주차)/image-20240421003516637.png){: .img-default-setting}

**LINC 3.0 지원 사업**을 통해, User Interface용 LCD 디스플레이와 팬톤 TCX를 확보하기로 하였다. 현재 구매 계획중인 LCD 디스플레이는 60,500원(부가세 포함)이며, 팬톤 TCX 30장은 개당 20,900원(부가세 포함)으로 총 627,000원이다. 팬톤 TCX 30장 선정 기준은 색상환을 기준으로 한 대표 색조 6가지와 해당 색조 계열에서 명도 100%, 75%, 50%, 25%에 가까운 스와치를 선별하여 24개를 선정하였다. 추가적으로 무채색 계열의 스와치에서 명도 100%, 75%, 50%, 25%, 0%에 가까운 스와치 여섯 개를 추가로 선별하여 총 30개의 스와치를 선정하였다. 해당 TCX는 데이터로 활용해 노이즈를 보정하는 데에 활용된다.

<br>

![image-20240421003540069](/images/2024-04-19-Weekly Diary(7주차)/image-20240421003540069.png){: .img-width-large}

⇒ *예산이 70만원 배정되었다.* 

<br>

## 2) 프로젝트 구조 설계 및 주요 프로그램 제작

### 프로젝트 구조 소개

![image-20240420130200324](/images/2024-04-19-Weekly Diary(7주차)/image-20240420130200324.png){: .img-default-setting}

`tree -d` 명령어를 통해 살펴본 전체 프로젝트의 구조이다. 주요 디렉터리는 다음과 같다.

- **config**: 프로젝트에서 필요한 설정 파일들을 보관하는 디렉터리이다.
- **data_class**: 프로젝트 전반에서 필요한 data class를 정의한 디렉터리이다. 다만, data class의 실질적인 의미와는 다르게 경로 등의 하드 코딩을 피하기 위한 PathFinder Class 등을 정의한 것이므로 추후에 이름을 변경할 필요가 있다.
- **function**: 프로젝트의 주요 함수들을 모아놓은 디렉터리이다.
- **image**: 촬영한 이미지를 모아두는 디렉터리이다. 내부에서 다시 세분화 되어 있다.
- **module**: 프로젝트 전반에서 import 해서 사용할 수 있는 모듈, 라이브러리들을 모아놓은 디렉터리이다.
- **utility**: 프로젝트에서 유용하게 사용할 수 있는 모든 유틸리티 함수를 모아놓은 디렉터리이다. 특히 내부에 있는 shortcut 디렉터리는 shell에서 프로그램을 간단히 실행하고, 인자를 넘겨 설정을 변경할 수 있도록 하는 shortcut을 위한 shell script 들을 모아둔 디렉터리이다.

다음부터는 각 디렉터리를 자세하게 소개하고, 주요 코드를 기록한다. 코드에 대한 모든 내용을 기록할 수는 없기 때문에 일부 코드와 프로그램의 주요 역할 및 코딩 과정에서 겪은 어려움, 활용 방안 등을 중점적으로 기록하고자 한다. 또한 project 전체 파일을 입샛노랑 깃허브 organization에 있는 project repository에 push 하였으며, 매 주 업데이트를 하며 코드를 관리할 계획이다.

<br>

### 주요 프로그램(function)

- **function** (dir)
  - capture.py

<br>

- **capture.py**

capture.py는 이미지를 촬영하기 위해 만든 촬영 프로그램이다. 현재 본격적인 main 프로그램을 작성하는 단계가 아니기 때문에 실험에 사용하기 편하게 terminal에서 조작할 수 있도록 설계했다. capture의 주요 기능은 다음과 같다.

1. **swatch image**를 촬영하고 저장한다.
2. **tpg image**를 촬영하고 저장한다.
3. **tcx image**를 촬영하고 저장한다.
4. 촬영된 이미지를 **Google Cloud Storage**에 업로드한다.

촬영은 raspberry pi에서 제공하는 최신 카메라 라이브러리인 rpicam을 사용하며, 그 중에서도 이전 버젼인 raspi-still과 가장 잘 호환되는 **rpicam-still** 명령어를 사용한다.

또한 촬영한 이미지 데이터를 윈도우 환경에서 필요할 때마다 다운받아 사용할 수 있도록 **Google Cloud Storage**에 자동 업로드 되는 코드를 추가하였다. 이와 같이 세 개의 촬영 방식을 구분한 이유는 각각의 이름 저장 규칙이 다르고, 디렉터리도 다르기 때문에 이를 구분하기 위해서이다.

<br>

- **swatch**

swatchg는 tpg, tcx도 아닌, 실험을 위해 확보한 임시 원단을 말한다. Labeling 되지 않은 데이터이기 때문에 보정값을 생성하는데 사용하기 보다는 **대략적인 경향성을 파악하여 보정값 생성에 참고**할 수 있도록 하였다. 이후에 swatch를 촬영한 사진 및 활용을 자세히 설명한다.



- **tpg**

보정값 생성을 위해 사용하는 주요 데이터셋으로, 프린팅 된 인쇄지 형태이다. tpg는 무광이기 때문에 빛반사는 존재하지 않지만, 면 재질인 tcx와는 재질에서 차이가 있고, 같은 색을 염색하더라도 나타나는 색이 다르다. 따라서, **근본적으로 tcx와 같은 면 재질에 대해서 추론하는데 사용할 수 없지 않느냐?** 라는 의문이 제기될 수도 있다. 하지만 입샛노랑은 tpg를 tcx의 색을 추론하는 보정값을 생성하는데 사용할 수 있다고 생각한다. 이유는 다음과 같다.

**첫 번째로**, tpg와 가장 유사한 tcx의 색에 대한 정보가 공개되어 있으므로 적은 비용을 들여 tcx처럼 활용할 수 있는 것이다. 이는 예산 문제로 tcx를 대량으로 구매할 수 없는 현재 상황에서 상대적으로 저렴한 tpg를 사용할 수 있는 방안이 될 것이다. 다시 말하면, tpg를 촬영하고 색을 추론할 때, tpg 자체에서 제공되는 색으로 labeling 하는 것이 아니라 가장 유사한 tcx의 색으로 labeling해서 마치 tcx인 것처럼 학습하는 것라고 할 수 있다.

**두 번째로**, 입샛노랑이 고안한 알고리즘을 고려할 때, 재질 간의 차이는 문제가 되지 않을 것이다. 면 재질과 종이의 가장 큰 차이는 크게 두 가지라고 볼 수 있는데, tcx(면 재질)에서는 원사의 직조에 의해서 음영 부분이 생기지만 tpg에서는 그러한 음영이 존재하지 않는다는 것이 첫 번째이고, tcx는 원사를 직조하여 만들기 때문에 염색이 완벽히 고르게 되지 않는 반면 tpg는 대체적으로 고르게 염색된다는 것이 두 번째이다. 그러나 적절한 알고리즘을 설계하면 이러한 재질에 의한 차이(*음영과 염색 정도의 차이*)를 보완할 수 있을 것이다. 이에 대한 자세한 설명은 **5. 주요 알고리즘에 대한 고찰 및 설계**에서 다루도록 한다.

**마지막으로**, tcx 원단을 통해 검증 과정을 진행한다. 위에서 언급한 대로 LINC 3.0을 통해 추가 예산을 마련하였으며, 이를 사용해 tcx 원단을 구매하여 Calibration의 정확도를 평가하고자 한다. tpg를 기준으로 학습을 하는 것에 오차가 존재한다고 하더라도 tcx를 통해 보정값을 최종 조율할 것이므로 정확도를 더욱 향상시킬 수 있다.



- **tcx** 

위에서 언급한 대로 100% 면 재질로 만들어진 swatch이며, 어떠한 색을 염색한 것인지 labeling이 된 데이터이다. 최종적으로 시스템의 정확도 검사 및 보정값 수정에 활용된다.

<br>

- **capture.py 코드**

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


''' 클래스 인스턴스 생성 및 사용 '''
path_finder = PathFinder() # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
path_finder.ensureDirectoriesExist() # 
cloud_controller = CloudController(path_finder) # 클라우드 관련 작업을 관리하는 클래스

''' 카메라 커맨드 지정 '''
# 카메라 미리보기 커맨드 설정, '-t 0'은 타이머 없음을 의미
preview_command = ['rpicam-hello', '-t', '0'] 
# 이미지 캡처 커맨드, '-o'는 출력 파일 경로, '-t 1000'은 1000ms 동안 실행
capture_command = ['rpicam-still', '-o', '', '-t', '1000'] 

''' 사용자 입력에 대한 처리를 위한 키 설정 '''
preview_stop_key = 'q'  # 미리보기 종료 키
user_input_exit = '0'  # 사용자 입력: 종료
user_input_capture = '2'  # 사용자 입력: 이미지 캡처
user_input_preview = '1'  # 사용자 입력: 미리보기 시작

# 저장될 이미지의 파일 확장자 설정
image_file_extension = '.jpg'  # 이미지 파일 확장자로 '.jpg'를 기본값으로 설정

# ----------------------------------------------------------------------------- #

def readImageNumber():
    """
    이미지 번호 파일에서 마지막 이미지 번호를 읽어오는 함수
    파일이 존재하지 않을 경우 1을 반환하고 파일을 생성
    """
    try:
        # 이미지 번호 파일을 읽기 모드로 열어 처리
        with open(path_finder.image_number_file_path, 'r') as file:
            # 파일에서 읽은 값을 정수로 변환하여 반환
            return int(file.read().strip())
        # 파일이 존재하지 않을 경우 예외 처리
    except FileNotFoundError:
        # 파일이 없으면 새로 생성
        with open(path_finder.image_number_file_path, 'w') as file:
            file.write('1') # 초기 값으로 '1'을 작성
        return 1

def saveImageNumber(number):
    """
    캡처된 이미지의 번호를 파일에 저장하는 함수
    캡처마다 번호를 증가시켜 파일에 저장
    """
    
    # 이미지 번호 파일을 쓰기 모드로 열어 처리
    with open(path_finder.image_number_file_path, 'w') as file:
        file.write(str(number)) # 전달받은 이미지 번호를 문자열로 변환하여 파일에 기록

def validate_and_format_image_name(stdscr):
    """
    사용자로부터 이미지 이름을 입력 받고, 적절한 형식으로 변환하는 함수
    입력 형식은 '00-0000' 이며, curses 라이브러리를 사용하여 터미널에서 입력을 받음
    """
    curses.noecho()  # 사용자 입력을 화면에 바로 표시하지 않도록 설정
    stdscr.clear() # 화면을 초기화
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
    # 처리된 이름 반환
    return formatted_name

def main_loop(stdscr):
    """
        메인 루프를 구성하여 사용자 입력에 따라 다양한 기능을 수행
    """
    global image_number # 이미지 번호를 전역 변수로 사용
    image_number = readImageNumber()  # 이미지 번호 로드
    
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
        
        # 내부 루프 시작
        while True:
            stdscr.clear()
            stdscr.addstr("Enter '1' to preview\\nEnter '2' to capture an image\\nEnter '9' to reselect the image type\\nEnter '0' to quit\\n")
            stdscr.refresh()

            # 사용자 입력 받기
            cmd_input = stdscr.getkey()
            # 입력받은 이미지 유형에 따라 저장 경로 설정
            directory = path_finder.swatch_original_dir_path if image_type == '1' else path_finder.tpg_image_dir_path if image_type == '2' else path_finder.tcx_image_dir_path

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
                if image_type == '1':  # Swatch 유형 선택 시
                    image_name = str(image_number)  # 이미지 번호를 파일 이름으로 사용
                    upload_path = os.path.join("swatch_image", "original", f"{image_name}{image_file_extension}")
                else: # TPG 또는 TCX 유형일 경우
                    # 사용자 입력으로 파일 이름을 받음
                    image_name = validate_and_format_image_name(stdscr) # 이미지 이름 입력 받음
                    type_folder = 'tpg_image' if image_type == '2' else 'tcx_image'
                    upload_path = os.path.join(type_folder, f"{image_name}{image_file_extension}")
              
                # 전체 파일 경로 설정
                file_name = os.path.join(directory, f"{image_name}{image_file_extension}")
                capture_command[2] = file_name  # 캡처 명령에 파일명 설정
                subprocess.run(capture_command)  # capture command 실행(rpicam-still)
                stdscr.addstr(f"Image captured and saved as {file_name}.\\n")
                stdscr.refresh()
               
               # 클라우드에 이미지 업로드
                cloud_controller.upload_file(file_name, os.path.basename(file_name), image_type)

                image_number += 1 # 이미지 번호 증가
                saveImageNumber(image_number) # 새 이미지 번호 저장

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

=> *command line에서 숫자 입력을 통해 촬영 모드를 선택하고 저장할 이미지 파일의 이름을 지정할 수 있다.*

<br>

### 유틸리티 프로그램(utility)

<br>

- **shortcut** (dir)
  - capture.sh
  - clswim.sh
  - im2hmap.sh
  - imcomb.sh
  - imcrop.sh
  - tpg2sw.sh
  - viswab.sh

- imcrop.py
- imcomb.py
- im2hmap.py
- viswab.py
- tpg2sw.py
- imclear.py
- im2cloud.py

<br>

**유틸리티**는 시스템 구현 과정에서 주요한 기능을 하지는 않지만 유용하게, 혹은 이미지 분석 등과 같은 **부가적인 작업에 사용될 수 있는 프로그램**을 모아놓은 디렉터리이다. 현재는 imcrop.py, imcomb.py, im2hmap.py, viswab.py, tpg2sw.py를 사용하고 있으며, imclear.py와 im2cloud.py는 8주차에 구현 후 사용할 예정이다.

**shortcut**은 위에서 언급된 **.py 유틸리티 프로그램을 전역적으로 빠르게 실행할 수 있도록 제작한 shell script**이다. 그 외의 역할은 없으며, shell에서 argument를 전달받고, 이를 지정된 path에 위치한 프로그램에 전달하여 실행하는 역할이다.

모든 프로그램의 코드를 기록하기에는 내용이 너무 길어지므로, 먼저 각 유틸리티 프로그램의 역할에 대해서 기록한 후, 대표적인 유틸리티와 숏컷 코드를 하나씩만 기록하도록 한다. (*그 외의 코드는 github repository에 있음.*)

<br>

#### 각 유틸리티의 역할

- **imcrop.py**: 이미지의 이름을 전달받고, 해당 이미지를 원하는 사이즈로 crop하여 저장한다.
- **imcomb.py**: 합칠 두 개의 이미지의 이름을 전달받고, 해당 이미지를 수직적, 혹은 수평적으로 합쳐 저장한다.
- **im2hmap.py**: 이미지의 이름을 전달받고, 해당 이미지에 대한 heatmap을 생성하여 저장한다.
- **viswab.py**: visualize_swatch_brightness.py의 줄임말이며, 원단 촬영시 발생하는 지역적 조도 차이를 보정하기 위해 만든 프로그램이다. 기본적으로 초기에는 조도 센서를 통해 조도 차를 극복하고자 하였으나, 정밀한 조도 센서를 구매하였음에도 조도값이 다소 튀는 것으로 확인되어 S/W 적으로 시도하고자 한다.
- ***imclear.py***: 생성한 이미지를 모두 제거하는 프로그램이다.
- ***im2cloud.py***: 지정한 이미지 파일을 Google Cloud Storage에 저장하는 프로그램이다.

> imclear.py와 im2cloud.py는 아직 사용되지 않으며, 그 이외 각각의 프로그램의 실질적 활용은 **topic 4)** **지역별 조도 차이 보정을 위한 Swatch 촬영**과 **topic 5) 지역별 조도 차이 보정을 위한 TPG 촬영**에서 다룬다.

<br>

- **imcrop.py**

```python
import argparse

# import custom modules
from image_utility import ImageUtility
from path_finder import PathFinder

# create utility instances
util = ImageUtility()
pf = PathFinder()

# 파일 동작에 필요한 경로 설정 변수들 선언
image_file_name="example"
extension=".jpg"

crop_size = (0, 0, 0, 0)

# create argument parser
# argument로 crop 인자를 전달받음.
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, nargs=4, required=False)
parser.add_argument("--name", type=str, required=False)

# argument parsing
args = parser.parse_args()

if args.size:
    size = args.size
    crop_size = (size[0], size[1], size[2], size[3])

if args.name:
    file_name = args.name.split('.')
    image_file_name = file_name[0]
    extension=f".{file_name[1]}"

# 2024.04.17, jdk
# 이미지의 촬영은 반드시 'capture.py'를 거치므로,
# 촬영된 이미지는 /project/image/swatch_image/original에 저장된다.
# 따라서, 별도의 path는 지정하지 않고 file의 이름만 argument로 전달받는다.
# 그리고 crop된 이미지는 역시 /project/image/swatch_image/cropped에 저장된다.
image_file_path=f"{pf.swatch_original_dir_path}/{image_file_name}{extension}"
cropped_image_path=f"{pf.swatch_cropped_dir_path}/{image_file_name}_cropped{extension}"

util.setImagePathAndOpen(image_file_path)

im_size_before_crop = util.getImageSize()
util.cropImage(crop_size[0], crop_size[1], crop_size[2], crop_size[3])
im_size_after_crop = util.getImageSize()

print(f"Image size: {im_size_before_crop[0]}, {im_size_before_crop[1]} ->  \\
{im_size_after_crop[0]}, {im_size_after_crop[1]}")

try:
    util.saveImage(cropped_image_path)
    print(f"Successfully saved {image_file_path}")
    print(f"Cropped image name: {cropped_image_path}")
except:
    print(f"Failed to save {cropped_image_path}")
```

기본적으로 모든 utility는 **argparse** 모듈을 통해 **인자를 받아 유연하게 사용**할 수 있도록 만들었다. 하지만 reuqired=False로 설정해서 **필요할 경우 인자를 하드코딩해서 사용**할 수 있도록 했다. 

인자로는 이미지를 자를 Left, Upper, Right, Lower pixel을 전달하고, 파일의 이름을 전달한다. 이때 파일의 path를 전달하는게 아니라 name을 전달하는 것인데, 그 이유는 image의 상태를 original, cropped, combined, heatmap과 같이 구분해 두었기 때문에 **imcrop에서 다루는 이미지는 항상 original에 있다는 것을 확신할 수 있기 때문**이다. 위의 코드를 확인하면 original 디렉터리의 path를 가져오기 위해 **PathFinder**라는 클래스 인스턴스를 생성하는 것을 볼 수 있는데, 이는 이후에 **데이터 클래스(data_class)** 파트에서 다루고자 한다.

이미지 편집과 수정은 모두 (*지난 6주차 기록에서 기록한*) **ImageUtility** 클래스를 통해서 이루어지는데, 이는 module 디렉터리에 존재하는 image_utility.py에서 import한 클래스이다. ImageUtility 클래스에 대해서는 아래 **모듈(module)** 파트에서 자세히 설명한다.

<br>

- **imcrop.sh**

```bash
#!/bin/bash

arg_cnt=0
mode=0

# arguments
declare -a size # declare an empty list
file_name="" # file name will be opend

for arg in "$@"
do
	if [ "$arg" = "-s" ]; then
		arg_cnt=0
		mode=1
	elif [ "$arg" = "-n" ]; then
		arg_cnt=0
		mode=2
	else
		if [ $mode = 1 ]; then
			size+=($arg)
			((arg_cnt+=1))

			if [ $arg_cnt = 4 ]; then
				mode=0
			fi

		elif [ $mode = 2 ]; then
			file_name=$arg
			((arg_cnt+=1))

			if [ $arg_cnt = 1 ]; then
				mode=0
			fi
		fi	
	fi
done

python /home/pi/project/utility/imcrop.py --size ${size[0]} ${size[1]} ${size[2]} ${size[3]} --name $file_name
```

imcrop.py를 전역적으로 실행하기 위해서 만든 imcrop.sh이다. 본 프로그램을 구현하는 과정에서 어려웠던 점은, **custom하게 multiple argument를 전달받는 shell script를 작성할 방법이 없었다**는 것이다. 인터넷에서 여러 자료를 찾아봤지만 참고할 코드가 없었고, 이에 따라 간단하게 직접 구현하기로 했다.

$@ 키워드를 사용하여 모든 argument에 대해서 loop를 돌고, 초기에 argument의 값으로 option을 구분한 다음, 순차적으로 option별 argument의 값을 저장하는 방식이다. 이와 같이 모든 argument를 저장하고, 이를 python script에 전달하여 imcrop을 항상 전역적으로 실행할 수 있도록 했다.

이렇듯 모든 기능을 최대한 모듈화 하기 위해서 노력했고, 이후에 **viswab.py**와 **tpg2sw.py**에서 모듈화 된 유틸리티 프로그램을 활용한 방식을 기록하고자 한다.

<br>

#### 유틸리티 프로그램을 만든 이유(viswab.py, tpg2sw.py)

이와 같이 imcrop.py, imcomb.py, im2hmap.py와 같은 **이미지 처리 모듈을 만든 이유**는 **viswab**과 **tpg2sw**를 만들기 위해서이다. 현재 Calibration 팀에서 하고자 하는 것은 결국 '**S/W를 통한 지역적 조도 보정**'이기 때문에, 촬영한 이미지에서 **픽셀 별 밝기를 수치화** 할 필요가 있었다.  

이에 따라 viswab이라는 프로그램을 만들고자 하였는데, 현재 가지고 있는 스와치의 크기가 전체 Target Area를 덮지 못한다는 문제가 발생했다. 현재 현업에서 사용되는 스와치의 크기가 대략 100 x 100 (mm^2) 가량임을 고려하면, Target Area의 크기를 줄일 수는 없었다.   

이때, 원단을 두 번 촬영하여 정확히 절반을 잘라 이어붙여서 하나의 원단이 촬영된 것처럼 만든다면 원단의 크기 문제를 해결할 수 있을 것이라고 생각했고, 결과적으로 crop, combine, heatmap convert를 수행하는 세 개의 모듈을 만들게 된 것이다.

그런데, 누군가는 **'한 원단을 촬영하고 그것을 반전해서 사용하면 되지 않겠느냐?'** 라고 의문을 제기할 수도 있다. 그리고 나는 아주 좋은 질문이라고 말하고 싶다. 하지만 유감스럽게도 그렇게 할 수는 없는데, 그 이유는 우리는 **'조명에 의해서 발생하는 지역적 조도 차이'**를 보정하기 위해서 본 과정을 수행하는 것이기 때문에 반드시 **'실제 해당 위치'**에서 촬영된 원단의 이미지가 필요하다. 따라서 단순히 반전을 시키는 것은 본래의 목적을 잃게 되는 것이므로, 반드시 두 개의 이미지를 각각 촬영한 후에 합쳐야만 한다.
{: .notice--success}

<br>

- **viswab.py**

> viswab은 Visualize Swatch Brightness의 줄임말이며, Crop -> Combine -> Convert(Heatmap)의 역할을 수행한다. 앞서 언급한 대로, 촬영한 스와치 이미지를 바탕으로 지역적 조도 차이 특성을 분석하기 위해 만들었다.

<br>

- **코드**

```python
"""
    촬영한 원단을 잘라 이어붙여 하나의 원단 이미지로 만들고,
    Heatmap Visualization을 진행하는 프로그램
"""

# import python modules
import os
import subprocess
import argparse

# import custom modules
from image_utility import ImageUtility
from path_finder import PathFinder
from cloud_controller import CloudController

# 20240.04.19, jdk
# swatch의 이름을 숫자로 넘버링하므로,
# 변환을 시작할 숫자와 마지막 숫자를 입력한다.
# viswab 프로그램을 bash command로 실행했을 때
# argument를 주지 않는다면 코드에서 hard coding 된
# default value를 바탕으로 변환이 시작된다.

file_num_start = 208
file_num_end = 217
combine_extension = ".jpg"

# create argument parser
# argument로 crop 인자를 전달받음.
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=False)
parser.add_argument("--end", type=int, required=False)

# argument parsing
args = parser.parse_args()

if args.start:
    file_num_start = args.start

if args.end:
    file_num_end = args.end

util = ImageUtility()
path_finder = PathFinder()
cloud_controller = CloudController(path_finder=path_finder)

original_image_dir = path_finder.swatch_original_dir_path
cropped_image_dir = path_finder.swatch_cropped_dir_path
combined_image_dir = path_finder.swatch_combined_dir_path
heatmap_image_dir = path_finder.swatch_heatmap_dir_path

# image crop 영역 설정
# 정사각형을 기준으로 자르고자 하므로,
# 자르는 영역 또한 정사각형이 되도록 설정해야 한다.

# Orig. 4056x3040
# Cropped. 2500x2500

# Left, Upper, Right, Lower
crop_size = (700, 260, 3200, 2760)

# 촬영된 모든 이미지에 대해서 imcrop 실행
for file_num in range(file_num_start, file_num_end+1):
    file_name = f"{file_num}.jpg"
    
    print(file_name)
    result = subprocess.run(['imcrop', '-s', f'{crop_size[0]}', f'{crop_size[1]}', f'{crop_size[2]}', f'{crop_size[3]}', '-n', file_name], capture_output=True, text=True)
    print(file_name + " cropping done")

print("\\n")

# # crop 된 모든 이미지에 대해서 imcomb 실행
# # 이때, 짝수는 upper image이고, 홀수는 lower image이다.
for file_num in range(file_num_start, file_num_end+1, 2):
    upper_file_num = file_num
    lower_file_num = file_num+1

    upper_file_name = f"{upper_file_num}_cropped{combine_extension}"
    lower_file_name = f"{lower_file_num}_cropped{combine_extension}"
 
    combined_image_file_name = f"{upper_file_num}_{lower_file_num}{combine_extension}"

    print(f"{upper_file_num}, {lower_file_num}")
    result = subprocess.run(['imcomb', '-u', upper_file_name, '-l', lower_file_name, '-n', combined_image_file_name], capture_output=True, text=True)
    print(f"{upper_file_num}, {lower_file_num} combining done")

# print("\\n")

# # combine 된 이미지에 대해서 heatmap 변환 실행
for file_num in range(file_num_start, file_num_end+1, 2):
    upper_file_num = file_num
    lower_file_num = file_num+1

    combined_image_file_name = f"{upper_file_num}_{lower_file_num}{combine_extension}"

    print(f"{combined_image_file_name}")
    result = subprocess.run(['im2hmap', '-o', combined_image_file_name])
    print(f"{combined_image_file_name} heatmap converting done")

# upload to google cloud service
# combine 된 이미지에 대해서 heatmap 변환 실행
for file_num in range(file_num_start, file_num_end+1, 2):
    upper_file_num = file_num
    lower_file_num = file_num+1

    heatmap_image_file_name = f"{upper_file_num}_{lower_file_num}-hmap{combine_extension}"

    print(f"{heatmap_image_file_name}")
    
    cloud_controller.upload_file(heatmap_image_dir + f"/{heatmap_image_file_name}", heatmap_image_file_name, 1)

    print(f"{heatmap_image_file_name} upload done")
```

=> *코드의 동작은 다음과 같이 이루어진다.*

1. 인자를 통해 start_num과 end_num을 전달받는다.

   capture.py를 통해 촬영된 스와치 이미지는 숫자로 이름이 지정된다. 따라서 촬영한 이미지에 대해서 몇 번부터 몇 번까지 분석을 진행할지 결정하는 숫자라고 볼 수 있다.

2. 모든 이미지에 대해서 imcrop을 진행한다.

   이때, 이미지의 크기는 2500x2500으로 고정한다.

3. 모든 이미지에 대해서 imcomb를 진행한다.

4. 모든 이미지에 대해서 im2hmap을 진행한다.

> 이를 통해 분석한 결과는 **topic 4) 지역별 조도 차이 보정을 위한 Swatch 촬영**에서 자세히 다룬다.

<br>

- **tpg2sw.py**

> tpg2sw.py는 TPG to Swatch의 줄임말이며, viswab과 유사하게 촬영한 tpg를 하나의 원단처럼 만드는 역할을 하는 프로그램이다. tpg는 크기가 다소 작기 때문에 8장을 촬영하고 합쳐야 한다.

<br>

- **코드**

```python
import subprocess
import argparse

from image_utility import ImageUtility
from path_finder import PathFinder

# create utility instances
util = ImageUtility()
pf = PathFinder()

# 이미지 파일 시작 이름
base_num=-1

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, required=False)

args = parser.parse_args()

extension=".jpg"

comb_type_vetically="v"
comb_type_horizontally="h"

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
if args.base:
    base_num=args.base

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
```

=> *viswab.py와 크게 다른 점은 없으며, 초기에 전달받는 인자와 crop_size만 달라졌다.*

현재는 테스트를 위해 만들어진 코드이므로, 인자로 base_num만 전달받는다. base_num은 tpg2sw 코드를 돌릴 첫 번째 이미지의 번호를 말하는데, base_num ~ base_num + 7 까지의 이미지에 대해서 변환을 수행한다. 아직 본 코드는 촬영한 모든 이미지에 대해서 동작하도록 설계된 것은 아니기 때문에 추후에 수정이 필요하다.

<br>

![image-20240420172337283](/images/2024-04-19-Weekly Diary(7주차)/image-20240420172337283.png){: .img-default-setting}

=> *tpg를 8장으로 구분지어 촬영하기 위해 모니터에 경계선을 그린 모습*

<br>

### 모듈(module)

- cloud_controller.py
- image_utility.py
- network_controller.py

<br>

모듈은 **시스템 구현 과정에서 사용될 수 있는 커스텀 라이브러리**이다. 현재는 cloud_controller.py, image_utility.py, network_controller.py가 있다.

<br>

#### 각 모듈의 역할

- **cloud_controller.py**: Google Cloud Controller와의 연결 및 통신을 담당하는 클래스
- **image_utility.py**: 이미지 분석 및 편집에 사용되는 모듈 클래스
- **network_controller.py**: 웹 서버와의 통신에 사용되는 네트워크 클래스이지만, 아직 정확히 구현되지는 않았다. 웹 서버에 대한 내용은 이후에 나올 **Chatper 4. 공통 작업**의 **topic 1) 딥러닝 활용을 위한 원격 웹 서버 구현**에서 다룬다.

<br>

#### image_utility.py

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
# from vispy import app, scene, color, io

class ImageUtility:
    """
        image display, visualization 및 editing과 관련한
        모든 메서드를 포함하는 이미지 유틸리티 클래스
    """

    def __init__(self):
        self.image_path = None
        self.image = None
        self.image_brightness = []

    def setImagePathAndOpen(self, image_path):
        """
            현재는 한 개의 image에 대해서만 다루는 것으로 설정하여
            instance 생성 시 image path를 받아오도록 한다.

            setImagePathAndOpen 함수를 실행할 시,
            기존에 저장된 다른 이미지에 대한 데이터는 모두 None으로 초기화한다.
        """

        self.image_path = image_path
        self.image = Image.open(self.image_path)

        self.image_brightness = []

    def saveImage(self, path):
        """
            open된 image를 지정된 path로 저장하는 함수이다.
        """

        self.image.save(path)

    def displayImage(self):
        """
            open된 image를 display하는 함수이다.
        """

        plt.imshow(self.image)
        plt.axis('off')
        plt.show()
        plt.close()

    def getImageSize(self):
        """
            지정된 image의 Size(Width, Height)를 저장하고, 반환하는 함수이다.
        """

        image_width, image_height = self.image.size

        return (image_width, image_height)

    def cropImage(self, left, upper, right, lower):
        """
            지정된 image를 crop하는 함수이다.
            전달받은 left, upper, right, lower를 crop area로 설정하며,
            open된 img 객체를 crop하고 변형한다.
        """

        crop_area = (left, upper, right, lower)
        self.image = self.image.crop(crop_area)

    def combineImagesVertically(self, upper_image_path, lower_image_path, combined_image_path):
        """
            가로 사이즈가 동일한 두 개의 이미지를 전달받고,
            절반을 잘라 수직적으로 이어붙이는 함수이다.
        """

        # 이미지 오픈
        upper_image = Image.open(upper_image_path)
        lower_image = Image.open(lower_image_path)

        # 가로 길이가 동일한지 체크
        upper_image_size = upper_image.size
        lower_image_size = lower_image.size

        # 이미지의 크기가 동일한지 체크하고
        # 크기가 다르다면 Error를 일으킨다.
        if (not upper_image_size == lower_image_size):
            raise ValueError("Two images have different widths. The widths of the two images have to be the same.")

        width, height = upper_image_size # 두 이미지의 크기가 동일하므로 크기 변수 통일
        # crop_height = height // 2

        """
            TODO 2024.04.16, jdk
            이전에는 스와치를 사용하였기 때문에 crop하는 자동화 코드가 있었지만,
            이제는 TPG에 대해 동작하도록 해야 하므로 crop을 주석처리 함.
            이후에는 추가적인 모듈화를 통해 두 동작을 분리해 주어야 함.
        """

        # 새 이미지 생성
        combined_image = Image.new('RGB', (width, height*2), (255, 255, 255))
        combined_image.paste(upper_image, (0, 0))
        combined_image.paste(lower_image, (0, height))

        # 이미지 저장.
        # 기존에 저장되어 있던 image 변수를 temp_image에 옮겨놨다가
        # cropped_image의 저장이 완료되면 다시 복사한다.
        temp_image = self.image
        self.image = combined_image

        self.saveImage(combined_image_path)
        self.image = temp_image

    def combineImagesHorizontally(self, left_image_path, right_image_path, combined_image_path):
        """
        가로 사이즈가 동일한 두 개의 이미지를 전달받고,
        두 이미지를 수평적으로 이어붙이는 함수이다.
        """

        # 이미지 열기
        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)

        # 이미지의 높이가 동일한지 확인
        left_image_height = left_image.size[1]
        right_image_height = right_image.size[1]

        if left_image_height != right_image_height:
            raise ValueError("Images have different heights. The heights of both images must be the same.")

        # 이미지의 가로 길이를 계산하여 새 이미지의 크기 결정
        total_width = left_image.size[0] + right_image.size[0]
        height = left_image_height  # 이미지의 높이는 동일

        # 새 이미지 생성
        combined_image = Image.new('RGB', (total_width, height))

        # 이미지 이어붙이기
        combined_image.paste(left_image, (0, 0))  # 왼쪽 이미지 위치
        combined_image.paste(right_image, (left_image.size[0], 0))  # 오른쪽 이미지 위치

        temp_image = self.image
        self.image = combined_image

        self.saveImage(combined_image_path)
        self.image = temp_image

    def getImageMode(self):
        """
            현재 지정된 image의 mode를 확인한다.
        """

        mode = self.image.mode

        return mode

    def setImageMode(self, mode):
        """
            Image의 Pixel 값을 어떻게 표현할 것인지 mode를 설정하는 함수이다.
            이때, mode는 L(Gray Scale)RGB, RGBA, CMYK, HSV로 설정 가능하며,
            Open한 image에 대해서 convert를 적용해 변환한다.
        """

        if (not (mode == "L" or mode == "RGB" or mode == "RGBA" or mode == "CMYK" or mode == "HSV")):
            print(f"Wrong mode. The mode have to be one of (L, RGB, RGBA, CMYK, HSV)")
            print(f"Mode you set: {mode}")
            return

        self.image = self.image.convert(mode)

    def calcBrightnessOfPixel(self, RGB):
        """
            RGB값을 기반으로 특정 픽셀의 밝기를 계산하는 함수이다.
            본 방식은 YIQ 색 공간에서 Y 성분을 계산할 때 사용되는 방식이다.
            최솟값 0, 최댓값 255이다.

            TODO 2024.04.18, jdk
            RGB로 밝기를 수치화하는 것이 아니라, CIELAB으로 판단하는 것이
            더 나을 수도 있겠다는 생각이 들었음. 추가적인 논의가 필요해 보임.
        """

        (R, G, B) = RGB

        return 0.299*R + 0.587*G + 0.114*B

    def calcImageBrightness(self):
        """
            현재 지정된 image의 밝기를 Pixel별로 알아내는 함수이다.
        """

        # image.load()를 통해 pixel 값에 접근할 수 있도록 변경
        # Image Library의 mode에 따라서 L(Gray Scale)RGB, RGBA, CMYK, HSV에 접근할 수 있음.
        # 1100x950
        pixels = self.image.load()

        # 2024.04.13, jdk
        # width, height 기준으로 뽑아오므로, 주의 필요함.
        # print(pixels[1099, 949])

        (width, height) = self.getImageSize()

        for w in range(width):
            image_brightness_row = []

            for h in range(height):
                pixel_value = pixels[w, h]
                brightness = self.calcBrightnessOfPixel(pixel_value)
                image_brightness_row.append(brightness)

            self.image_brightness.append(image_brightness_row)

    def getBrightnessOfImage(self):
        """
            지정된 image의 밝기를 반환한다.
        """

        return self.image_brightness

    def getHeatmapSeaborn(self, heatmap_image_path):
        imsize = self.image.size
        width = imsize[0]
        height = imsize[1]

        print(f"Original image size: width {width}, height {height}")

        brightness_values = np.array(self.image_brightness)
        min = brightness_values.min()
        max = brightness_values.max()

        normalized_values = ((brightness_values - min) / (max - min)).T

        # 히트맵 설정
        plt.figure(figsize=(width/100, height/100))
        ax = sns.heatmap(normalized_values, xticklabels=False, yticklabels=False, cbar=False)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # 축 및 라벨 숨기기
        plt.axis('off')

        # 파일로 저장: 원본 이미지와 동일한 해상도와 크기
        plt.savefig(heatmap_image_path, dpi=100, pad_inches=0)
        plt.close()

        heatmap_image = Image.open(heatmap_image_path)
        heatmap_size = heatmap_image.size
        print(f"Heatmap image size: width {heatmap_size[0]}, height {heatmap_size[1]}")
```

이미지 유틸리티는 이미지 편집에 필요한 모든 기능을 메서드화 한 클래스 모듈이다. 구현하는 과정에서 대부분 원활하게 진행 되었지만, 히트맵을 원본 이미지와 동일한 크기로 생성하는데 어려움을 겪었다. 

6주차에 소개한 ImageUtility는 파이썬의 Vispy 라이브러리를 사용하여, 데스크탑의 그래픽카드를 사용하는 방식을 사용하고자 했다. 이를 통해 3D Histogram을 그려 지역별 조도를 Visualization하고자 했으나, 이 방식은 Overhead가 너무 크기 때문에 Vispy를 이용한 Heatmap 방식으로 전환했다.

그런데 이 방식은 그래픽카드를 사용할 수 없는 라즈베리파이에서는 활용할 수 없었고, 결국 matplotlib과 Seaborn을 사용해 Heatmap을 그리게 되었다. 이 과정에서 해상도가 달라지는 문제가 발생했는데, 다음 Stackoverflow 게시글에서 해답을 얻을 수 있었다.

[Stackoverflow 게시글](https://stackoverflow.com/questions/71623207/how-to-change-the-size-of-image-created-by-savefig-seaborn-heatmap-to-100010)

기본적으로 matplotlib은 dpi 설정을 통해 화면에 그려낼 이미지의 크기를 결정하는데, 이때 이미지의 크기는 figsize * dpi로 정해진다. 따라서 figsize를 figsize(desired_width/dpi, desired_height/dpi)로 설정해 주어야만 원하는 해상도의 이미지를 얻어낼 수 있다.

<br>

- **why Heatmap?** 

시스템 구현을 위해서는 반드시 지역별로 존재하는 조도 차이가 없도록 촬영 환경을 조성하여, 최대한 원단의 모든 부분에서 원본과 동일한 색상이 촬영되도록 만들어야만 한다. 그러나 비용적인 한계로 인해 완전한 조명을 구할 수는 없으므로, 조도 보정 알고리즘을 설계하기 위한 인사이트를 얻기 위해서 지역별 조도를 시각화할 필요성을 느꼈다. 이에 따라 Heatmap을 통해 현재 환경에서는 어떠한 조도 특성이 나타나는지 확인하고자 하는 것이다.

(*픽셀 별 밝기는 0.299R + 0.587G + 0.114*B 공식을 통해 얻어냈다.)

<br>

- **추가) Python Interpreter가 Custom Module을 인식하게 하는 방법**

Python에서 어떠한 Script를 실행할 때, 해당 Script보다 상위에 있는 어떠한 Module을 Import하려면 종속성 오류가 발생하게 된다. 이는 Python Interpreter가 경로를 정상적으로 읽어오지 못하기 때문이다.

이 문제를 해결하기 위해서는 다음과 같이 .bashrc에서 PYTHONPATH에 경로를 추가해 주어야 한다.

```bash
export PYTHONPATH=$PYTHONPATH:~/path/to/python/module
```

그 이후에는 특정 Module이 있는 directory에서 __ init __.py 를 만들어 주어야 한다.

<br>

### 이미지(image)

- **swatch_image** (dir)
  - combined (dir)
  - cropped (dir)
  - heatmap (dir)
  - original (dir)
- **tcx_image** (dir)
- **tpg_image** (dir)

<br>

#### swatch_image 디렉터리

스와치 이미지 디렉터리는 촬영한 스와치 이미지를 저장하는 디렉터리이다.

- **original**: original 디렉터리는 capture.py를 통해 촬영한 원본 스와치 이미지를 저장하는 디렉터리이다.
- **cropped**: cropped 디렉터리는 original image를 crop한 이미지를 저장하는 디렉터리이다.
- **combined**: combined 디렉터리는 cropped image를 combine한 이미지를 저장하는 디렉터리이다.
- **heatmap**: heatmap 디렉터리는 combine된 이미지를 heatmap으로 변환한 이미지를 저장하는 디렉터리이다.

이와 같은 구조로 이미지의 역할과 저장되는 디렉터리 공간을 구분해 두었기 때문에, 각 유틸리티 모듈에서 별도의 path를 지정하지 않고 적절한 이미지를 찾아낼 수 있는 것이다.

> 이 외의 디렉터리는 추가적인 설명이 필요하지 않을 것으로 판단되어 적지 않음.

<br>

### 데이터 클래스(data_class)

#### 구성 모듈

- network_url.py
- path_finder.py

<br>

#### 각 모듈에 대한 설명

- **network_url.py**: 웹 서버와 통신할 때 활용되는 API URL을 기록하기 위한 클래스 모듈
- **path_finder.py**: 프로젝트 전체에서 사용되는 경로를 하드코딩하여 클래스 내부 property로 저장한 클래스 모듈

<br>

## 3) 지역별 조도 차이 보정을 위한 Swatch 촬영

### Swatch 히트맵 분석

![image-20240420180455042](/images/2024-04-19-Weekly Diary(7주차)/image-20240420180455042.png)

![image-20240420180426325](/images/2024-04-19-Weekly Diary(7주차)/image-20240420180426325.png)

위의 두 이미지는 viswab.py를 통해서 combine 된 원본 이미지와, im2hmap을 통해 조도에 대한 히트맵을 나타낸 이미지이다. 확인해 보면, 정확한 수치적 정보는 얻어낼 수는 없으나 전반적으로 중앙부의 조도가 상대적으로 부족함을 알 수 있다. 주목해야 할 점은 이것이 밝기가 낮은 색 원단에서만 나타난다는 것이다. 또한 두 이미지를 합친 것이기 때문에 경계선이 나타나는 것을 알 수 있는데, 이를 방지하기 위해서 추후에 Smoothing을 해야 한다.

(*이와 같이 중앙부에서 밝기가 낮게 관측되는 것은 조명을 측정부의 가장자리에 위치시켰기 때문이다.*)



<br>

![image-20240420180734670](/images/2024-04-19-Weekly Diary(7주차)/image-20240420180734670.png)

![image-20240420180801660](/images/2024-04-19-Weekly Diary(7주차)/image-20240420180801660.png)

위 이미지는 동일한 시스템 환경에서 촬영한 흰색 원단에 대해 viswab.py를 수행한 결과이다. 이전과 마찬가지로 정확한 수치적 차이를 얻어내지는 않았지만, 직관적으로 지역적 조도 차이가 존재하지 않음을 알 수 있다.

<br>

![image-20240420180919456](/images/2024-04-19-Weekly Diary(7주차)/image-20240420180919456.png)

![image-20240420180936611](/images/2024-04-19-Weekly Diary(7주차)/image-20240420180936611.png)

이것은 검은색 원단보다는 밝기가 높지만, 흰색 원단보다는 밝기가 낮은 어두운 파란색 원단을 촬영한 결과이다. 이전에 검은색, 흰색 원단에서 나타난 지역적 조도 차이의 중간 정도로 조도 차이가 나타나는 것을 알 수 있다.

<br>

본 기록에는 모든 색에 대한 변환 결과를 첨부하지는 않았지만, 다른 색의 원단에서도 유사한 결과가 도출되었으며, 결론적으로 지역적 조도의 차이는 '**색조**' 보다는 원단의 '**밝기**'에서 비롯된다고 판단하였다. 즉, **원단이 어두울 수록 지역적 조도 차이가 나타날 가능성이 높은 것**이다. 

물론 '**색조**'에 의한 영향이 없다고 완벽하게 결론 지을 수는 없는데, 현재 밝기를 계산하는데 사용한 공식이 녹색에 대해서 강한 가중치를 주고 파란색에 대해서 가장 약한 가중치를 주고 있기 때문이다. 

그러나 우리는 '**색조**'에 의한 영향은 **매우 미미할 것**이라고 생각한다. 그 이유를 아래 두 가지 실험을 보면 알아보자.

<br>



### Swatch 추가 실험



- **<u>실험 (1)</u>**

>  밝기의 가중치를 R, G, B가 모두 동일하게 0.333씩 부여받도록 밝기를 계산하여 히트맵을 계산해 보았다.

- **녹색 계열 원단**

  - 원본 이미지

  ![image-20240420181740196](/images/2024-04-19-Weekly Diary(7주차)/image-20240420181740196.png)

  - 공식 수정 이전

  ![image-20240420190554907](/images/2024-04-19-Weekly Diary(7주차)/image-20240420190554907.png)

  - 공식 수정 후

  ![image-20240420190554907](/images/2024-04-19-Weekly Diary(7주차)/image-20240420190554907.png)

<br>

- **파란색 계열 원단**

  - 원본 이미지

  ![image-20240420181948889](/images/2024-04-19-Weekly Diary(7주차)/image-20240420181948889.png)

  - 공식 수정 이전

  ![image-20240420182024042](/images/2024-04-19-Weekly Diary(7주차)/image-20240420182024042.png) 

  - 공식 수정 후 (0.333)

  ![image-20240420190650845](/images/2024-04-19-Weekly Diary(7주차)/image-20240420190650845.png)

분석 결과, 공식을 수정하여 동일한 가중치를 가했을 때 **녹색 계열 원단**은 이전보다 **중앙부가 다소 어두워지는 것**을 확인할 수 있고, **파란색 계열 원단**은 이전보다 **미약하게 밝아지는 것**을 확인할 수 있다. 

이렇듯 공식의 변화가 히트맵의 결과를 눈에 띄게 바꿀 만큼 영향을 준다고 생각하기는 힘들다. 따라서 밝기를 계산하는데 가해지는 가중치가 정확한 히트맵을 얻는데 중요한 요소로 활용되지는 않는다고 볼 수 있으며, 기존 방법대로 사람의 눈에 가장 잘 들어맞는다고 평가되는 **NTSC의 밝기 공식**을 사용하는 것이 가장 적합할 것이다.

<br>

- **실험 (2)**

> 밝은 녹색, 어두운 녹색, 그리고 밝은 파란색, 어두운 파란색 원단에 대해서 실험을 진행했다.

- **녹색 계열 원단에 대한 viswab 실험**

  - 어두운 원단 (평균 밝기 값: 84.071)

  ![image-20240420190037069](/images/2024-04-19-Weekly Diary(7주차)/image-20240420190037069.png)

  ![image-20240420190102141](/images/2024-04-19-Weekly Diary(7주차)/image-20240420190102141.png)

  - 밝은 원단 (평균 밝기 값: 112.234)

  ![image-20240420181740196](/images/2024-04-19-Weekly Diary(7주차)/image-20240420181740196.png)

  ![image-20240420181802354](/images/2024-04-19-Weekly Diary(7주차)/image-20240420181802354.png)

<br>

- **파란색 계열 원단에 대한 viswab 실험** 

  - 어두운 원단 (평균 밝기 값: 76.548)

  ![image-20240420181948889](/images/2024-04-19-Weekly Diary(7주차)/image-20240420181948889.png)

  ![image-20240420182024042](/images/2024-04-19-Weekly Diary(7주차)/image-20240420182024042.png)

  - 밝은 원단 (평균 밝기 값: 161.687)

  ![image-20240420185207605](/images/2024-04-19-Weekly Diary(7주차)/image-20240420185207605.png)

  ![image-20240420185454714](/images/2024-04-19-Weekly Diary(7주차)/image-20240420185454714.png)



비교 실험의 결과를 확인해보자. 실험에 사용된 네 원단의 밝기를 오름차순으로 정렬하면 다음과 같다.

1. **어두운 파란색** (76.548)

2. **어두운 녹색** (84.071)

3. **밝은 녹색** (112.234)

4. **밝은 파란색** (161.687)

이 순서대로 Heatmap을 정렬해 보도록 하겠다.



![image-20240420182024042](/images/2024-04-19-Weekly Diary(7주차)/image-20240420182024042.png)

![image-20240420190102141](/images/2024-04-19-Weekly Diary(7주차)/image-20240420190102141.png)

![image-20240420181802354](/images/2024-04-19-Weekly Diary(7주차)/image-20240420181802354.png)

![image-20240420185454714](/images/2024-04-19-Weekly Diary(7주차)/image-20240420185454714.png)

정렬하여 확인해 본 결과, **파란색이라고 무조건 지역적 조도 차이가 존재하는 것은 아니며, 녹색이라고 무조건 지역적 조도 차이가 존재하지 않는 것은 아니다.** 히트맵을 확인해 보면, 밝기가 유사한 **어두운 파란색**(76.548)과 **어두운 녹색**(84.071)은 **매우 유사한 지역적 조도 차이**를 보이는 것을 확인할 수 있으며, 밝기 차이가 꽤 난다고 볼 수 있는 **밝은 녹색**(112.234)과 **밝은 파란색**(161.687)은 **지역적 조도 차이가 눈에 띄게 다른 것**을 볼 수 있다. (*밝은 녹색은 미세한 차이가 존재하지만, 밝은 파란색은 거의 존재하지 않는다고 보임.*)

<br>

본 비교 실험이 모든 색조의 원단에 대해서 진행한 실험이 아닌 간단한 실험이므로 무조건적으로 신뢰하는 데는 무리가 있으나, 본 Weekly Diary에는 담지 못한 다른 이미지에서 나타나는 경향성을 볼 때, **원단의 밝기**가 **지역적 조도 차이에 영향을 주는 가장 큰 요인**인 것으로 생각된다. (*깃허브에서 확인 가능하다.*)

**입샛노랑은 이러한 결과가 나타나는 이유가 다음과 같다고 추정**한다. 특정한 물체가 색을 띄는 이유는 해당 물체가 특정한 파장의 빛을 반사하기 때문이다. 그런데 원단의 평균 밝기가 낮아질수록 원단은 전체적으로 어두워지며, 점차 빛을 흡수하는 양이 늘어나기 시작한다. (*검은색은 모든 파장의 빛을 흡수한다.*) 그러므로 원단에서 발생하는 빛의 난반사가 줄어들게 되고, 이에 따라 조명이 충분히 가해지지 않는 영역이 발생하는 것이다. 반대로 원단의 밝기가 밝을 수록 난반사되는 빛의 양이 충분하기 때문에 조도가 거의 일정하게 나타나는 것으로 보인다.

따라서 조도 보정 알고리즘을 설계할 때, 원단의 밝기를 가장 주요한 요인으로 설정해야 하며, 그 이후에 색조에 따라 달라질 수 있는 영향을 고려하는 것이 필요해 보인다.

> 추가적으로, 히트맵을 생성할 때 양자화를 진행하게 되는데, 원본 이미지에서 밝기 값이 가장 낮은 픽셀과 가장 높은 픽셀을 기준으로 0~255 양자화를 진행한 것이므로, 히트맵 색상이 똑같다고 해서 절대적인 밝기가 같은 것은 아니기 때문에 해석에 주의가 필요하다.

<br>

## 4) 최적의 촬영 환경 설계를 위한 조명 부착 위치 변경 실험

> 이전까지는 지역적 조도 차이를 확인하기 위하여 Swatch를 촬영하고 분석해 보았다. 이번에는 지역적 조도 차이를 직접적으로 보정하는 방법으로 가장 먼저 시도할 수 있는, 부착된 조명의 위치 변경을 시도하고자 한다.

- **실험용 Swatch**

![image-20240420193551422](/images/2024-04-19-Weekly Diary(7주차)/image-20240420193551422.png)

<br>

- **(1) 기존 조명 부착 방식에서의 Heatmap**

![image-20240420203850527](/images/2024-04-19-Weekly Diary(7주차)/image-20240420203850527.png)

=> *기존과 같이 가장자리에 조명을 부착했을 때 실험용 Swatch의 Heatmap이다.*



- **(2) 바람개비 모양으로 조명을 부착한 경우**

![image-20240420203008112](/images/2024-04-19-Weekly Diary(7주차)/image-20240420203008112.png){: .img-default-setting}

원단의 중앙부에 조도가 충분히 가해지지 않고 있으므로, 바람개비 모양으로 조명을 부착해 보았다.

<br>

- **실험용 Swatch를 대상으로 Heatmap을 생성한 결과**

![image-20240420203509089](/images/2024-04-19-Weekly Diary(7주차)/image-20240420203509089.png)

<br>

- **(3) 직사각형 모양으로 조명을 부착한 경우**

![image-20240420202959550](/images/2024-04-19-Weekly Diary(7주차)/image-20240420202959550.png){: .img-default-setting}

<br>

- **실험용 Swatch를 대상으로 Heatmap을 생성한 결과**

![image-20240420203756734](/images/2024-04-19-Weekly Diary(7주차)/image-20240420203756734.png)

**중앙 부분에 가해지는 광량이 부족**하기 때문에 **최대한 조명이 균등하게 가해지도록 두 가지 방법으로 조명의 부착 위치를 변경**해 보았으나, 결과적으로 **기존 방식보다 못한 결과**를 보였다. 바람개비 모양으로 부착한 경우에는 중앙부에 가해지는 광량이 이전보다 부족해져 어두운 영역이 더욱 커졌으며, 직사각형 모양으로 부착했을 때는 가장자리 부분이 전반적으로 더 어두워진 것으로 보아 원단에 가해지는 광량 자체가 감소한 것으로 보인다.

정리하자면 **기존의 방식이 가장 우수한 성능을 보였다**는 것인데, 이는 내부 측정부를 구성한 소재가 불투명 흰색 아크릴이다 보니 가장자리에 부착했을 때 빛의 반사가 적절하게 일어나 중앙으로 모이는 광량이 다른 방법에 비해서 더 많았던 것으로 생각된다. 이에 따라 기존의 부착 방식을 유지하도록 한다.

한 가지 아쉬운 점은, 중앙에서 **일렬로 조명을 부착해 보지 않았다**는 것인데, 이후에 추가로 시도해 보아야 할 것이다.

<br>

## 5) 지역별 조도 차이 보정을 위한 TPG 촬영

> 본 시스템에서 주요한 보정 데이터로 활용하고자 하는 것은 tpg이다. tpg를 촬영하고, 마치 하나의 원단을 촬영한 것처럼 보이도록 하기 위해 tpg2sw.py를 사용한다.

tpg에서 한 이미지의 크기는 4.5cm x 2.5cm이다. 따라서 tpg로 하나의 Target Area를 모두 커버하기 위해서는 한 컬러당 8개의 이미지를 촬영한 후, 하나의 이미지로 합쳐야만 한다. 본 과정은 이전에 제작한 utility인 tpg2sw.py를 사용하도록 한다.

<br>

- **tpg 촬영 실험**

![image-20240420210916414](/images/2024-04-19-Weekly Diary(7주차)/image-20240420210916414.png)

![image-20240420210030190](/images/2024-04-19-Weekly Diary(7주차)/image-20240420210030190.png)

위의 두 이미지는 한 tpg에 대하여 8장의 이미지를 찍고, tpg2sw.py를 사용해 2500x2500 이미지로 만든 것이다. 이전에 swatch를 촬영할 때도 확인할 수 있었듯이 단순히 Crop한 후 Combine 한 이미지이므로, 경계선이 나타나는 것을 확인할 수 있다. 이것에 대해서는 추후에 Smoothing을 적용하여 최대한 자연스러운 이미지로 만들어야 한다.

그런데 첫 번째에서는 가장자리에 파란색이 강조되어 보이는 부분이 일부 존재하고, 두 번째 이미지에서는 그러한 부분이 없는 것을 확인할 수 있다. 첫 번째 이미지는 Target Area에 아무것도 올리지 않고 단순히 tpg만 촬영한 것인데, 촬영해보니 원인을 알 수 없는 파란색 선이 생기는 것을 확인했다. 정확한 이유는 알 수 없었으나, 이는 Target Area에서 반사된 빛의 양에 의해서 이미지의 전반적 색조가 영향을 받은 것으로 보인다.

이러한 영향을 감소시키기 위해서 두 번째 이미지는 Target Area를 검은색 불투명 시트지로 가린 채로 촬영하였다. 이에 따라 매우 깔끔한 이미지를 얻어내기는 했으나, 이미지가 전반적으로 밝아진 것을 확인할 수 있다.

<br>

- **Heatmap 분석 결과**

![image-20240420211517304](/images/2024-04-19-Weekly Diary(7주차)/image-20240420211517304.png)

![image-20240420211457867](/images/2024-04-19-Weekly Diary(7주차)/image-20240420211457867.png)

분석 결과, 두 방법 모두 **상대적 조도 차이에 있어서는 큰 차이가 없는 것**으로 보인다. 다만 8장의 이미지를 따로따로 촬영한 것이다 보니 이미지 자체가 하나의 원단처럼 보이지 않기도 하며, 전반적으로 중앙이 어둡게 보여지긴 하나, 대체적으로 **연속적이지 않은 조도가 확인**된다. 이러한 데이터를 하나의 원단을 촬영한 데이터로서 활용하기에는 다소 무리가 있을 것으로 보인다. 

따라서 추후에는 **밝은 색 tpg에 대한 촬영을 진행**해보고, **밝은 색 tpg에서도 이와 같이 불균형한 조도가 관측되는지 확인**해 보아야 한다. 그리고 **어두운 색 tpg에 대해서는 어떻게 불균형한 조도를 제어할 것인지** 생각해 보아야 하며, 최종적으로 **하나의 이미지 처럼 보이도록 접합면에 대한 Smoothing을 실시**해야 한다. 

<br>

# 4. 공통 작업

## 딥러닝 활용을 위한 원격 웹 서버 구현

Clustering 팀의 기록에서 언급되었듯이, 고해상도 이미지 라즈베리파이에서 온전히 프로세싱하는 것은 현실적으로 불가능하다고 보인다. 따라서 우리는 **1) 저해상도 이미지 사용**과 **2) 프로세싱 서버 사용**, 총 두 가지에 대한 접근을 염두에 두고 있다. 본 챕터에서는 고해상도 이미지에 대해서 딥러닝을 적용할 수 있는 딥러닝 서버를 구현하고 테스트한 과정을 기록한다.

<br>

- **목표**

윈도우 데스크탑의 WSL에서 Flask 웹 서버를 열고, 이를 라즈베리 파이와 통신할 수 있도록 설정한다.

> 이 기능을 구현하기 위해 수행한 작업은 다음과 같다.

<br>

- **SSH 서버 오픈**

1. 윈도우 데스크탑의 라우터에서 포트 포워딩을 통해 22번 포트를 오픈한다.
2. 22번 포트에 대해서 인바운드/아웃바운드 규칙을 추가해 방화벽을 연다.
3. Ubuntu의 net-tools 프로그램을 사용하여 윈도우의 22번 포트와 WSL의 22번 포트를 연결한다.
4. WSL에서 openssh-server를 다운받고, SSH 서버를 동작시킨다.

<br>

- **Web 서버 오픈**

1. 윈도우 데스크탑의 라우터에서 포트 포워딩을 통해 6500번 포트를 오픈한다.
2. 6500번 포트에 대해서 인바운드/아웃바운드 규칙을 추가해 방화벽을 연다.
3. 서버 컴퓨터의 WSL에서 Flask 서버를 8000번 포트에서 연다.
4. net-tools를 통해 윈도우의 6500번 포트와 WSL의 8000번 포트를 연결한다.

<br>

- **PyTorch, Cuda 설치**

1. CUDA 12.1 버젼을 위한 Cuda Toolkit을 설치한다. (Ubuntu 버젼)
2. Cuda Path를 추가한다.
3. PyTorch 웹사이트에 접속하여 CUDA 12.1 버젼에 맞는 PyTorch 버젼을 다운로드한다.
4. GPU 사용을 확인한다.

<br>

이와 같은 과정을거쳐 Flask 웹 서버를 오픈하였으며, 라즈베리파이에서 정상적으로 HTTP 통신이 가능함을 확인했다. 또한 내부에서 GPU를 사용하여 PyTorch가 동작함을 확인할 수 있었다. 따라서 지금부터 서버를 통해 DBSCAN을 수행할 수 있으며, 필요할 경우 GPU를사용해 CNN 학습까지 진행할 수 있는 것이다.

그러나 문제가 발생했는데, **메모리가 64GB인 서버 컴퓨터에서 조차 2500x2500 이미지의 DBSCAN이 불가능**했다. 계속해서 이미지의 픽셀을 줄여나가며 확인해 본 결과, **625x625 이미지에 대해서는 약 1분 ~ 1분 30초**의 결과가 나왔고, **500x500 이미지에 대해서는 약 20초**의 결과가 나왔다.

(*이 결과는 hyperparameter 값에 따라 바뀔 수 있음.*)

이전까지는 정확한 결과를 얻기 위해 최대한 고화질인 이미지에 대해서 DBSCAN을 다뤄보고자 했다. 그러 64GB 메모리를 가진 컴퓨터에서조차 메모리 부족 현상이 발생하는 것을 확인했고, 이에 따라 **알고리즘에 전면적인 수정이 필요함을 깨달았다.**

> 지금까지의 결과를 바탕으로 알고리즘을 전면적으로 개편하고자 하며,  
> 현재까지 설계된 내용에 대해서 챕터 5에서 자세히 다루고자 한다.

---

# 5. 주요 알고리즘에 대한 고찰 및 설계

> 이전까지는 고해상도 이미지에 대해서 DBSCAN을 진행하고 색상 군집을 구분한 다음, 적절한 보정과 연산을 적용해 군집별 대표 색상을 추론하여 유사도를 비교하는 알고리즘을 구현하고자 했다.  
> 하지만 연산 속도 이슈가 발생하면서 이전과 같은 초고해상도 이미지(2500x2500)를 다루는 것은 현실적으로 불가능하다고 판단했고, 이에 따라 저해상도 이미지로 고해상도 이미지를 다루는 것과 유사한 결과를 얻을 수 있는 알고리즘을 다시 설계해보고자 한다.

새롭게 고안한 알고리즘은 다음과 같은 과정을 거친다.

1. **색조 보정**

2. **Adaptive Convolution Filter를 사용한 이미지 축소 및 조도 보정**

   > 여기까지의 과정을 **Calibration**이라고 한다.

3. **DBSCAN을 이용한 색상 클러스터링**

   > 이 과정에서 색상 클러스터를 구분한다.

4. **CNN을 활용한 대표 색상 추론**

   > 이 과정을 **Color Inference**라고 한다.

<br>

## 1) 색조 보정

**색조 보정**은 카메라가 특정한 색조의 원단을 촬영할 때 이미지의 색조 자체가 변화되는 현상을 보정하는 것을 말한다. 예를 들자면, 이 현상은 파란색이 강한 원단을 촬영할 때 이미지가 전반적으로 빨간 색조를 띄게 되는 현상을 말한다. 아직까지는 이 현상이 왜 발생하는지 완벽히 파악하지는 못했으나, 실제 색상을 알아내는데 영향을 줄 것이라고 판단되어 보정 과정으로 추가하게 되었다.

![image-20240420235955761](/images/2024-04-19-Weekly Diary(7주차)/image-20240420235955761.png)

=> *색조 변화가 일어나는 상황의 예시*

<br>

보정 방식은 다음과 같다. 

1. 기존에 아무것도 없는 상태에서 바닥을 촬영한 이미지를 바탕으로 Swatch Area의 바깥 쪽 영역의 RGB 색깔을 저장해 둔다. 이때 Swatch Area는 약 100 x 100 (mm^2)의 크기를 가진 영역이며, Target Area는 그 내부에서 80 x 80 (mm^2)의 크기를 가진 영역이다. 촬영하고자 하는 Swatch는 Swatch Area 보다는 작아야 하며, Target Area 보다는 커야 한다.
2. 특정한 원단을 촬영했을 때, 전체 영역에서 **Swatch Area**를 제외한 **Outer Area**의 색상이 기존과 다르게 변한다면, 각 픽셀마다 편차를 구하고, 편차에 대해서 평균을 구해 원단에서 빼준다.

![outer_area](/images/2024-04-19-Weekly Diary(7주차)/outer_area.jpg)

=> *Outer Area를 녹색 사각형으로 표시한 이미지*

> 이 방식은 아직까지 테스트해 보지는 못했으며, 추후에 방식이 변경될 수도 있다.    
> 즉, 연산량과 연산 속도를 고려하여 현재 단계보다 나중에 진행될 수도 있다.

<br>

## 2) Adaptive Convolution Filter를 사용한 이미지 축소 및 조도 보정

7주차에서 알아낸 사실 중 가장 중요한 것은, 시간 복잡도가 높은 알고리즘을 사용해 고해상도 이미지를 프로세싱 하기에는 요구되는 컴퓨팅 리소스가 굉장히 크다는 것이다. 따라서 Single Board Computer를 사용하는 현재 수준에서는 2500x2500 크기의 이미지를 최악의 경우 O(n^2)의 시간 복잡도를 가지며, 매우 많은 양의 메모리를 요구하는 DBSCAN에 적용하는 것이 **현실적으로 불가능에 가까운 것**이다.

더군다나 64GB의 메모리를 갖는 서버 컴퓨터에서 조차 메모리 부족으로 Kernel이 죽은 것을 생각하면, **현재 사이즈의 이미지를 처리하는 것은 불가능하다**. 따라서 **반드시 이미지의 해상도를 감소시켜야 한다.** 하지만 그렇다고 해서 곧바로 **저해상도 이미지를 촬영하는 것은 좋지 않다.** 그 이유는 **면 원단의 재질적인 특성** 때문이다.

<br>

- **예시 이미지**

![image-20240421001123771](/images/2024-04-19-Weekly Diary(7주차)/image-20240421001123771.png)

![image-20240421001020349](/images/2024-04-19-Weekly Diary(7주차)/image-20240421001020349.png)

위의 두 이미지는 예시 스와치 이미지, 그리고 그 스와치를 국소적으로 확대한 이미지이다. 확대하기 이전의 원단 이미지를 볼 때는 단순히 **분홍색 계열의 단색 원단**이라고 생각되었지만, 확대를 해 보니 육안으로 보기에도 여러 개의 색깔이 존재함을 알 수 있다. 어두운 분홍색, 중간 정도의 분홍색, 밝은 분홍색, 흰색 혹은 회색. 즉, **적어도 네 개 정도의 색상이 보인다고 판단**할 수 있다. (*이러한 판단도 주관적이긴 하지만, 대체적으로 그렇다고 판단할 것이다.*)

이러한 특성을 보이는 이유는 옷의 소재인 면 재질이라는 특성 때문이며, 면 원단은 기본 단위인 '원사'를 직조하여 만드는 것이기 때문에 한 색을 갖는 염색약을 사용하더라도, 지역적으로 염색 정도의 차이가 존재해 실제로는 여러 개의 색상이 보일 수밖에 없다. 사람의 눈도 이를 인지하기는 하나, 여러 색상이 연속적으로 위치하고 있을 때 그 색을 혼합하여 처리하는 눈의 메커니즘에 의해 하나의 색상이라고 판단하게 된다.

정밀한 시스템을 개발하고자 한다면, 이와 같은 **재질의 특성을 고려해야 한다**고 생각한다. 그렇지 않고 저해상도 이미지를 촬영한다면 **재질의 특수한 성질을 담아내지 못하게 되어 정확한 결과를 도출하기 어려울 것**이다.

- **해상도를 저하시켜 촬영한 이미지**

![image-20240421002545881](/images/2024-04-19-Weekly Diary(7주차)/image-20240421002545881.png)

본 이미지는 해상도를 1000x750으로 저하시켜 위 원단을 촬영한 이미지이다. 이 경우, 위에서 촬영한 고해상도 이미지에 비해서 **원사가 자세히 관찰되지 않는다.** 또한 **이전에는 네 개 정도로 보였던 색상이, 이번에는 분홍색, 검은색으로 대략 두 개 정도밖에 파악되지 않는 것을 알 수 있다.** 이러한 관찰을 통해, 우리는 본 프로젝트에서 구현하고자 하는 색상 유사도 분석 시스템의 정확도를 확보하기 위해서는 고해상도 이미지를 활용하여 관찰되는 모든 색상이 대표 색상에 반영될 수 있도록 하는 것이 더 좋다고 판단했다.

'*그렇다면 제한된 컴퓨팅 리소스라는 현실적인 한계가 있을 때, 어떻게 고해상도 이미지를 프로세싱 할 수 있을까?*' 고민해 본 결과, **Adaptive Convolution Filter**를 적용하여 이미지의 사이즈를 축소시켜야겠다고 판단했다.

<br>

- **MCU**(Minimum Correction Unit)

**Adaptive Convolution Filter**에 대한 개념을 제시하기 전에, 먼저 **MCU**(Minimum Correction Unit)에 대해서 제시하고자 한다. **MCU**란 입샛노랑이 고안한 개념으로, 어떠한 것인지 상황적 예시를 들어 설명하겠다. 다음과 같이 n x n인 이미지가 있다고 해보자. (*여기서는 2500x2500 크기라고 가정한다.*)

![image-20240421004942813](/images/2024-04-19-Weekly Diary(7주차)/image-20240421004942813.png){: .img-default-setting}

이 경우, MCU는 다음과 같이 결정할 수 있다.

<br>

![image-20240421005018811](/images/2024-04-19-Weekly Diary(7주차)/image-20240421005018811.png){: .img-default-setting}

**MCU**는 m x m 크기를 갖는 특정한 픽셀의 집합이며, 이미지를 보정하는 과정에서 **Position에 따른 보정의 최소 단위가 되는 픽셀의 집합**이라고 표현할 수 있다. 이번 예시에서는 이해를 쉽게 하기 위해 MCU의 크기를 적당히 큰 수치인 500 x 500이라고 가정한다. 이제 본 이미지에 **Adaptive Convolution**을 적용한다.

<br>

- (Brightness Weighted) **Adaptive Convolution**

이 방식은 Convolution을 할 때마다 Filter의 값을 바꾸는 방식이며, 밝은 픽셀에 더 큰 가중치를 부여하는 방식이다. 이렇게 했을 때 얻어지는 이점은 다음과 같다. 

> 이전에 DBSCAN을 통한 군집의 대표 색상을 추론하는 과정에서, 일반적으로 특정 군집에 속한 픽셀 값들 중 가장 밝은 값이 Labeling 값과 가장 가깝다는 것을 확인하였다. 따라서 **Brightness Weighted Adaptive Convolution**을 적용한다면 군집의 대표 색상에 가까운 색상이 살아남을 가능성이 높아지게 될 것이다. 또한 육안으로 관측 가능한 모든 색이 적절하게 고려되면서 대표 색상에 가까워지게 되므로, 좀 더 정확한 결과를 도출할 가능성이 높을 것이라고 생각된다.

<br>

![image-20240421011412248](/images/2024-04-19-Weekly Diary(7주차)/image-20240421011412248.png){: .img-default-setting}

이제 적절한 횟수의 Convoltuion을 진행하여 (n/m) x (n/m) 크기의 이미지로 축소한다. Convolution의 횟수나 Stride, Padding 등은 더 고민이 필요하다.

<br>

- **MCU의 대표성**

Adaptive Convolution을 진행한 후에는, **500 x 500 크기의 MCU가 한 픽셀로 축소된 것이라고 생각할 수 있다.** 

![image-20240421013448693](/images/2024-04-19-Weekly Diary(7주차)/image-20240421013448693.png){: .img-default-setting}

**MCU의 축소**는 **Adaptive Convolution**을 거친 결과이며, 한 픽셀은 축소되기 이전 MCU에 속해 있던 모든 픽셀의 색상 특성을 적절하게 압축하여 보유한다고 고려한다. 즉, 500 x 500 크기의 MCU 내부에 속해 있던 25,000개의 픽셀에 대한 정보가 1픽셀로 압축된 것이라고 보는 것이다. (*이때 축소되기 이전 500 x 500 크기의 MCU 내부에 속해 있던 25,000개의 픽셀을 **Original Pixel Set**이라고 하자.*)

<br>

- **조도 보정에서 Position이 중요한 이유**

이전에 **MCU**는 **Position에 따른 보정의 최소 단위가 되는 픽셀의 집합**이라고 표현했는데, 이는 조도 보정에 있어서 **Position**이 매우 중요하기 때문이다. 

본 시스템 환경에서 촬영된 원단은 색조 또는 밝기에 따라서 조도가 차이나는 위치가 다르며, 위치 별로 차이나는 수치도 다르다. 따라서 **조도를 올바르게 보정하기 위해서는**, 우선 원단을 촬영했을 때 **그 원단에서 조도가 낮게 관측되는 영역이 어딘지 파악할 수 있어야 한다**.

그러므로 **Position에 따른 보정값을 생성**하고자 한다. 보정값은 **단색 원단**을 기준으로 생성한다.

<br>

- **Position에 따른 보정값 생성 과정**

1. 원단마다 색조와 밝기가 다르므로 **조도가 차이나는 위치와 정도가 다를 것**이다.

2. 우선 **단색 원단**에 **Adaptive Convolution**을 적용하여 **MCU를 1픽셀로 축소**시킨다.

   > 이는 2500 x 2500 크기의 이미지를 5 x 5로 축소시킨 과정을 의미한다.

3. 1픽셀마다 픽셀의 RGB 값인 **h0** (Hue), 밝기를 의미하는 **b0** (Brightness)를 구한다.

4. 1픽셀에 대해서 h0, b0라는 값이 구해졌을 때, Labeling 데이터에서 구한 밝기와의 편차인 **α**를 구한다.

5. 모든 픽셀에 대해서 4번을 반복하여 보정한다.

6. 이 결과를 종합하면 다음과 같은 일반적 정리를 얻을 수 있다.

   Adaptive Convolution을 적용한 후에 특정한 position에 위치한 1픽셀에서 h', b'이 구해졌을 때, 그 값이 만약 이전에 구했던 h0, b0와 동일하다면, 해당 픽셀에 보정값 **α**를 적용할 수 있다. (*이때, α는 R, G, B 값이다.*)

7. h0, b0와 동일한 값이 나오는 경우는 거의 없을 수도 있으므로, 구해진 h0, b0 값에 대해서 양자화를 진행하여 특정 구간에 속할 경우 h0, b0와 동일하다고 판단하여 보정값 **α**를 적용하는 방식으로 확장할 수 있다.

> 이와 같은 과정을 통해 이러한 알고리즘을 적용할 수 있는 이유는 **MCU**라는 전제 개념이 있기 때문인데, MCU가 n x n 크기의 집합에 속하는 모든 픽셀의 색상 특성을 적절하게 압축하고 있다고 가정하기 때문이다. 그렇게 적용하기 위한 최소한의 이론적 근거가 바로 **Adaptive Convolution**이다. 이후에 더욱 고민해보며 추가적인 이론적 근거를 마련하기 위해 노력해야 한다.

<br>

- **MCU가 [Minimum]인 이유**

**MCU**가 **[Minimum]** Correction Unit인 이유는, MCU는 가능한 작아야 하기 때문이다. 그 이유는 다음과 같다.

1. 원단을 촬영했을 때 원단의 조도 차이는 위치에 따라서 연속적으로 달라진다. 따라서 이러한 조도 차이를 정확하게 잡아내기 위해서는 **MCU가 작을수록 유리하다.** 즉, **Original Pixel Set이 작을수록 유리하다.** 만약 Original Pixel Set이 크다면 **조도가 극심하게 달라지는 부분이 포함되어 잘못된 결과를 낼 수도 있다.**

2. 본 시스템의 주요 목적은 다색상 원단에 대한 색상 유사도 분석이다. 즉, Target Swatch는 다색상이라는 것이다. 그러므로 만약 MCU의 크기가 크다면, 실제 Calibration 과정에서 한 MCU에 여러 개의 색상이 많이 겹치게 된다. 이럴 경우 완전히 잘못된 결과를 도출할 수 있으므로, **MCU는 가능한 작아야 한다.**

   > 그러나, 다색상 원단을 다룬다면 반드시 여러 개의 색상이 겹치는 부분은 존재하게 된다. 하지만 이러한 것이 크게 문제가 되지는 않을 것이라고 생각하는데, 그 이유는 **MCU의 크기를 가능한 작게 설정한다면**, 애초에 여러 색상이 겹치는 MCU는 거의 만들어지지 않을 것이며, 만들어진다고 하더라도 Adaptive Convolution을 지속적으로 적용하기 때문에 더 밝은 색상에 가깝게 대표 색상이 만들어지기 때문이다.

<br>

- **MCU의 도입이 가져오는 이점**

1. 다색상 원단에 대해서도 MCU마다 h0, b0, α값을 구하여 적절한 조도 보정이 가능하다.
2. Adaptive Convolution을 통해 이미지를 축소시킨 후,  
    복잡도가 높은 알고리즘을 적용하면 연산량을 크게 줄일 수 있다.

<br>

- **MCU의 크기에 따른 Tradeoff**

1. MCU가 클수록 연산량은 줄어들지만, 정확도는 낮아진다.
2. MCU가 작을수록 연산량은 늘어나지만, 정확도는 높아진다.

<br>

## 3) DBSCAN을 이용한 색상 클러스터링

Adaptive Convolution을 적용하면 Outlier는 거의 존재하지 않게 되며, 이미지가 적절한 크기로 축소 된다. 또한 대표 색상에 가까울 것으로 추정되는 MCU들만 남게 되므로, 해당 MCU들에 대해서 DBSCAN을 적용하여 대표 색상 클러스터를 추출한다.

<br>

## 4) CNN을 활용한 대표 색상 추론

이제 각 색상 클러스터 별로 대표 색상을 추론해야 한다. 이전에 이미 색조, 조도에 대한 보정을 진행하였으므로 지역적인 색차는 거의 존재하지 않는다고 봐도 무방하다. 따라서 각 클러스터 별로 픽셀(MCU)을 떼어낸 다음에 새로운 단색 원단 이미지를 생성한다. 만약 색조와 조도에 대한 보정이 이루어지지 않았다면 이러한 과정을 수행해서는 안되지만, 이미 보정이 되었으므로 이렇게 생성되는 새로운 단색 원단 이미지는 '**조명 환경이 완벽한 환경에서 촬영된 단색 원단 이미지**'로 판단해도 된다.

그런데 이 과정에서 주의해야 할 것은, 이전에 색조 및 조도 보정을 진행하였지만 그것만으로는 카메라 자체에 존재하는 노이즈를 보정할 수 없다는 것이다. 따라서 새롭게 생성된 단색 원단 이미지에 대해서 단순히 색상 평균을 구하는 것만으로 대표 색상을 구해서는 안된다. 

즉, 추가적인 Calibration을 적용하고 Color Inference를 수행해야 한다. 만약 이 과정을 통계적으로 수행한다고 한다면 수많은 데이터에 대해서 색조, 밝기 등을 구분하고 그에 따른 보정값을 생성해야만 한다. 그러기보다는 간단한 CNN 모델을 구현하여 Calibration 및 Color Inference를 동시에 수행하고자 한다.

만약 CNN 모델을 매우 복잡하게 구현한다면, 오히려 통계적 기법보다 Overhead가 클 수 있으므로 주의해야 한다.

<br>

> 이것이 7주차에 설계한 알고리즘이고, 이후에 지속적으로 보완 및 개선하며 구현하고 성능을 검증해야 한다.

<br>

---

# 6. 향후 계획

## 1) Clustering 팀

> Clustering 팀에서는 기존 클러스터링 방식의 문제점을 해결하기 위해 고해상도 이미지를 처리하는 방식인 컨볼루션 방법에 대해 추가적인 학습이 필요하며 추가적으로 PyTorch나 CUDA와 같은 GPU 가속을 통한 클러스터링 알고리즘을 최적화하고 속도를 향상 시키는 방안들도 같이 생각 해봐야 한다.

- [ ]  컨볼루션을 이용한 고해상도 이미지 처리 방법 모색

- [ ]  PyTorch 등의 라이브러리를 활용한 클러스터링 속도 개선에 대한 추가 논의

<br>

## 2) Calibration 팀

- [ ] 색조 보정 알고리즘 구현 및 개선

- [ ] MCU based Brightness Weighted Adaptive Convolution을 활용한 조도 보정 알고리즘 구현 및 개선
- [ ] CNN based Calibration and Color Inference 알고리즘 논의 및 구현
- [ ] tpg 스무딩 코드 작성 

<br>

## 공통 계획

- [ ] LINC 3.0 주문
- [ ] tpg 촬영

<br>
