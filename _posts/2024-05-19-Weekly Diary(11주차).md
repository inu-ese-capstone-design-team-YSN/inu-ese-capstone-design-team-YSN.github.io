---
layout: single
title:  "Weekly Diary 11주차(2024.05.13 ~ 2024.05.19)"
excerpt: "11주차 개발 기록 정리"
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
고명훈: 36시간  
김성웅: 36시간 30분  
김우찬: 47시간  
정동교: 63시간  
(*모든 내용은 디스코드 스레드 및 출석인증표에 증빙되어 있음.*)
{: .notice--success .text-center}

<br>

# 2. 공통 활동

## **1) TPG 데이터 셋 제작**

- 원활한 촬영과 데이터셋 가장자리에서의 색 영향을 최소화하기 위해서 기존의 팬톤 가이드의 모든 색상을 각 팬톤 색상 크기에 맞춰 잘라내었다.

![image-20240519233714537](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233714537.png){: .img-default-setting}

- 팬톤 가이드의 2626개의 색상을 모두 촬영하기에는 너무 많은 시간이 소요됨으로 예상되어 우선적으로 팬톤 넘버링 시스템을 통해 색조, 밝기 및 채도 순서에 따라 분류한 이후 CNN 구현을 위한 최소 필요 데이터셋을 선정하여 진행하기로 하였다. 아래는 팬톤 넘버링 시스템의 분류 기준이다. 

  (남은 데이터셋은 순차적으로 추가하며 진행하기로 하였다.)

  - 각 색상에 지정된 **첫번째** 두자리 숫자는 **밝기**로 숫자 11에서 19까지 **9**단계로 분류되어 있다.
  - **두번째** 2자리 숫자는 **색조**를 의미하며 노랑, 파랑, 녹색 등을 의미한다. **64**개의 섹션은 모든 순수한 색상을 포함하고 있으며, 서클의 중심인 00은 중립점이다.
  - **세 번째** 두자리 숫자는 색의 **채도** 수준을 나타내며 **65단계**로 나누어지며 00 중립에서 시작하여 최대 64 컬러의 채도로 끝난다.

![image-20240519233725054](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233725054.png){: .img-default-setting}

***⇒ 위는 팬톤 넘버링 시스템을 바탕으로 같은 색조를 가진 팬톤 컬러를 밝기에 맞춰서 순차적으로 나열한 예시 이미지이다.***

![image-20240519233739282](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233739282.png){: .img-default-setting}

- CNN 구현을 위한 최소 데이터셋 개수를 지정하기 위해서 색조-밝기를 기준으로 정렬한 데이터에 대해 다음과 같은 기준을 적용하여 랜덤 샘플링을 진행했다.
  - 40개 이상인 경우 : 1/2개를 랜덤으로 선택 (*최소 20개 확보*)
  - 60개 이상인 경우 : 1/3개를 랜덤으로 선택 (*최소 20개 확보*)
  - 80개 이상인 경우 : 1/4개를 랜덤으로 선택 (최소 20개 확보)

![image-20240519233755453](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233755453.png){: .img-default-setting}

⇒ 이와 같이 총 1,327개의 샘플을 선택하였음.

<br>

## 2) **TPG 데이터 셋 촬영 및 combine**

1,327개의 샘플을 촬영하는 과정에 있으며, 현재까지 약 525개 가량을 촬영하였음. 12주차 수요일까지 촬영을 모두 종료할 계획임.

<br>

------

# 3. CI 팀

## **1) CNN 대표색 추론에 적용하기 위한 픽셀 이미지 처리**

> Adaptive Convolution을 적용하면 Outlier는 거의 존재하지 않는 대략 200*200으로 축소된 이미지가 출력 된다. 이후 DBSCAN을 적용하여 대표 색상 클러스터 군집을 추출한 이후  각 클러스터 별 픽셀들을 **CNN 대표색의 정확한 추론 과정을 위해서 모두 동일한 조건의 픽셀 이미지를 적용하여야 한다고 판단하여 픽셀 이미지 처리 방법에 대해 고안하여 보았다.**

<br>

Adaptive Convolution을 적용하였다고 가정한 이미지 크기를 200*200으로 생성한 이후 기존 DBSCAN 알고리즘을 바탕으로 클러스터링된 색상 군집 각각의 모든 픽셀의 개수를 확인해 보고자 하며 이후에 100*100 크기 이미지 내에서 각 픽셀들을 정렬하는 알고리즘을 구현하고자 한다. 마지막으로는 CNN 대표색 추론 과정까지 연계하여 결과값을 확인 해 보고자 한다.

> ***DBSCAN을 통한 클러스터링 진행 후 각 군집 별 픽셀 수 확인***

```python
# 클러스터 크기 계산
cluster_sizes = Counter(labels)

# 노이즈 제외한 클러스터 픽셀 수 계산
cluster_sizes_without_noise = {label: size for label, size in cluster_sizes.items() if label != -1}

# 대표 색상 추출abs
cluster_centers = np.array([np.mean(color_tbl[labels == label], axis=0) for label in set(labels)])

cluster_image = cluster_centers[labels].reshape(img.shape).astype(int)

# 픽셀 수가 가장 많은 클러스터부터 출력하기 위해 클러스터 번호를 픽셀 수의 내림차순으로 정렬
sorted_cluster_labels = sorted(cluster_sizes_without_noise, key=cluster_sizes_without_noise.get, reverse=True)

# 픽셀 수가 가장 많은 클러스터부터 순회하면서 출력
for label in sorted_cluster_labels:
    color = cluster_centers[label].astype(int)
    cluster_size = cluster_sizes_without_noise[label]
    
    # 클러스터 이미지 생성
    color_mask = np.all(cluster_image == color, axis=-1)
    segmented_image = np.zeros_like(img)
    segmented_image[color_mask] = img[color_mask]
    
    # 클러스터 출력
    plt.figure(figsize=(5, 5))
    plt.imshow(segmented_image)
    plt.title(f"Cluster {label} RGB: {color.astype(int)} - Pixel_Size: {cluster_size}")
    plt.axis('off')
    plt.show()

# 대표 색상 추출abs
cluster_centers = np.array([np.mean(color_tbl[labels == label], axis=0) for label in set(labels)])
# 클러스터된 색상 이미지 출력
plt.figure(figsize=(6, 6))
plt.imshow(cluster_centers[labels].reshape(img.shape).astype(int))
# 대표 클러스터 색의 RGB 값 출력 
for label, color in enumerate(cluster_centers):
    if label == -1:
        continue
    color = np.clip(np.squeeze(color), 0, 255).astype(int)
    print(f"Cluster {label} RGB: {color.astype(int)}")
plt.show()  
from skimage import color
```

***⇒***

![image-20240519233855419](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233855419.png){: .img-default-setting}

![image-20240519233905175](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233905175.png){: .img-default-setting}

클러스터링하여 구분된 군집들을 CNN에 적용하기 위한 방법으로 픽셀 이미지 축소 방법과 확대 방법 중 이미지 축소 방식은 loss가 발생할 수 있기 때문에 **1) 군집 중 최대 픽셀 개수를 가진 이미지 크기에 맞추는 방법** 과  **2) AC과정을 거쳐 축소된 이미지 크기에 따라 정해진 이미지 크기에 맞춰 군집화된 픽셀을 채워 넣는 방법**을 고려하였는데 동일한 조건에서 CNN을 적용하기  위해서는 두 번째 방법이 적합하다고 판단하고 테스트를 진행해 보았다.

- **순차 정렬**

![image-20240519233913873](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233913873.png){: .img-default-setting}

- **랜덤 정렬**

![image-20240519233922024](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233922024.png){: .img-default-setting}

> ***순차 정렬과 랜덤 정렬에 따른 CNN 대표색 추론값 차이 확인***

![image-20240519233935935](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233935935.png){: .img-default-setting}

![image-20240519233943983](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233943983.png){: .img-default-setting}

![image-20240519233949738](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233949738.png){: .img-default-setting}

![image-20240519233955860](/images/2024-05-19-Weekly Diary(11주차)/image-20240519233955860.png){: .img-default-setting}

순차 정렬한 그리드 이미지와 랜덤 정렬한 그리드 이미지를 개선을 거친 MobileNet CNN 모델에 적용하여 테스트한 결과 모든 **RGB값이 동일하거나 최대 1 차이 밖에 발생하지 않는 거의 동일한 대표값을 추론함을 통해 순차 방식과 랜덤 방식 모두 이상 없이 정확한 결과가 출력 되었음을** 알 수 있있고  픽셀 이미지 처리 방법을 CNN에 적용하기에도 문제가 없음을 확인할 수 있었다. 다만 픽셀을 그리드하는 과정에서 10분 가량의 예상보다 많은 시간이 소요됨에 따라 최적화를 통한 알고리즘 개선 필요성을 확인하고 추가적으로 진행해 나가고자 한다.

<br>

## **2) 라즈베리 파이 터치 스크린 GUI 구현**

**1.LCD 디스플레이 환경 설정**

- 드라이버 설치

  5inchDisplay(MPI5008) 모델 검색 후 설치

```bash
sudo rm -rf LCD-show
git clone <https://github.com/goodtft/LCD-show.git>
chmod -R 755 LCD-show
cd LCD-show
# ↓ MPI5008모델에 맞는 드라이버
sudo ./LCD5-show
```

- 디스플레이 연결

![image-20240519234010902](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234010902.png){: .img-default-setting}

- 디스플레이와 라즈베리파이의 핀맵 확인 후 연결
- HDMI로 라즈베리파이 포트에 연결

**2. pyqt 설치 및 pyqt designer설치**

- 가상환경 설정 권장

```bash
pip install pyqt5
pip install pyqt-tools
```

이후 파이썬이 설치된 경로로 가보면 qtDesigner exe가 있다.

하지만 pyqt5 모듈을 다운받는 과정에서 문제가 있었는데 사용중인 모든 모듈이 설치된 가상환경의 파이썬 버전과 pyqt5의 버전이 충돌이 생기는것을 검색을 통해 확인했다 때문에 파이썬 기본으로 내장되어있는 표준 라이브러리인 tkinter를 사용하기로 했다.

**3. Interface 구조 설계**

보다 효율적으로 인터페이스를 설계하고 추후 모듈화한 알고리즘을 연결하기 위해 간단한 flowchart를 만들어 보았다.

![image-20240519234021412](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234021412.png){: .img-default-setting}

**4. 코드**

3번의 flowchart를 기반으로 gui코드를 작성 중에 있음.

**5. 추가로 진행하여야 할 사항**

- rpicam preview화면 출력 테스트
- 라즈베리파이 환경에서 테스트
- 유사도 분석에서 다음 3가지 기능으로 분리
  - 한장 색상 추출 확인
  - 두장 유사도 분석
  - 한장과 기존 데이터 분석
- 과거 데이터에서 두가지로 분리
  - 한 스와치 색상 기록
  - 두 스와치 비교 유사도 비교 기록

<br>

------

# 4. AC 팀

## 1) Outer Area Balancing

### **커스텀 RGB 색온도 보정설계**

- **알고리즘**

1. **Label Outer Area RGB 추출:** 아무 것도 올려놓지 않은 상태에서 촬영하여 Outer Area의 좌 우 특정 픽셀들의 RGB 평균을 구한다.

2. **Test Outer Area RGB 추출:** TPG 하나를 올린 상태로 촬영하여 동일한 Outer Area의 RGB 평균을 구한다.

3. **조정 비율 계산:** 세 가지 색 채널 모두 Label Outer Area / Test Outer Area 를 하여 보정값을 생성한다.

   ex) Label Outer Area RGB : (100, 150, 120) , Test Outer Area RGB : (80, 125, 115)

   ⇒ 보정값: (1.25,  1.2, 1.043)

   세 가지 색 채널에 대해 비율적으로 보정값을 주는 이유는 채널간의 상대적인 밸런스를 유지하기 위함이다. 단순한 덧셈 뺄셈으로는 Balancing이 균일하지 않을 수 있다.

4. **이미지 전체에 적용:** 계산된 비율을 이미지 전체에 적용한다. Outer Area의 색이 아무 것도 올려놓지 않고 촬영한 상태의 색상과 동일하게 색온도를 보정한다. 각 RGB가 비율적으로 TPG에도 적용된다.

5. **값 제한:** 색상 값을 최소값 0과 최대값 255 사이로 제한한다.

위의 알고리즘으로 swatch_image의 original을 보정한다.

- Outer Area, Swatch Area, Target Area를 촬영한 사진(Standard)

이미지 크기: 4056x3040

![image-20240519234042261](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234042261.png){: .img-default-setting}

보정 전

![image-20240519234125272](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234125272.png){: .img-default-setting}

보정 후

![image-20240519234133426](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234133426.png){: .img-default-setting}

**결과**: Label과 동일하게 OuterArea가 잘 보정되었다.

<br>

**결과값 비교 및 CIELAB2000 유사도 출력**

다음 코드의 cielab 비교 공식은 아래 논문에 기초한다.  
(*The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations*)

<br>

**결과 예시:**  

*보정 전*  
Outer Area Value: 240 209 223  
Delta E (CIEDE2000) with Standard LAB: 10.959628485944787  
Diff with Standard 26 -2 6  
Target Label: 59 114 95  
Target Area Value: 78 104 104  
Diff: -19 10 -9

<br>

*보정 후*  
Outer Area Value: 214 210 216  
Delta E (CIEDE2000) with Standard LAB: 0.5049259123609368  
Diff with Standard 0 -1 -1  
Target Label: 59 114 95  
Corrected Target Area Value: 69 105 101  
Diff: -10 9 -6

<br>

결론: RGB로 보았을 때에는 미세하게 개선되었고, delta E의 값이 현저히 줄어들었다. 보정된 이미지의 Target LAB이 standard LAB과의 유사도 비교에서 모두 1미만의 결과를 얻을 수 있었다. (업계 스탠다드를 제공하는 팬톤에 따르면, **면 원단을 사용하는 실제 현업에서는 면 원단의 대표색 간의 delta E가 0.5 이하라면 동일한 색이라고 판단**한다. 현재 개발 중인 시스템에서 촬영한 이미지에 대해 화이트 밸런싱을 적용하여 delta E가 0.5가 나왔다는 것은 화이트 밸런싱이 색 추론에 도움이 될 수 있는 결과라고 해석할 수 있다. 다만 모든 색에 대해서 적용해 본 것이 아니므로 추가적인 확인이 필요하다.)

<br>

- **추가 실험 계획**

TPG의 **6가지 색조**(빨, 주, 노, 초, 파, 보), **3가지 명도**, **3가지 채도**에 대한 WB를 진행해보고, H, S, V중 어떠한 요소가 본래 RGB에서 보정 후에도 큰 차이를 야기하는지 확인해본다. (이 과정은 TCX 선정 과정에서도 필요한 실험이다.) HSV to RGB를 통해 적합한 코드를 찾는다.

이러한 실험을 하는 이유는, 다음과 같다.

1. 색조별로 보정 후 RGB의 비율이 밸런스를 유지하는지 확인하기 위함
2. 색조 이외에도 채도, 명도가 색온도 변화에 큰 영향을 미치는지 확인하고, 있다면 색온도 변화량이 값의 변화량(S, V)과 비례적인지/연속적인지 확인하기 위함

- **2차 보정 알고리즘 (보류)**

**2차 보정:** 위의 실험을 통해 촬영한 이미지를 활용하여 이번에는 TPG의 RGB를 Label로 활용해본다. 촬영한 TPG의 RGB와 실제 RGB를 동일하게 만들고 Outer Area의 RGB 차이를 확인한다.

<br>

------

### **CIELAB2000 색공간을 이용한 보정**

- **알고리즘**

1. **Label Outer Area LAB 추출:** 아무 것도 올려놓지 않은 상태에서 촬영하여 Outer Area의 좌 우 특정 픽셀들의 RGB 평균을 구한다. 이 값을 CIELAB2000 색공간으로 변환한다.

2. **Test Outer Area LAB 추출:** TPG 하나를 올린 상태로 촬영하여 동일한 Outer Area의 RGB 평균을 구한다. 이 값을 CIELAB2000 색공간으로 변환한다.

3. **조정 비율 계산:** 세 가지 LAB 채널 모두 Label Outer Area - Test Outer Area 를 하여 보정값을 생성한다.

4. **이미지 전체에 적용:** 이 보정값을 CIELAB색공간으로 변환한 이미지에 전체 더해준다.

   결과: Outer Area의 LAB은 standard와 동일한 LAB을 갖게되고, Target Area 또한 보정값이 더해진다.

5. **유사도 비교(delta E):** CIELAB2000 색상 유사도 비교를 통해 보정 전후 Target Area의 색 변화를 관찰한다.

   **비교 대상:**

   보정 전 이미지에서 Postcard Label LAB vs Target Area LAB

   보정 후 이미지에서 Postcard Label LAB vs Target Area LAB

- **실험 결과**



보정 전

![image-20240519234406107](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234406107.png){: .img-default-setting}

보정 후

![image-20240519234413687](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234413687.png){: .img-default-setting}

**결과값 비교 출력 및 유사도 비교**



**결과 예시:**  
*보정 전*  
Outer Area Value: 240 209 223  
Delta E (CIEDE2000) with Standard LAB: 10.959628485944787  
Diff with Standard 26 -2 6  
Target Label: 59 114 95  
Target Area Value: 78 104 104  
Diff: -19 10 -9

<br>

*보정 후*  
Outer Area Value: 214 211 216  
Delta E (CIEDE2000) with Standard LAB: 0.4891898585716618  
Diff with Standard 0 0 -1  
Target Label: 59 114 95  
Corrected Target Area Value: 43 104 98  
Diff: 16 10 -3

<br>

**결론: RGB로 보았을 때에는 크게 개선되지 않은듯 하나, delta E의 값이 현저히 줄어들었다. 보정된 이미지의 Target LAB이 standard LAB과의 유사도 비교에서 모두 1미만의 결과를 얻을 수 있었다.**

<br>

**<결론>**

**rgb로 색보정한 것이 cielab2000 색공간 색차 비교에서 더욱 좋은 성과를 얻었다.** 따라서 rgb색공간을 이용한 이미지 색온도 보정을 진행한다.

<br>

**정확한 Area 구간 구하기** (코드)

```python
from path_finder import PathFinder
from PIL import Image
from pathlib import Path
#전체 이미지 크기 4056x3040
# PathFinder 인스턴스 생성
path_finder = PathFinder()
# 경로를 Path 객체로 변환
original_image_dir = Path(path_finder.tcx_image_dir_path)

# 원본 이미지 파일 경로
input_image_path = original_image_dir / 'test.jpg'

# 크롭된 이미지를 저장할 파일 경로
# output_image_path = original_image_dir / 'cropped_target_test.jpg'
# output_image_path = original_image_dir / 'cropped_left_test.jpg'
# output_image_path = original_image_dir / 'cropped_right_test.jpg'

# 크롭할 영역의 좌표 (left, upper, right, lower)
# crop_area = (725, 220, 3325, 2830) #target
# crop_area = (0, 0, 350,3040) #left
# crop_area = (3680, 0, 4056, 3040) #right

# 이미지 열기
with Image.open(input_image_path) as img:
    # 이미지 크롭
    cropped_img = img.crop(crop_area)
    
    # 크롭된 이미지 저장
    cropped_img.save(output_image_path)

print(f"Cropped image saved to {output_image_path}")
```

위 코드를 통해 Outer Area와 Target Area의 정확한 x,y 좌표를 구하였다. 해당 이미지의 픽셀을 기존 balancing 코드에 적용하였다.

```python
areas = [
    (0, 0, 350,3040),      # 첫 번째 영역(Left)
    (3680, 0, 4056, 3040),  # 두 번째 영역(Right)
    (725, 220, 3325, 2830)  # 세 번재 영역(Center, Target Area)
]
```

<br>

### WB를 TPG에 적용하기

> TPG 촬영을 시작하여 TPG이미지에 Outer Area 색온도 보정을 적용한다.

보정된 이미지 예시는 다음과 같다.

<br>

![image-20240519234554960](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234554960.png){: .img-default-setting}

*18-6018 원본*

![image-20240519234603886](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234603886.png){: .img-default-setting}

*18-6018 hue corrected*

<br>

balancing 기준은 Target Area에 아무 것도 없는 상태에서의 Outer Area RGB와 TPG를 올린 후 촬영한 이미지의 Outer Area RGB를 동일하게 만드는 것이다.

<br>

### Combine 된 이미지의 내부 색차 확인

> 하나의 TPG 코드 8장을 합쳐 하나의 combined 이미지를 생성하는 과정을 거친다. 이 과정에서 TPG 이미지 8장을 개별적으로 Outer Area Balancing을 진행한 후 crop 및 combine을 하기 때문에, 각 이미지마다 TPG의 색이 다르게 보이는 결과를 초래할 수 있다고 판단하였다. 따라서, 보정한 이미지들을 combine까지 진행하여 각 Target Area 구역별로 이미지 색차가 있는지 확인한다.

- **구역 나누기**

![image-20240519234621337](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234621337.png){: .img-default-setting}

해당 이미지의 각각 8가지 구역을 평균으로 RGB를 계산하여 보았다.

- **결과**

(파일명:targetColor.py) 코드

```python
from PIL import Image
import numpy as np
from path_finder import PathFinder

path_finder = PathFinder()

tpg_combined_dir = path_finder.tpg_combined_dir_path

file_name = "18-6018"
extension = ".jpg"
# 이미지 불러오기
combined_image = Image.open(f'{tpg_combined_dir}/{file_name}_combined{extension}')
image_array = np.array(combined_image)

# 영역 정의 (height, width)
regions = [
    (0, 610, 0, 1250),        # 0~610, 0~1250
    (610, 1250, 0, 1250),     # 610~1250, 0~1250
    (1250, 1890, 0, 1250),    # 1250~1890, 0~1250
    (1890, 2500, 0, 1250),    # 1890~2500, 0~1250
    (0, 610, 1250, 2500),     # 0~610, 1250~2500
    (640, 1250, 1250, 2500),  # 640~1250, 1250~2500
    (1250, 1890, 1250, 2500), # 1250~1890, 1250~2500
    (1890, 2500, 1250, 2500)  # 1890~2500, 1250~2500
]

def calculate_rgb_mean(image_array, region):
    """
    주어진 영역의 RGB 평균을 계산하는 함수
    """
    region_array = image_array[region[0]:region[1], region[2]:region[3]]
    r_mean = np.mean(region_array[:, :, 0])
    g_mean = np.mean(region_array[:, :, 1])
    b_mean = np.mean(region_array[:, :, 2])
    return r_mean, g_mean, b_mean

# 각 영역의 RGB 평균 계산
rgb_means = []
for region in regions:
    r_mean, g_mean, b_mean = calculate_rgb_mean(image_array, region)
    # 소수점 첫째자리에서 반올림하여 정수로 변환
    r_mean = round(r_mean)
    g_mean = round(g_mean)
    b_mean = round(b_mean)
    rgb_means.append((r_mean, g_mean, b_mean))

# 결과 출력
for i, (r_mean, g_mean, b_mean) in enumerate(rgb_means):
    print(f"Region {i + 1}: R mean = {r_mean}, G mean = {g_mean}, B mean = {b_mean}")
```

![image-20240519234652320](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234652320.png){: .img-default-setting}

Region 1: R mean = 61, G mean = 114, B mean = 96  
Region 2: R mean = 55, G mean = 110, B mean = 92  
Region 3: R mean = 54, G mean = 110, B mean = 92  
Region 4: R mean = 60, G mean = 113, B mean = 96  
Region 5: R mean = 59, G mean = 114, B mean = 96  
Region 6: R mean = 53, G mean = 110, B mean = 91  
Region 7: R mean = 53, G mean = 110, B mean = 91  
Region 8: R mean = 58, G mean = 113, B mean = 95 

<br>

**Outer Area Balancing 이후에도 구역별로 RGB 격차가 크게 변화하지 않은 모습을 확인할 수 있다. (해당 RGB는 구역별 모든 픽셀의 평균값에 기반한다.)**

<br>

## 2) Adaptive Convolution

### **밝기 가중 AC 알고리즘**

**RGB픽셀값 밝기 가중**

- **효과**

> 밝기 가중을 하는 이유는 다음 세 가지 효과를 얻을 수 있기 때문이다.

1. **강조 효과**: 밝기가 높은 부분이 더 강조되어 출력 이미지에서 더 두드러지게 나타난다.
2. **밝기 보존**: 원본 블록의 평균 밝기를 보존하면서, 밝기 변화를 반영하여 새로운 값을 생성한다.
3. **이미지 특징 추출**: 밝기를 기준으로 이미지의 주요 특징을 추출하는 데 도움이 될 수 있다.

- **알고리즘**

combine된 TPG이미지의 크기는 2480 * 2500이다. 200*200의 MCU가 100픽셀씩 이동하며 컨볼루션 한다. 디지털 이미지에서 각 픽셀은 일반적으로 그 위치의 색상 정보를 나타내는 숫자로 표현된다. 그레이스케일 이미지에서는 픽셀 값이 0부터 255 사이의 숫자(8비트 이미지 기준)로, 0은 검은색, 255는 흰색을 나타낸다. 색상이 밝을수록 숫자가 커지고, 어두울수록 숫자가 작아진다.

이러한 원리를 기반으로 그레이스케일 이미지에서 픽셀 값 자체를 밝기로 취급하여 컨볼루션한다.

```python
# 밝기 가중치 적용된 Convolution 함수 정의
def brightness_weighted_convolution(block):
    # 블록의 밝기(픽셀 값) 가중치를 적용한 평균을 계산합니다.
    brightness = np.mean(block)
    weighted_sum = np.sum(block * brightness)
    return weighted_sum / (block.size * brightness)
```

brightness = np.mean(block) → 밝기를 계산한다.

weighted_block = block * brightness → 각 픽셀에 해당하는 값에 평균을 곱하므로써, 밝기가 높은 부분은 더 큰 값으로 변환되고, 밝기가 낮은 부분은 더 작은 값으로 변환된다. 이는 밝기 가중치를 적용한 평균 값을 반환하여 이미지의 밝기 정보를 유지하면서 변환된 이미지를 생성한다.

![image-20240519234721848](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234721848.png){: .img-default-setting}

- **최적화**

> MCU의 적절한 크기를 구해야한다. MCU의 크기는 가능한 작아야 한다. 만약 MCU의 크기가 크다면 **조도가 극심하게 달라지는 부분이 포함되어 잘못된 결과를 낼 수도 있다.** 또한 ****실제 Correction 과정에서 한 MCU에 여러 개의 색상이 많이 겹치게 된다. 이럴 경우 완전히 잘못된 결과를 도출할 수 있으므로, **MCU는 가능한 작아야 한다.** 그러나 지나치게 작을 경우, 원단의 오염된 부분이 큰 비중으로 포함될 수 있다. 따라서, 이러한 요인들을 고려하여 적절한 크기의 MCU를 결정해야 한다.

**<TPG : 색상 코드 18-6018>**

![image-20240519234730861](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234730861.png){: .img-default-setting}

MCU_SIZE(KERNEL_SIZE) = 30

STRIDE = 15

![image-20240519234739067](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234739067.png){: .img-default-setting}

<br>

### **Luma 밝기 가중 AC 알고리즘**

- **효과**

밝기 가중의 방식을 Luma 공식을 이용한다. LUMA를 사용하여 밝기를 가중하는 이유는 사람의 눈이 다양한 색상에 대해 서로 다른 민감도를 가지고 있기 때문이다. 특히, 사람의 눈은 녹색에 가장 민감하고, 그 다음으로 빨간색에 민감하며, 파란색에는 가장 덜 민감하다. 이 점을 고려하여 각 색상의 밝기를 가중 평균하면 인간의 시각에 더 잘 맞는 밝기 값을 얻을 수 있다.

LUMA 공식은 다음과 같다:

**𝑌=0.299×𝑅+0.587×𝐺+0.114×𝐵**

이 공식은 사람의 눈이 녹색에 더 민감하고 파란색에 덜 민감하다는 사실을 반영한 것이다. 따라서, 이미지의 밝기를 계산할 때 LUMA 공식을 사용하면 사람의 눈으로 보았을 때 더 자연스럽고 일관된 밝기 값을 얻을 수 있다. 이를 통해 밝기를 기반으로 한 이미지 처리가 더 정확해진다.

**Sliding 방식은 MCU가 서로 주변 MCU 밝기에 영향을 주기 때문에 밝기 변화가 자연스럽고 밝기 변화량이 일관된다는 특징을 갖는다.**

- **알고리즘**

```python
def brightness_weighted_convolution(block):
    # 블록의 밝기(픽셀 값) 가중치를 적용한 평균을 계산합니다.
    R, G, B = block[:,:,0], block[:,:,1], block[:,:,2]
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    weighted_sum = np.sum(block * brightness[:, :, np.newaxis], axis=(0, 1))
    return weighted_sum / np.sum(brightness)
```

- **Test**

MCU_SIZE(KERNEL_SIZE) = 30

STRIDE = 15

![image-20240519234751645](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234751645.png){: .img-default-setting}

**결론: 인간의 시각에 민감하게 보여지는 Luma 알고리즘 기반 밝기 가중을 선택한다.**

<br>

### **밝기 가중 Nesting AC 알고리즘**

- **알고리즘**

Nesting AC는 MCU가 n이라고 가정하면, Stride는 MCU와 동일하며 이 값으로 이미지를 축소하여 밝기 가중한다. MCU와 Stride를 반으로 나누어 한 번 더 이미지를 축소하며 밝기 가중한다.

**Nesting 방식은 중첩된 MCU가 주변 MCU에 서로 영향을 주지 않기 때문에 밝기 차이를 명확하게 확인할 수 있지만, 다소 불연속적으로 변하는 느낌을 받을 수 있다는 것이 특징이다.**

MCU = 4

stride = 4

![image-20240519234805207](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234805207.png){: .img-default-setting}

이 방식은 MCU가 서로 영향을 주지 않는다.

### **밝기 가중 혼합 AC 알고리즘**

> 이 방식은 Sliding AC와 Nesting AC를 혼합하여 제작한 방식이다. stride 값을 MCU의 절반으로 하여 중첩 가중한다.

```python
# 첫 번째 단계: n을 선택 (예: 64으로 가정)
n = 8

# 각 단계의 MCU_SIZE와 stride 설정
mcu_sizes = [n, n // 2]
strides = [n // 2, n // 4]
```

![image-20240519234814444](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234814444.png){: .img-default-setting}

**최종 선택**

모든 방식에 대해 밝기 보정을 적용해보며 가장 적절한 방법을 조금 더 논의해보기로 한다. 모든 방식은 Luma 밝기 공식에 의거한다.

<br>

## 3) 색수차 보정(Fringing Correction)

![image-20240519234827748](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234827748.png){: .img-default-setting}

촬영한 TPG를 Combine 한 이후에는 위의 이미지와 같이 파란색 선이 나타나곤 하는데, 이는 저가형 렌즈를 사용할 때 나타나는 **색수차 현상**(*Fringing*)이다. **색수차 현상**은 백색광이 매질인 렌즈를 통과할 때 파장마다 전달되는 속도를 일정하게 조절하지 못하여 발생하는 현상이다. 이는 고가의 렌즈를 사용하면 교정이 가능하지만, 캡스턴 디자인의 비용 및 경제적 측면을 고려하여 S/W적으로 보정을 수행하도록 한다.

<br>

### **색수차가 문제가 되는 이유**

![image-20240519234843418](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234843418.png){: .img-default-setting}

색수차가 문제가 되는 이유는 AC 과정에서 잘못된 결과를 도출하기 때문이다. 이미지에서 Adaptive Convolution을 거쳤으나, 파란색으로 나타나는 색수차 Pixel이 대체로 밝기가 더 밝기 때문에 축소된 이미지에서도 두드러지게 나타나는 것이다. 이는 색 추론에 큰 영향을 줄 수 있으므로 반드시 보정을 해야만 한다.

<br>

### **색수차 보정 알고리즘(fringing_correction.py)**

> 현재까지의 보정 알고리즘을 정리하면 다음과 같다.

1. TPG 이미지를 Combine한 Joint Surface를 기준으로 네 개의 영역을 나눈다. (area_1, area_2, area_3, area_4)
2. 각 area에 대해서 row의 Blue Pixel값 평균을 구하고, BoxPlot을 통해 1.5 whishi 이상의 값을 갖는 Anomaly Point Row를 찾아낸다. 즉, 파란색으로 나타나는 Row를 찾아내는 것이다.
3. 2번에서 얻어낸 row index를 기준으로 각 area를 위/아래 영역을 나눈다. 이후, 각 영역에서  column을 기준으로 다시 한번 BoxPlot을 수행한다. 이때는 3.0 whishi 이상의 값을 Anomaly로 판단한다.
4. 3번에서 Anomaly로 판단된 점을 주위 값으로 보정한다.
5. 4번을 진행한 후에는 2번에서 찾아낸 Row를 주위 값으로 보정한다.

- **이전 이미지를 보정한 예시**

![image-20240519234855436](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234855436-1716130369989-1.png){: .img-default-setting}

색수차를 보정할 경우 위와 같은 이미지로 나타나게 된다. 다만 아직까지 세로선 색수차(area_4)는 보정하지 못했으며, area_1~3에 속하지 않는 부분에 대해서는 적용되지 않았으므로 코드 수정이 필요하다. 그런데, 보정이 된 부분에 대해서 값은 대략적으로 비슷해졌지만 아직 선형적인 밝기 변화가 잡히지 않은 것을 확인할 수 있다. 따라서 추가적인 과정이 필요할 것으로 생각된다.

<br>

## 4) 밝기 보정 알고리즘 수정

현재 개발 중인 Fringing Correction/AC에 대한 구현이 끝나고 데이터셋 촬영이 완료되면 밝기 보정을 진행한 후, 기존에 구현해두었던 DBSCAN과 CI(Color Inference)를 진행한다. 그런데, 기존에 구상했던 밝기 보정 알고리즘에 문제가 존재하여 다음과 같이 수정하고자 한다.

![image-20240519234939075](/images/2024-05-19-Weekly Diary(11주차)/image-20240519234939075.png)

> 그런데 이 과정을 수행하기 위해 약 53,000,000개의 값을 저장해야 하므로, 최적화와 성능을 고려하여 AC를 200x200에서 50x50 가량으로 더 축소하는 방향을 고려하도록 한다.

<br>

------

# 5. 향후 계획

## 1) CI 팀

> TPG 데이터셋 제작을 마무리하고 픽셀 이미지 처리 알고리즘 최적화 및 LCD GUI 기능 추가 및 개선과 간단한 테스트를 진행 하고자 한다.

- [ ]  TPG 데이터셋 학습 및 튜닝 마무리
- [ ]  픽셀 이미지 처리 알고리즘 최적화
- [ ] LCD GUI 기능 추가 및 테스트 진행

<br>

## 2) AC 팀

> 밝기 가중 이미지를 활용하여 조도 보정 방안을 마련하고 적용한다. 현재까지 개발한 모든 코드를 하나로 통합하여 보정 절차를 축소한다.

- [ ]  조도 보정
- [ ]  코드 통합
- [ ] 랜덤 샘플링을 통한 TCX 선택 및 TCX 주문

<br>

## 3) 공통 계획

- [ ]  TPG 촬영
- [ ]  동대문 방문하여 패턴 원단 확보

<br>