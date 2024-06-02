---
layout: single
title:  "Weekly Diary 13주차(2024.05.27 ~ 2024.06.02)"
excerpt: "13주차 개발 기록 정리"
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
고명훈: 32시간 30분  
김성웅: 29시간 00분  
김우찬: 37시간 30분  
정동교: 33시간 10분
{: .notice--success .text-center}

<br>

------

# 2. CI 팀

## **1) CustomDataset 클래스 개선**

> 기존 CustomDataset 클래스에서는 이미지의 평균 RGB 값만 레이블로 사용하고 있었다. 이 방식은 이미지의 전체적인 색상 정보 만을 단순화하여 제공하므로, 특정 색상의 정확한 예측이나 분류에는 적합하지 않았고 이미지 변환(transform)을 옵션으로 받지만, 실제 변환 적용 과정에서 ToTensor()만 사용되고 있어 변환의 유연성이 부족한 문제점이 있었다. 따라서 이러한 문제점을 개선한 MakeDataset_train을 만들었다.

**개선 사항**

1. **정확한 레이블 정보 사용**

   MakeDataset_train은 csv_path에서 제공된 CSV 파일을 사용하여 각 이미지에 대한 구체적인 RGB 레이블을 제공한다. 이는 모델이 특정 색상을 더 정확하게 학습하고 예측할 수 있게 해준다.

2. **데이터 무결성 검증**

   레이블 데이터에 결측치가 있는지 확인하고, 결측치가 있을 경우 오류를 발생시켜 데이터의 무결성을 보장할 수 있다.

3. **데이터셋의 유연성**

   이미지 경로를 필터링하여 CSV 파일에 존재하는 이미지만을 데이터셋에 포함시킨다. 이는 데이터의 일관성을 유지하고, 불필요한 데이터 처리를 방지하도록 한다.

<br>

**MakeDataset_train 클래스 실행 과정**

1. 초기화 (**init** 메소드)
   - root_dir: 이미지가 저장된 디렉토리 경로.
   - csv_path: 레이블 정보가 저장된 CSV 파일 경로.
   - labels: CSV 파일을 읽어서 pandas DataFrame으로 저장. 인덱스는 이미지 이름
   - image_paths: root_dir에서 이미지 파일을 읽고, 이 파일들이 labels 인덱스에 존재하는지 확인하여 유효한 이미지 경로만 저장
2. 길이 반환 (**len** 메소드)
   - 데이터 셋의 길이, 즉 유효한 이미지 파일의 수를 반환
3. 아이템 가져오기 (**getitem** 메소드)
   - idx: 가져올 이미지의 인덱스
   - img_name: 해당 인덱스의 이미지 경로를 구성
   - 이미지를 읽고 RGB 색 공간으로 변환
   - 해당 이미지의 파일 이름을 사용하여 labels DataFrame에 해당하는 레이블(RGB 값)을 가져옴
   - 이미지와 레이블을 모두 0에서 1 사이로 정규화
   - 이미지와 레이블을 반환
4. 데이터 로더 사용
   - DataLoader를 사용하여 MakeDataset_train 인스턴스에서 배치 크기, 셔플 여부 등을 설정하여 데이터 로딩을 준비한다. 이를 통해 모델 학습 시 배치 단위로 데이터를 효율적으로 로드할 수 있다.

```python
# 학습 데이터셋 정의
class MakeDataset_train(Dataset):
    def __init__(self, root_dir, csv_path):

        self.root_dir = root_dir

        self.csv_path = csv_path
        self.labels = pd.read_csv(csv_path, index_col=0)  # 첫 번째 컬럼을 인덱스로 사용
        self.image_paths = [img for img in os.listdir(root_dir) if os.path.splitext(img)[0] in self.labels.index]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 파일 이름을 사용하여 CSV에서 해당 라벨을 찾음
        label = self.labels.loc[os.path.splitext(self.image_paths[idx])[0], ['R_tpg', 'G_tpg', 'B_tpg']].values
        
        # 라벨 데이터 타입 확인 및 변환

        if np.any(pd.isnull(label)):
            raise ValueError("Label contains NaN values. Check your CSV file for missing entries.")

        
        # 0~1 사이로 정규화
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label.astype(np.float32), dtype=torch.float32) / 255.0
        
        return image, label

# 데이터셋 로드 및 아이템 확인
train_dataset = MakeDataset_train(root_dir='train_set', csv_path='zzinLast.csv')
# 데이터셋 로드
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

<br>

## 2) CNN 모델 변경

> CNN을 사용하여 색 추론을 수행하는 과정에서 Classifier와 Regression 중 어느 것이 더 적합한지 고려해 보았다. 조사 결과 색 추론 문제는 일반적으로 연속적인 값을 예측하는 문제이므로, RGB 값을 직접 예측할 수 있는 회귀 모델이 적합하다고 판단했다.  또한 회귀 모델은 색상 값의 연속성을 반영할 수 있으며, 출력이 연속적인 실수 값으로 이루어지기 때문에 classifier보다 정확한 색상 값을 예측할 수 있다고 판단 할 수 있었다. 따라서 기존에 classifier에 적합하도록 설계된 MobileNetLIke CNN 모델을 Regression에 적합한 CNN 모델로 변경하여 사용하고자 한다.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()  # Sigmoid 활성화 함수
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
```

개선된 CNN 모델은 출력 레이어에서 **1) Sigmoid 활성화 함수를 사용하여 출력을 0과 1 사이로 제한**한다. 이는 회귀 작업, 특히 정규화된 연속 값을 예측하는 작업에 적합하다. 예를 들어, 색상 값을 [0, 255] 범위에서 [0, 1]로 정규화하여 예측하는 경우, Sigmoid 함수는 이 범위 내에서 값을 제한하여 더 정확한 예측을 가능하게 하는 반면, 기존 MobileNetLike 모델은 특별한 활성화 함수 없이 출력을 내보내어 회귀 작업에서 추가적인 출력 범위 조정이 필요했었다. 또한 **2) Adaptive 평균 풀링을 사용하여 다양한 크기의 입력 이미지에 대해 고정된 크기의 출력을 생성**한다. 이는 모델이 다양한 입력 크기에 유연하게 대응할 수 있게 하며, 회귀 작업에서 입력 데이터의 다양성을 효과적으로 처리할 수 있게 한다. 마지막으로 'same' 패딩을 사용함으로서 **3) 입력 이미지의 공간적 차원을 유지**할 수 있다. 이는 이미지의 모든 부분에서 정보를 손실 없이 활용할 수 있게 하며, 특히 이미지의 경계 부분에서도 중요한 특징을 추출할 수 있게 한다. 회귀 모델에서는 입력 데이터의 모든 정보가 중요하기 때문에 이러한 특성은 기존 모델보다 성능을 향상 시킬 수 있었다.

<br>

## 3) 결과 분석 클래스 개선

> 기존 방식의 결과 분석에서는 추론한 색상과 레이블 색상의 코드 차이만 보여주는 정도의 간단한 피드백만을 제공했다. 한눈에 차이를 확인할 수 있고 추후 TPG샘플의 재촬영과 추가 촬영을 위해 어떤 색조, 밝기에서 낮은 결과를 보였는지와 전체 테스트 셋에서 몇퍼센트의 정확도를 보였는지 확인하기 위해 결과 분석 클래스를 수정했다.

### 결과 이미지

```python
df = pd.read_csv('result_log/analysis.csv')
# 이미지 불러오기
image_name = os.path.basename(image_path)
image_name = os.path.splitext(image_name)[0]
parts = image_name.split('-')
brightness = parts[0]
colorTone = parts[1][:2]
# 주어진 RGB 코드를 [0, 1] 범위로 변환
color = np.array(rgb_code) / 255.0
color2 = np.array(label_rgb) / 255.0

# Cie Lab 색공간으로 변환
cie_color = rgb2cie(color)
label_array=np.array(label_rgb)/255
cie_label = rgb2cie(label_array)
cie_dis = round(cie_distance(cie_color, cie_label),4)

# 플롯 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f'Pantone code: {image_name}')
logger.debug(f'Pantone code: {image_name}')

# 추론 색상 플롯
ax1.imshow([[color]])
ax1.set_title(f'Inference RGB: {rgb_code}\\nLab: {cie_color}')
logger.debug(f'Inference RGB: {rgb_code} , Lab: {cie_color}')
ax1.axis('off')

# 레이블 색상 플롯
ax2.imshow([[color2]])
ax2.set_title(f'Label RGB: {label_rgb}\\nLab: {cie_label}')
logger.debug(f'Label RGB: {label_rgb} , Lab: {cie_label}')
ax2.axis('off')

fig.text(0.07, 0.5, f'Distance\\nRGB:{round(color_distance,4)}\\nLab:{round(cie_dis,4)}', va='center', ha='center')
logger.debug(f'RGB Distance:{round(color_distance,4)} , Lab Distance:{cie_dis}')
logger.debug('=============================================================')
plt.savefig(f'result_image/{image_name}.png')
#
new_row = pd.DataFrame([[image_name, brightness, colorTone, color_distance, cie_dis]], columns=df.columns)
df = pd.concat([df, new_row], ignore_index=True)   
df.to_csv('result_log/analysis.csv', index=False) 
plt.close()
```

- **결과 이미지 예시**

![image-20240602234015122](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234015122.png)

레이블과 추론된 rgb,lab를 직관적으로 제공하며 시각적으로 차이를 확인할 수 있고 두 색상공간에서의 거리 또한 알 수 있다.

<br>

### 결과 분석 방식 추가

```python
df = pd.read_csv('result_log/analysis.csv')
df_sorted = df.sort_values(by='Lab Distance', ascending=True)

n = 20

# 특정 범위의 행 선택
df_subset = df.iloc[-n:]
df_subset2 = df.iloc[0:n]
# 'Brightness'와 'colorTone'의 빈도수 계산
brightness_counts = df_subset['Brightness'].value_counts()
colortone_counts = df_subset['colorTone'].value_counts()
brightness_counts2 = df_subset2['Brightness'].value_counts()
colortone_counts2 = df_subset2['colorTone'].value_counts()

# 결과 출력
print(f"성능 하위{n}개 이미지에서의 빈도수:{brightness_counts}\\n")
print(f"성능 하위{n}개 이미지에서의 빈도수:{colortone_counts}\\n")

print(f"성능 상위{n}개 이미지에서의 빈도수:{brightness_counts2}\\n")
print(f"성능 상위{n}개 이미지에서의 빈도수:{colortone_counts2}")
```

테스트 결과를 pandas사용을 위해 csv로 저장하고 cie lab distance로 오름차순 정렬한뒤 상위 20개 하위 20개에서의 팬톤 코드의 밝기와 색조를 분석해서 어떤 조건에서 결과가 좋은지 또는 나쁜지를 분석한다.

- **결과 분석 결과**

  현재 다양한 경우에서의 데이터 분석을 해본 결과 어떠한 색조나 밝기에서 좋거나 나쁜 성능을 보인다기 보다는 규칙성이 없었기 때문에 데이터셋 촬영 과정중 실수가 있었던 샘플에서 저조한 성능을 보였다고 분석할 수 있었다. 때문에 데이터 샘플의 개선이 우선적으로 진행되어야 할 것으로 보인다.

<br>

## 4) 데이터 augmentation

> 보유하고 있는 이미지 데이터셋이 약 1300장 정도 있지만 충분하지 않은 양이기 때문에 데이터 증강 기법을 이용하여 데이터셋의 규모를 키우고 과적합을 방지하고자 한다.

```python
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 이미지 로드 함수
def load_image(image_path):
    return Image.open(image_path)

# 증강 함수 정의
def augment_image(image):
    transformations = {
        'rotate_90': transforms.Compose([
            transforms.RandomRotation((90, 90))
        ]),
        'rotate_180': transforms.Compose([
            transforms.RandomRotation((180, 180))
        ]),
        'rotate_270': transforms.Compose([
            transforms.RandomRotation((270, 270))
        ]),
        'horizontal_flip': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
        ]),
        'vertical_flip': transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0)
        ])
    }

    augmented_images = {}
    for transform_name, transform in transformations.items():
        augmented_images[transform_name] = transform(image)

    return augmented_images

# 디렉토리 내의 모든 이미지에 대해 증강 수행
def augment_images_in_directory(input_directory_path, output_directory_path):
    for filename in os.listdir(input_directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_directory_path, filename)
            original_image = load_image(image_path)
            augmented_images = augment_image(original_image)

            # 증강된 이미지 저장
            for name, img in augmented_images.items():
                save_path = os.path.join(output_directory_path, f'{os.path.splitext(filename)[0]}_{name}.png')
                save_image(transforms.ToTensor()(img), save_path)

# 디렉토리 경로 지정
input_directory_path = '/home/kmh/project/cnn1/original_set'
output_directory_path = '/home/kmh/project/cnn1/augmented_set'
augment_images_in_directory(input_directory_path, output_directory_path)
```

- **augmentation 결과 예시**

![image-20240602234053904](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234053904.png)

<br>

------

# 3. AC 팀

## **1) 데이터셋 분석**

> TPG와 TCX의 색상 관계를 대략적으로 파악하기 위해 데이터셋을 분석하였다. 약 2600개의 색상을 원단에 염색할 때, 어떠한 경향성을 가지고 적용되는지 분석해보았다. TCX(원단)의 색상은 동일한 코드의 TPG로 염색했을 경우 원단에 적용되는 색이다. 즉, 원단의 색상을 보고 어떠한 색으로 염색한 것인지 파악할 수 있다. 선형적인 관계가 파악된다면, 사용자에게 색상 피드백을 줄 때 도움이 될 수 있다고 판단하여 진행하였다.

**밝기-색조-채도 순 오름차순**

![image-20240602234109559](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234109559.png)

**밝기-채도-색조**

![image-20240602234118141](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234118141.png)

**색조-밝기-채도**

![image-20240602234124831](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234124831.png)

**색조-채도-밝기**

![image-20240602234136650](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234136650.png)

**채도-밝기-색조**

![image-20240602234154702](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234154702.png)

**채도-색조-밝기**

![image-20240602234202654](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234202654.png)

<br>

**경향성 분석**

- **색조** 오름차순 시 R, G, B의 값 차이가 각각 두드러지는 부분이 다르다.
- **밝기** 오름차순 시 R, G, B의 값 차이가 **밝기가 어두워질수록** 더 **커진다.**
- 원단의 색채와 **보색**인 파장의 픽셀 값이 **차이가 많이 난다.**
- 채도가 클수록(색이 쨍할수록) RGB색차가 커지는 경향이 있다, 노이즈도 커진다.

> 어느 정도의 경향성은 파악되나,  비선형적인 특성을 띄므로 적용하기 어렵다고 판단한다. 이에 따라 추가적인 연구를 진행해보되, 우선적으로는 RGB 값 차이에 대한 비율로 피드백을 제공하는 것을 목표로 하자.

<br>

## 2) 조도 보정 알고리즘 수정

> 밝기 보정 알고리즘을 수정한다. 지역별 밝기의 차이가 존재하기 때문에 이를 잡아주는 보정 과정을 통해 지역별 색차를 줄이는 과정을 거쳐야 한다. 기존에는 루미넌스 공식을 사용하여 모든 픽셀에 적용하였는데, 해당 공식이 실물의 색상에 대한 밝기를 구하는 과정에서는 적합하지 않다고 판단하여, 알고리즘을 변경하였다. 변경한 알고리즘은  CIELAB색공간에서 휘도, 즉 명도를 동일하게 만드는 방법이다.

![image-20240602234218309](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234218309.png)

원단에서 가장 밝은 부분이 실제 Label값과 가장 유사한 색상을 띄기 때문에, 가장 밝은 부분을 기준으로 전체 이미지의 명도를 끌어올려준다.

![image-20240602234223742](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234223742.png)

그 결과는 다음과 같다. Difference with Center reigion에 해당하는 원단의 가장 밝은 부분과 어두운 부분이 밝기 보정 후, 얼마나 유사해졌는지를 파악하는 부분이다. 왼쪽은 기존 밝기 보정 알고리즘을 적용한 이미지의 분석 결과이고, 오른쪽은 CIELAB 색공간의 휘도를 이용해서 적용한 이미지의 분석 결과이다. 색차가 많이 나지 않는 이미지에 대해서는 큰 변화가 없지만, 색차가 많이 나던 이미지에서는 좋은 결과를 얻어낼 수 있었다.

![image-20240602234242710](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234242710.png)

<br>

## **3) 코드 통합**

> 이미지를 보정하는 모든 코드를 통합하였다. 원단 촬영 시, 모든 이미지의 보정 과정을 거쳐 색상 추론이 가능한 이미지까지 저장될 수 있도록 통합되었다.

**원단 촬영**

1. 색온도 보정(Hue Correction)
2. 이미지 crop(Image Crop)
3. 이미지 combine(Image Combine)
4. 밝기 가중 이미지 축소(Adaptive Convolution)
5. 조도 보정(Brightness Correction)
6. 최종 이미지 생성

이미지 촬영을 모두 완료하고 나서 잘못 촬영된 원단 이미지를 히트맵으로 분석했고, 이를 재촬영하는 과정에 있다. 랜덤 샘플링 한 데이터 1354개 중 약 100개 가량이 잘못 촬영된 것으로 판단되어 모델 정확도에 크게 영향을 미쳤을 것으로 보인다. 따라서 잘못된 데이터는 폐기하고, 회복 가능한 것은 새롭게 생성하고 있다.

<br>

- **잘못 찰영된 원단 이미지 예시**

![image-20240602234251374](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234251374.png)

대표적인 예시로, 본 이미지(13-0442)는 촬영 과정에서 원단의 상태가 통제되지 않은 것을 미처 확인하지 못하고 촬영했다. 원단 자체의 크기가 매우 작고, 노이즈를 발생시키는 원인이 인간의 육안으로 판단하기 어려운 것이 문제이다. 따라서 이러한 데이터들에 대해서만 선별한 후 재촬영을 진행했다.

<br>

- **개선 결과**

![image-20240602234303881](/images/2024-06-02-Weekly Diary(13주차)/image-20240602234303881.png)

개선 후에는 정상적인 데이터로 활용할 수 있게끔 히트맵이 잡힌 것을 확인할 수 있다. 본 데이터는 아직 **조도 보정(BC)을 진행하지 않은 결과**이다.

CIELAB 색공간을 활용한 현재 조도 보정 방식은 개선을 통해 지역적 RGB 비율은 감소시키는데 어느정도 효과를 보였으나, **다색상 원단에서 적절하게 활용되기 위해서는 반드시 RGB 비율 또한 올바르게 보정할 수 있어야 한다.** 따라서 우선 개선된 데이터셋으로 모델이 어느정도 성능 개선을 보이는지 확인하고, 그와 동시에 조도 보정 알고리즘에 대해서 **2차 수정을 진행**해야 한다. 2차 개선을 진행한 후에는 최종적으로 알고리즘이 완성되는 것이며, 이후에 **DBSCAN을 접목한 다색상 원단에 대한 추론으로 넘어가게 될 것**이다.

<br>

## 4) 데이터 개선 후 데이터 추가 생성

데이터 개선 후 변화된 조도 보정 알고리즘을 적용한 후 학습을 진행할 예정이다.

<br>

------

# 5. 향후 계획

## 1) CI 팀

> 현재 개선된 CNN 모델 이외에 추가적인 CNN 모델을 조사 및 적용해보고 학습 과정에서 발생할 수 있는  과적합 방지를 위한 여러 테스트를 진행하고자 한다.

[ ]  ResNet 모델 연구 및 적용

- [ ] 과적합 방지 테스트

<br>

## 2) AC 팀

> 밝기 보정 단계에서 밝은 부분과 어두운 부분의 색조가 다르다는 문제를 해결하기 위한 방안을 마련한다.

- [ ] 밝기 보정 알고리즘 추가 고안

<br>

## 3) 공통 계획

> 추론의 정확도를 높이기 위해 데이터셋 학습을 추가로 진행한다.

[ ]  TPG 촬영

- [ ] TPG색상 추론 및 학습 튜닝
- [ ] 다색상 학습 알고리즘으로 전환

<br>