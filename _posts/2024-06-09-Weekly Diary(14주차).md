---
layout: single
title:  "Weekly Diary 14주차(2024.06.03 ~ 2024.06.09)"
excerpt: "14주차 개발 기록 정리"
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
성웅: 29시간 00분  
김우찬: 36시간  
정동교: 33시간 10분

<br>

# 2. 공통 활동

## **1) 원단 공장 방문**

> 일신방직 공장에 방문하여 현재 개발중인 시스템에 대해 확인하고 전문가와 인터뷰하였다.

![image-20240609234802367](/images/2024-06-09-Weekly Diary(14주차)/image-20240609234802367.png)

공장 견학과 전문가 인터뷰를 통해 얻은 내용은 다음과 같다.

<br>

- **문제 상황 재확인**

입샛노랑팀이 제기한 문제 상황들은 실제로 발생하며, 실제로 원단 색상 차이 검증을 한다. 우리 팀은 사염 다색상 원단과 포염 단색 원단에 집중하기로 하였으며, 실제로 이 두 가지 원단은 색차 검증 과정을 거친다. 제작 중인 원단에 대한 색상 유사도의 판단에서 차이가 생긴다면, 공장 입장에서는 매우 큰 손해를 입게 될 가능성이 있다는 사실 또한 확인하였다.

색차에 대한 의견 차이가 존재할 경우, [**유사도 판단에 대한 관점**] 자체적인 판단 기준이 있으나, 이것은 (*숙련자가 사용할 때 의미가 있고, 유지 비용이 매우 비싸서 업계 탑급이 아니면 유지하기 힘든*) 단점이 있다. 또한 피드백의 관점에서는 [**피드백의 관점**] 공장 자체적으로도 조정하는 과정이 있었으나, 이 역시 숙련된 컬러리스트(20~30년 경력) 조차 수 차례 반복해서 염색해야 맞출 수 있다는 사실을 확인했다.

<br>

- **색상 유사도 기준**

DeltaE는 업계 표준이 0.8이고, 일산방직의 피드백 방식도 현재 입샛노랑팀이 제시하고 있는 CIELAB으로 준다. 레시피 유출에 대한 우려 때문에 실제로 염색 테스트를 진행해 보지는 못하였으나, 피드백의 관점에서 보다도 유사도 분석 관점에서의 가치도 충분하다는 것을 확인 받았으며, (피드백 테스트를 진행해 보지는 못하였지만) CIELAB 수치를 제공하는 것만으로도 피드백이 현장 컬러리스트들에게 재염색 피드백으로 유효한 방식이라는 것을 전문가를 통해 직접 확인하였다. 

<br>

- **입샛노랑 팀의 시스템 방식에 대한 리뷰**

전문가가 판단하기에, 현재 입샛노랑팀이 제시하는 시스템의 방식은 분명 업계에서 도움이 될 수 있고, 특히 규모가 작은 공장에서는 기존 제품을 사용할 수 있는 경력자가 없거나 제품 유지 보수가 어렵기 때문에 큰 도움이 될 수 있다고 설명하였다.

<br>

------

# 3. CI 팀

## **1) CNN 학습 성능 개선을 위한 모델 튜닝**

> 최적의 CNN 모델과 하이퍼 파라미터를 최종적으로 결정하기 위해서 BC과정까지 거친 1600여개의 이미지 데이터셋을 사용하여  CNN 모델 학습에서의 데이터 정확성을 확인하고자 했다. Train Loss와 Validaton Loss 값을 바탕으로 최적의 모델을 찾기 위해 학습률, 배치 사이즈, 에폭 등을 변경해가며 테스트를 진행해 보았다.

**Train Loss & Validation Loss 의 이상적인 형태는 학습 단게 마다 아래와 같은 형태를 유지해야 한다.**

**초기 학습 단계**

- **train loss와 validation loss 모두 높은 값을 가지며, 에포크가 진행됨에 따라 감소한다.**

**중간 학습 단계**

- **train loss와 validation loss 모두 지속적으로 감소한다.**
- **train loss는 validation loss보다 약간 낮은 값을 가진다.**

**후기 학습 단계**

- **train loss는 계속해서 낮아지지만, validation loss는 어느 정도 수렴하여 더 이상 크게 감소하지 않는다.**
- **validation loss가 안정적으로 유지되거나 약간 증가할 수 있다.**

<br>

**Regression 모델**

![image-20240609235034160](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235034160.png)

테스트 데이터 정확도: 38.0% 최종 결과 (50.0, 'model/mobilenet_epoch:100_lr:0.0001_batch:32_best_weights.pth')

![image-20240609235041595](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235041595.png)

테스트 데이터 정확도: 50.0% 최종 결과 (50.0, 'model/mobilenet_epoch:100_lr:0.0001_batch:32_best_weights.pth')

![image-20240609235048287](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235048287.png)

테스트 데이터 정확도: 56.00000000000001% 최종 결과 (56.00000000000001, 'model/mobilenet_epoch:100_lr:1e-05_batch:16_best_weights.pth')

![image-20240609235101104](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235101104.png)

테스트 데이터 정확도: 54.50000000000001% 최종 결과 (54.50000000000001, 'model/mobilenet_epoch:100_lr:1e-06_batch:16_best_weights.pth')

BC 이미지 데이터셋을 사용 시 정확성 50%대에서 더 이상 개선되지 않음에 따라 AC 이미지 데이터 셋으로 교체하여 학습을 진행해 보았다.

![image-20240609235108138](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235108138.png)

테스트 데이터 정확도: 57.49999999999999% **최종 결과 (57.49999999999999, 'model_AC/mobilenet_original_size_epoch:100_lr:0.0001_batch:16_best_weights.pth'**

![image-20240609235116833](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235116833.png)

테스트 데이터 정확도: 55.00000000000001% 최종 결과 (55.00000000000001, 'model_AC/mobilenet_original_size_epoch:100_lr:5e-05_batch:16_best_weights.pth')

![image-20240609235127274](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235127274.png)

테스트 데이터 정확도: 61.5%

Best Training Accuracy: 72.09507042253522% 최종 결과 (61.5,

'model_AC/mobilenet_original_size_epoch:300_lr:0.0001_batch:16_best_weights.pth')

**CIE_DIS ( 0to0.8 ) : 123개  **
**CIE_DIS ( 0.8to1 ) : 22개  **
**CIE_DIS ( 1to2 )  : 44개  **
**CIE_DIS ( 2over )  : 11개**

BC 과정을 거치지 않은 AC이미지 데이터셋에서는 데이터 **정확성이 60%대까지는 개선된 모습을 확인**할 수 있었다. 이후 더 나은 데이터 정확성을 얻기 위하여 AC 이미지 데이터셋의 모든 픽셀을 랜덤 재배치하여 학습하도록 진행하는 방법도 테스트 해 보았지만 뚜렷한 개선 결과를 얻을 수 는 없었다. 계속된 모델 튜닝 결과에서 Train Loss와 validation Loss 그래프에서 이상적인 모델 후기 학습 형태가 반복하여 나옴에 따라 현재로서는 모델 튜닝으로만은 획기적인 정확성 개선이 어려울 것으로 판단하여 우선적으로 나머지 **데이터셋을 추가하고 이후 데이터 증강 및 픽셀 랜덤 재배치 데이터셋을 추가하는 방법 등을 통해 데이터 정확도를 개선**하도록 하고자 한다.

<br>

## 2) 다색상 원단 (Check, Stripe) DBSCAN 분석

> 단색상 원단인 TCX DBSCAN을 진행한 결과에서 정확한 결과를 얻을 수 있었다. 따라서 다색상 원단의 체크 무늬와 스트라이프 무늬를 가진 원단을 마련하여 다색상 원단에서의 DBSCAN 통해 기존에 마련한 하이퍼 파라미터와 클러스터링 알고리즘 성능에 대한 최종 확인 테스트를 진행하였다.

**TCX 단색상 원단**

![image-20240609235203364](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235203364.png)

**Check 다색상 원단**

![image-20240609235226950](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235226950.png)

![image-20240609235300219](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235300219.png)



**Stripe 다색상 원단**

![image-20240609235246024](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235246024.png)





모든 테스트 원단에서 정확한 클러스터 군집과 전처리 이미지 데이터들 얻을 수 있었다. 또한 현재 보유한 테스트 원단보다 정밀한 패턴의 원단을 사용 시 발생할 수 있는 문제에 대해서는 클러스터링된 군집 픽셀 수가 특정 개수 이하인 경우에는 군집에서 제거하는 방법을 사용하여 개선하기로 하였다.

다만  위의 원단은 실제로 사용된 원사의 색이 두 개이지만, 혼합되어 인식되어 눈에 보이는 색이 세 개로 보이는 점에 대해서는 현재 개발 시스템에서는 고해상도 이미지를 분석할 수 없기 때문에 개별의 원사 염색에 대한 정밀한 피드백은 다색상 원단에서는 어려울 것이라고 판단 했다. 하지만 원단에 프린트 과정을 거치는 나염 염색 원단 색에 대해서는 충분히 피드백 제공이 가능하다고 판단한다.

<br>

## **3) 모델 구조 변경 테스트 및 증강**

### 1. 채널 분리방식 학습 설계

차원 이미지에서 [r,g,b] 3채널을 한번에 뽑는것이 어려워 성능의 큰 개선이 없을 수도 있다고 생각했기 때문에 r,g,b 각각을 추론하는 모델로 수정해보았다.

- 입력 이미지를 색상채널별로 분리를 진행해서 할지 고민하다 연관성이 깨지는걸 고려해 입력은 그대로 넣기로 했다.

```python
total_epoch = total_epoch
epochs_arr = []
losses_arr = []

for epoch in range(total_epoch):
    
    running_loss = 0.0
    Lab_distance_array=[]
    aa=0
    for images, L_labels, a_labels, b_labels in data_loader:
        images = images.float().to(device)
        L_labels, a_labels, b_labels = L_labels.to(device).view(-1, 1), a_labels.to(device).view(-1, 1), b_labels.to(device).view(-1, 1)
        images = images.permute(0, 3, 1, 2)
        
        optimizer_L.zero_grad()
        L_outputs = model_L(images)
        L_loss = criterion(L_outputs, L_labels)
        L_loss.backward()
        optimizer_L.step()

        optimizer_a.zero_grad()
        a_outputs = model_a(images)
        a_loss = criterion(a_outputs, a_labels)
        a_loss.backward()
        optimizer_a.step()

        optimizer_b.zero_grad()
        b_outputs = model_b(images)
        b_loss = criterion(b_outputs, b_labels)
        b_loss.backward()
        optimizer_b.step()
        
        # 모델 출력과 레이블을 올바른 형태로 변환
        outputs_array = np.array([L_outputs.detach().cpu().numpy(), 
                                a_outputs.detach().cpu().numpy(), 
                                b_outputs.detach().cpu().numpy()])
        labels_array = np.array([L_labels.detach().cpu().numpy(), 
                                a_labels.detach().cpu().numpy(), 
                                b_labels.detach().cpu().numpy()])

        # 배열을 (batch_size, 3) 형태로 변환
        outputs_array = np.transpose(outputs_array, (1, 2, 0)).squeeze()
        labels_array = np.transpose(labels_array, (1, 2, 0)).squeeze()
        # 이제 outputs_array와 labels_array는 (batch_size, 3) 형태
        Lab_distance_array.append(normalize_cie_distance(outputs_array, labels_array))
        aa+=1

        
    epochs_arr.append(epoch)
    losses_arr.append(np.mean(Lab_distance_array))
    
    epoch_loss=np.mean(Lab_distance_array)
    print(f'Epoch {epoch+1}/{total_epoch}, Loss: {epoch_loss}')
```

각 채널을 나누어 별도의 모델에 넣고 채널별로 loss를 구하고 cieLab distance 사용자 피드백 loss로 알려준다.

### 2. 데이터 증강

> 랜덤 셔플링 방식으로 픽셀을 섞어 증강 데이터를 만들고 학습시켜 저조한 성능을 보인 샘플에 적용해 보았다.

```python
Lab_Distance = float(row['Lab Distance'].values[0])
                if Lab_Distance < 0.8:
                    shuffle_num = 1
                elif 0.8<=Lab_Distance < 1.0:
                    shuffle_num = 2
                elif 1.0<=Lab_Distance < 2.0:
                    shuffle_num = 4
                elif 2.0<=Lab_Distance :
                    shuffle_num = 8
```

csv파일에 기록된 테스트 결과와 팬톤코드 인덱스를 통해 기준에 맞춰 분리하고 각 구간마다 몇배의 셔플링 증강을 시킬것인지 결정한다.  그리고 이전 train_set에 누적시켜 가면서 못맞추는 데이터의 비중을 늘려나간다.

![image-20240609235329103](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235329103.png)

지속적으로 저조한 성능을 보이는 샘플은 다양한 조건에서 개선이 되지 않는 모습을 보였다.

불량 데이터라 판단하고 학습 데이터셋에서 제외하는 방법등을 고려할 필요가 있어보인다.

<br>

------

# 4. IC 팀

## **1) 밝기 보정 강화**

> 밝기에 의한 색조 차이를 극복하기 위한 방안을 마련한다.

- International Commission on Illumination (CIE):
  - CIE는 CIELAB 색공간을 정의한 기관으로, L* 값은 색의 밝기만을 나타내며 a*와 b* 값은 색조와 채도를 나타낸다고 설명한다.
- Wyszecki, G., & Stiles, W. S. (2000). Color Science: Concepts and Methods, Quantitative Data and Formulae:
  - 이 책에서도 CIELAB 색공간의 각 축의 의미와 역할을 설명하며, L* 값이 색의 밝기를 나타내고, a*와 b* 값이 색의 색조와 채도를 나타낸다고 설명한다.

**즉, L을 변경하는 것은 색조와 채도에는 영향을 주지 않는다는 결론을 낼 수 있다.** 내 생각에는 원단을 촬영할 때, 지역적 밝기의 차이가 나타나는 각 영역이 카메라에서 서로 다른 색조로 받아들인다고 생각한다.

그래서 AC 이미지에서 히트맵으로 구분되는 밝기 영역별로 Lab의 a와 b의 값을 확인한다. 이 값이 선형적으로 서서히 변하는지 확인해볼 필요가 있다.

<br>

CIELAB 색 공간

***L** 값은 **밝기**를 나타내며, L = 0*이면 **검은색**, *L = 100이면 **흰색***

***a***는 **빨강과 초록** 중 어느 쪽으로 치우쳤는지를 나타내며, **음수**이면 **초록**에 치우친 색깔이고, **양수**이면 **빨강/보라** 쪽으로 치우친 색깔입니다

**b**는 **노랑과 파랑**을 나타내며, **음수**이면 **파랑**, **양수**이면 **노랑**입니다

- **18-3418_AC_adjusted_combined.png**

![image-20240609235348355](/images/2024-06-09-Weekly Diary(14주차)/image-20240609235348355.png)

Pixel (10, 10) RGB: (128, 78, 140) Pixel (20, 20) RGB: (128, 78, 141) Pixel (30, 30) RGB: (128, 78, 140) Pixel (40, 40) RGB: (128, 78, 141) Pixel (50, 50) RGB: (128, 78, 141) Pixel (60, 60) RGB: (128, 78, 141) Pixel (70, 70) RGB: (129, 79, 141) Pixel (80, 80) RGB: (129, 79, 141) Pixel (90, 90) RGB: (129, 79, 141) Pixel (100, 100) RGB: (129, 78, 141)

Pixel (10, 10) LAB: (105, 160, 102) Pixel (20, 20) LAB: (105, 160, 101) Pixel (30, 30) LAB: (105, 160, 102) Pixel (40, 40) LAB: (105, 160, 101) Pixel (50, 50) LAB: (105, 160, 101) Pixel (60, 60) LAB: (105, 160, 101) Pixel (70, 70) LAB: (106, 160, 102) Pixel (80, 80) LAB: (106, 160, 102) Pixel (90, 90) LAB: (106, 160, 102) Pixel (100, 100) LAB: (105, 161, 102)

- (0.341, 0.208, 0.372)
- (0.341, 0.208, 0.372)
- (0.341, 0.208, 0.372)
- (0.341, 0.208, 0.372)
- (0.341, 0.208, 0.372)
- (0.341, 0.208, 0.372)
- (0.342, 0.209, 0.372)
- (0.342, 0.209, 0.372)
- (0.342, 0.209, 0.372)
- (0.341, 0.206, 0.372)

CIELAB 색공간에서 L을 동일하게 변경한 후 RGB 색상 비율을 맞추고 L을 다시 동일하게 맞춰주면 지역별 조도 차이가 사라진다.

<br>

------

# 5. 향후 계획

## 공통 계획

> 추론의 정확도를 최대한 높이고 개발을 마무리한다.

- [ ]  TPG 촬영
- [ ] TPG색상 추론 및 학습 튜닝
- [ ] GUI 프로그램 완성
- [ ] 발표 준비