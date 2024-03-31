---
layout: single
title:  "Weekly Diary 3주차(2024.03.18 ~ 2024.03.24)"
excerpt: "3주차 개발 기록 정리"
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

**[주간 정규 회의 기록]**  
2024-03-20 (수) 18:00 ~ 22:00  
2024-03-21 (목) 18:00 ~ 22:00  
2024-03-22 (금) 17:00 ~ 21:00  
총 활동 시간: 12시간 00분  
{: .notice--danger .text-center}

<br>

------

# **1. 주간 활동 정리**

## **1) 정규 회의 및 팀별 활동 계획**

기존에는 개발 과정에 따라 전체 인원이 참여하는 정규 회의를 위주로 활동을 진행하였다. 하지만 개발이 진행됨에 따라, 다음 주부터는 팀 활동을 위주로 개별적 활동을 위주로 진행할 것이며, 모든 팀원이 참여하는 정규 회의 시간은 단축하고자 한다. 정확한 정규 회의 시간은 추가적인 논의가 필요하나, 회당 1시간 ~ 2시간씩 주당 2회 진행하고자 한다.

각 팀은 매 주 달성하고자 하는 목표 및 계획을 해당 주의 첫 정규 회의(*월요일*)에 설정한다. 그리고 계획 대로 진행해 본 후, 논의할 사항이 있거나 진행사항이 있으면 다음 정규 회의(*목요일*)에서 공유한다. 해당 주가 끝날 때까지(*일요일*) 목표를 달성하기 위한 활동을 진행해야 하며, 달성 여부는 다음 주 첫 회의 때 정규 회의에서 보고한다.

또한 향후 개발 기록은 개인 활동 시간까지 모두 기록하기에는 어려움이 있으므로, 팀별로 진행 사항을 기록하고자 하여 **최소 활동 시간 기준인 5시간을 충족**하고자 한다.

<br>

## **2) 팀 구성 및 역할 분담**

- **팀 구분 기준**

핵심 S/W인 Calibration과 Clustering을 중심으로 하여 팀을 구분하고자 하며, 각 팀의 개발 과정에서 유기적으로 함께 개발하기 좋은 기능을 추가적으로 분담하고자 한다. 아래 내용은 핵심 S/W에 대한 설명이며, H/W 개발 및 기타 활동과 같은 부가적인 계획은 이후에 자세히 기록하고자 한다.

1. **Calibration**: Camera Module로 촬영한 Swatch를 통해 원본 색상을 추론하는 알고리즘 설계
2. **Clustering**: 추론된 원본 색상을 바탕으로 다색상 군집을 나누고, 대표 색상을 도출하는 알고리즘 설계

<br>

- **팀 구성원**
  - **Calibration 팀**: [정동교], 김우찬
  - **Clustering 팀**: [고명훈], 김성웅

<br>

## 3) 팀별 주간 계획

- **Calibration 팀** (동교, 우찬)
  - TCX 문의
  - H/W 재료 확정
  - TPG를 기반으로 한 Calibration 알고리즘 설계 및 테스트

> Linc 3.0 지원서를 작성하였다. 현재 팬톤 TCX를 확보하기 위한 예산을 배정받기 위해, 팬톤 Color code를 분석하여 내용을 추가할 계획이다. 팬톤 TCX를 대여 혹은 추가적으로 해결할 수 있는 방안을 확보하고자, **<2024.03.27 수요일>** 패션산업학과 이현승 교수님과 오전 11시 대면 상담이 예정되어 있다. 현재 패션산업학과는 팬톤 TCX를 보유하고 있지는 않다. 우리는 TPG U를 보유하고 있다. 해당 컬러북을 통해 PANTONE CONNECT를 통하여 TCX 코드로 변환할 수 있다.

<br>

- **Clustering 팀** (명훈, 성웅)
  - Raspberry Pi 동작 확인 및 카메라 세팅 확인
  - TPG 팬톤 컬러 사진 촬영 및 편집을 통해 클러스터링에 사용할 데이터 만들기
  - Clustering & DBSCAN 학습 및 아이디어 기반의 간단한 다색상 클러스터링 알고리즘 설계 및 코드 테스트
  - TPG 데이터를 통한 클러스터링 테스트 진행 후  임의 보정 값을 생성하여 비교

> 라즈베리파이 5의 운영체제 설치와 기존에 보유한 라즈베리파이 4 및 V2 카메라 모듈을 사용하여 간단한 작동 테스트 및 촬영을 진행하였다.  또한 Clustering & DBSCAN 이론에 대한 학습을 진행한 후 사이킷런 라이브러리를 이용한 DBSCAN 알고리즘 코드를 작성하여 촬영 이미지에 대한 클러스터링 테스트도 진행하였다. 마지막으로 실제 팬톤 컬러와 촬영 이미지 컬러, 클러스터링을 진행한 컬러들을 기반으로 임의의 보정 값을 생성하여 결과를 비교해 볼 수 있었다.

<br>

------

# **2. Calibration 팀 진행 사항**

## 1) 캡스톤 디자인 교과목 수강팀 LINC3.0 과제 지원 신청

> 현재 스와치의 색상에 대한 Label Data를 학습하는 과정에서, PANTONE TCX가 필요하다. PANTONE TCX의 가격은 한 장에 약 21,000원이다. 학습할 수 있는 수준의 TCX 개수를 확보하기 위해서 LINC 3.0에서 재료비를 지원받기로 한다.

![image-20240324225619539](/images/2024-03-24-Weekly diary(3주차)/image-20240324225619539.png){: .img-width-large}

지원금 지원 항목이다. 우리는 재료비에 해당하는 지원금을 신청한다.

<br>

![image-20240324225636384](/images/2024-03-24-Weekly diary(3주차)/image-20240324225636384.png){: .img-width-large}

신청서 제출 시 주의사항이다. 사진과 같이 인정 가능한 과제신청서의 형태로 제출하여야 한다. 신청서가 통과되면 아래 예시 사진과 같이 예산을 배정받을 수 있다.

<br>

![image-20240324225654773](/images/2024-03-24-Weekly diary(3주차)/image-20240324225654773.png)

![lnic](/images/2024-03-24-Weekly diary(3주차)/lnic.png)

현재까지 작성한 신청서이다. 신청서의 분량을 문의해본 후, 팬톤 TCX에 대한 예산을 받을 수 있도록 내용을 추가할 계획이다.

<br>

- **중요 내용**

  > 계획서 제출 기한: 2024.03.18 ~ 2024.04.16 (23:59:59) 교수님 서명은 촬영 이미지 인정 불가

- **TODO**

1. 팬톤 TCX의 필요성을 부각하고 예산을 확보할 수 있도록 보고서 수정
2. 개인 서명하기
3. 교수님 서명 받기

<br>

## 2) **팬톤 코튼 컬러칩**(TCX)과 관련하여 패션 산업 디자인 학과에 문의

> 보정값 생성을 위해 대량의 TCX가 필요한 상황이나, 예산적인 문제로 인해서 확보하기 힘든 상황이다. 이에 따라 패션산업학과 교수님께 연락을 드려서 도움을 받을 수 있을지 확인해 보았다.

<br>

- **문의 내용**

![KakaoTalk_20240321_151217923](/images/2024-03-24-Weekly diary(3주차)/KakaoTalk_20240321_151217923.png)

<br>

- **답변 내용**

![image-20240324230033367](/images/2024-03-24-Weekly diary(3주차)/image-20240324230033367.png)

패션산업학과 이현승 교수님께 위 사진과 같은 답변을 받았다. 우리는 **<2024.03.27 수요일>** 패션산업학과 이현승 교수님과 오전 11시 대면 상담이 예정되어 있다.

<br>

## 3) TPG 팬톤 패션 홈 인테리어 컬러 가이드 대여

> 패션 산업 학과 내에서 TCX 컬러칩을 보유하고 있지 않은 내용을 확인하고 우선적으로  패션디자인 회사에서  팬톤 패션 홈 인테리어 컬러 가이드를 4월 15일까지 대여 받았다.

![image-20240324230105445](/images/2024-03-24-Weekly diary(3주차)/image-20240324230105445.png){: .img-default-setting}

팬톤에는 크게 두 가지 컬러 시스템이 있다. 하나는 **그래픽 컬러 시스템(Pantone Matching System)**, 다른 하나는 **패션, 홈+인테리어 컬러 시스템(Fashion, Home+Interior)이다.**

각 컬러 시스템은 시장 또는 산업에 맞춰 사용할 수 있는 색상으로 설계되어 있다. 인쇄, 그래픽, 패키지 등의 산업 분야는 눈에 띄는 포인트가 되어줄 컬러 위주로, 패션 및 인테리어는 뉴트럴 컬러를 포함한 색상을 수록한다. 또한 원재료가 다른 이유로도 컬러 시스템이 분리된다.

색상은 생산되는 재료에 따라 달라질 수 있다. 실제로 일부 색상은 특정 재질에서는 절대 표현되지 않기도 한다. 각 분야에서 주로 사용되는 재료에 맞는 컬러 시스템을 사용하면 구현이 가능한지 확인하는 데에 도움이 된다.

<br>

- **그래픽 솔리드 컬러 시스템  특징**
  - 'PMS' 표시는 그래픽 시스템
  - 100 ~ 10,000번대 컬러 번호를 지닌다.
  - 숫자 뒤에 C 또는 U가 함께 표시된다.
  - C(Coated)는 코팅, U(Uncoated)는 비코팅으로 제지의 차이이다.
- **패션, 홈+인테리어 컬러 시스템 특징**
  - TPG : 종이, TCX : 면(코튼), TSX : 폴리에스터
  - TN : 나일론, TPM : 메탈릭쉬머스
  - XX-XXXX 형태의 6자리 숫자 코드를 가진다.
  - 각각의 컬러에는 색상의 영문 이름을 지닌다.

<br>

## 4) TPG를 기반으로 한 Calibration 알고리즘 설계 및 테스트

> 구매 요청한 H/W 제품들이 아직 도착하지 않은 상태이다. 따라서, 촬영 시 TPG의 RGB가 실제 RGB와 어떻게 일치하게 만들지에 대한 보정 방법과, 조도에 의한 스와치 지역별 색차를 어떻게 보정할 것인지에 대해 대략적으로 논의해보고자 한다.

기존에 제작한 샘플 H/W를 이용해 TPG를 측정한 결과, TPG의 모서리 부분에 빛이 비교적 강하게 반사되는 것을 확인할 수 있었다. TPG의 모서리 부분에서 빛이 비교적 강하게 반사되는 현상을 확인할 수 있었다. 그러나 카메라의 높이를 조정함으로써 조도 차이를 현저히 줄일 수 있었다.

아래 그림은 LED 바의 높이를 조절하여 지역별 조도 차이를 줄인 결과를 보여준다.

![image-20240324230115861](/images/2024-03-24-Weekly diary(3주차)/image-20240324230115861.png){: .img-default-setting}

![image-20240324230123157](/images/2024-03-24-Weekly diary(3주차)/image-20240324230123157.png){: .img-default-setting}

위의 사진에서 볼 수 있듯이, 육안으로 지역별 조도 차이를 느끼기 어려운 상태이다. 그럼에도 불구하고, 지역별로 미세한 조도 차이가 존재할 것이며, 이러한 조도에 의한 색차를 극복하는 것이 필수적이라고 판단한다.

기존에 계획하던 방법은 조도센서를 통해 1점법, 4점법, 5점법을 활용한 조도 측정이다. 그러나 H/W 설계 시 천장 모서리에 사각형 형태로 LED 바를 설치하게 되므로, 이러한 조도 측정법이 해당 H/W에 적합하지 않을 것으로 판단된다.

<br>

![image-20240324230145465](/images/2024-03-24-Weekly diary(3주차)/image-20240324230145465.png){: .img-width-large}

![image-20240324230157013](/images/2024-03-24-Weekly diary(3주차)/image-20240324230157013.png){: .img-width-large}

=> <u>이를 해결하기 위해서 최적의 Calibration 알고리즘을 대략적으로 고안해보기로 한다.</u>

<br>

![image-20240324230213127](/images/2024-03-24-Weekly diary(3주차)/image-20240324230213127.png)

![image-20240324230223570](/images/2024-03-24-Weekly diary(3주차)/image-20240324230223570.png)

<br>

- **Calibration** 방법을 **두 가지로 구분**하여 고려해보았다.

1. 특정 픽셀 구간별 조도차를 계산하는 **Sensor Calibration**
2. **CNN**을 활용하여 조도에 의한 보정값을 개선하는 방법

<br>

- 이번 시간에는 첫 번째 방법인 **Sensor Calibration**에 대해서 의논하였으며, 다음과 같은 절차를 고려한다.

1. Target Area의 구역별 조도에 의한 색차 보정
2. 보유한 TPG를 활용하여 지역별 색차 보정
3. 3차원 히스토그램을 통한 경향성 파악 및 보정
4. 조도 우선적 보정

<br>

- **Target Area의 구역별 조도에 의한 색차 보정**

지역별 픽셀 구역을 나누어 Calibration을 하는 방법을 생각해본다.

![image-20240324230241578](/images/2024-03-24-Weekly diary(3주차)/image-20240324230241578.png){: .img-default-setting}

측정 스와치를 36개의 구역으로 나누고, 각 구역 내에서 픽셀별로 RGB 값을 측정하여 평균 RGB 값을 도출한다.

![image-20240324230253274](/images/2024-03-24-Weekly diary(3주차)/image-20240324230253274.png){: .img-default-setting}

구역별로 정해진 대표 RGB 값을 바탕으로 Label RGB 값과 일치하도록 각 구역별로 색상 보정치를 적용한다. 이 과정을 통해 지역별 조도 차이와 색상 Calibration을 동시에 해결할 수 있다.  
=> <u>이 방법의 한계는 아래와 같다.</u>

1. 동일한 조도에서도 색상(Hue), 채도(Saturation), 명도(Value)에 따라 RGB 값의 변화 폭이 다르다.
2. 현재 보유 중인 TPG의 크기가 80mm*80mm보다 작아, Target Area의 크기와 동일한 Label로서 활용하기 어렵기 때문에 추가적인 방안이 필요하다.
3. 패턴이 있거나 다색상 스와치의 경우 한 구역 내에 여러 색상이 존재할 수 있으며, 이는 대표 RGB 값 설정에 오류를 발생시킬 수 있다.

<br>

- **보유한 TPG를 활용하여 지역별 색차 보정**

현재 보유한 TPG의 크기는 측정 대상인 스와치보다 작다. 가로와 세로 길이가 각각 45mm와 25mm인 반면, 측정해야 할 스와치 크기는 80mm*80mm로, 이로 인해 지역별 조도 차이의 경향성을 파악하는 데 어려움이 있을 수 있다.

이러한 크기 제약을 극복하기 위해, TPG를 이용한 전체 Target Area의 색차 보정 방법을 모색한다. 하나의 접근 방식은 TPG를 이동시키며 Target Area 전체를 촬영하는 것이다. 이 방법은 발생하는 작업량과 정확성의 한계를 고려해야 한다.

<br>

- **3차원 히스토그램을 통한 경향성 파악 및 보정**

촬영한 스와치의 RGB 값을 연속적인 3차원 히스토그램으로 나타내어 색차의 경향성을 파악한다. 이 접근법은 색상별 색차가 일관된 패턴을 가지는지 분석하는 데 필요하다.

![image-20240324230317973](/images/2024-03-24-Weekly diary(3주차)/image-20240324230317973.png)

![image-20240324230330681](/images/2024-03-24-Weekly diary(3주차)/image-20240324230330681.png)

<br>

- **조도 우선적 보정**

촬영한 RGB와 실제 RGB의 차이를 보정하는 것과 조도에 의한 지역별 색차를 보정하는 것을 동시에 할 수 없을 수 있다. 이러한 상황에서는 H/W 제작 단계에서 먼저 구역별 조도를 측정하고 보정하여, 조도에 의한 보정을 우선적으로 해결하는 방법을 생각해볼 필요가 있다. 조도에 의한 색차 보정값이 우선적으로 해결될 경우, 후속 작업으로 촬영한 TPG와 실제 TPG 색상을 매칭하는 절차만 진행하면 된다.

![image-20240324230348698](/images/2024-03-24-Weekly diary(3주차)/image-20240324230348698.png)

![image-20240324230402223](/images/2024-03-24-Weekly diary(3주차)/image-20240324230402223.png)

=> <u>현재 완전한 H/W 환경이 구축되지 않은 상태에서, 다음과 같은 의문을 갖게 된다.</u>

<br>

- **색상에 따른 조도에 의한 밝기(색채) 변화 차이**

  1. 색상에 따라 조도에 의한 색상 변화 폭의 차이가 존재하는가?

     - 존재한다면 색상의 범주는 어떻게 구분할 것인가?

     - 색상의 연속성에 비례하여 색상의 변화 폭도 연속적인 차이를 보이는가?

  2. 보정값을 색상별로 나누지 않고 하나로 만들 수 있을까?

<br>

- **조도 차이가 존재할까?** (*조도 센서를 통해 지역별 조도차이 확인 필요*)

  1. 조도 차이가 있다면?

     - 색상에 따른 밝기 변화가 다르다.
     - 픽셀 별 색차 폭이 커서 하나의 RGB값으로 통일할 수 없다.

  2. 조도 차이가 없다면?

     - 색상별로 촬영한 RGB와 Label RGB만 매칭하면 된다.

     - 미세한 차이는 여전히 존재하는데, 이를 간과할 수 있는 정도인가?

<br>

![image-20240324230413859](/images/2024-03-24-Weekly diary(3주차)/image-20240324230413859.png)

![image-20240324230424048](/images/2024-03-24-Weekly diary(3주차)/image-20240324230424048.png)

Calibration이 완료되었다고 가정했을 때, 다음 단계는 TPG RGB를 PANTONE CONNECT를 통해 TCX RGB로 변환하는 것이다. 학습된 TPG 색상과 가장 유사한 TCX 코드를 찾아 촬영한 샘플 스와치에 매칭하여 색상을 추론하는 기능을 구현할 수 있다. 이 과정에서 PANTONE API의 추가 비용이나 조건을 확인해야 한다.

<br>

![image-20240324230437447](/images/2024-03-24-Weekly diary(3주차)/image-20240324230437447.png)

![image-20240324230446831](/images/2024-03-24-Weekly diary(3주차)/image-20240324230446831.png)

정리하면, 하드웨어 환경의 미비로 인해 촬영된 TPG의 RGB와 실제 RGB 사이의 일치를 위한 정확한 보정에 어려움을 겪고 있다. 조도 변화에 따른 색상의 밝기 변화가 존재하는 것으로 보아, 이를 정교하게 보정할 방법을 모색 중이다. 현재로서는 조도센서를 사용한 다양한 측정 방법에 대한 적합성을 고려하고 있으며, TPG의 크기 제한으로 인한 문제점을 해결하기 위해 다각도로 접근하고 있다. 아직 확정된 방안은 없다. 조도와 색상 변화의 연속성을 어떻게 정량화할 수 있을지에 대한 연구가 필요하다.

조도 차이를 식별하고 이를 보정하는 방법은 현재 H/W 설계상의 한계로 인해 구체화되지 못했다. 색상별 보정값을 개별적으로 다루어야 할지, 아니면 통합적으로 접근해야 할지에 대한 결정도 아직 명확히 이루어지지 않았다. 이러한 문제를 해결하기 위해서는 더 많은 데이터 수집과 분석이 필요하며, 보다 H/W의 구축이 우선적으로 해결되어야 할 것 같다. H/W 구축이 완료될 경우, 해당 환경에 적절한 방안으로 모색해볼 수 있다.

<br>

------

# 3. Clustering 팀 진행 사항

## H/W 진행 사항

### Raspberry Pi 동작 확인 및 카메라 세팅 확인

> 구매 요청한  H/W제품들이 아직 도착하지 않아 현재 가지고 있는 라즈베리파이5 OS 설치 및  라즈베리파이4, 라즈베리파이 카메라 모듈 V2(8MP)을 바탕으로 OS설치 및 기본적인 카메라 작동 등의 기본적인 테스트를 진행하고자 한다.

- **라즈베리파이 5 OS 설치**

  - 윈도우에서 사용할 저장장치인 SD카드를 FAT32로 완전 포맷을 진행한다.
  - 라즈베리파이 홈페이지에서 Windows용 Raspberry PI imager를 다운 받는다.

  [Raspberry Pi OS – Raspberry Pi](https://www.raspberrypi.com/software/)

  - 라즈베리파이 Imager를 SD카드에 설치 및 실행한다.
  - OS선택 및 Erase를 진행 후 저장장치를 선택하고 쓰기를 시작한다.
  - 저장장치를 선택하기 전 옵션 항목 중 Enable SSH를 체크하고 비밀번호 변경한다.
  - Configure wifi 항목 기존에 인터넷이 연결되어 있어 따로 체크하지 않아도 된다.
  - OS 설치 완료 후 HDMI를 통해 모니터와 라즈베리파이 5를 연결하여 제대로 설치 되었는지 확인한다.
  - 추후에 원활한 편집을 위해서 원격 접속을 사용할 수 있어 SSH 원격 프로그램 설치도 고려해 본다.

<br>

- **라즈베리파이 카메라 설치 및 작동 테스트**

  > OS 설치를 진행한 라즈베리파이5에서 라즈베리파이 카메라 모듈 V2의 설치를 진행하려고 했지만 기존에 가지고 있는 카메라 케이블이 라즈베리파이 5와 호환되지 않아 OS설치가 되어있던 라즈베리파이4와 연결하여 기본적인 작동 테스트를 진행하고자 한다.

  *(라즈베리파이 3 혹은 4의 경우 0.5mm 피치의 22핀 FPC/FFC 케이블을 이용하며 라즈베리파이 5는 1.0mm 피치의 15핀 FPC/FFC 케이블을 이용한다.)*

![image-20240324230503347](/images/2024-03-24-Weekly diary(3주차)/image-20240324230503347.png){: .img-default-setting}

- 하드웨어에 연결한다.

  - HDMI 포트 옆 카메라 인터페이스에 0.5mm 피치의 22핀 FPC/FFC 케이블을 연결한다.

- libcamera를 이용하여 카메라를 호출한다.

  1. 카메라를 사용하기 전에 라즈베리파이를 최신 펌웨어로 업데이트한다.

     - 아래 명령어를 이용해 업데이트하고 재부팅 한다.

       - sudo apt-get update -y

       - sudo apt-get upgrade

  2. libcamera를 사용하려면 config.txt를 수정해야 한다.

     - 터미널에서 아래 명령어를 실행한다.

       - sudo nano /boot/config.txt

       *(config.txt에 들어갔다면 “camera-auto-Detect=1” 문구를 “**camera-auto-Detect=0**”으로 수정한다. (해당 문구가 없다면 [all] 밑에 camera-auto-Detect=0))*

  3. 재부팅 되었다면 미리 보기 화면을 출력하여 정상적으로 촬영이 되는지 확인한다.

     - 아래 명령어를 실행한다. (*창을 닫으려면 x 키 입력*)
       - sudo libcamera-hello -t 0

- libcamera에 관련한 명령어는 다음 사이트에서 추가로 확인하여 사용 가능하다.

  [Template:RPi Camera Libcamera Guide - Waveshare Wiki](https://www.waveshare.com/wiki/Template:RPi_Camera_Libcamera_Guide)

- 아래는 라즈베리파이 4와 라즈베리파이 카메라 모듈 V2를 이용하여 촬영한 이미지이다.

  - 사용한 명령어 :

  ```python
  sudo lib camera-still -o -test.jpg
  ```

![image-20240324230516293](/images/2024-03-24-Weekly diary(3주차)/image-20240324230516293.png){: .img-default-setting}

촬영한 이미지를 통해 초점 조절과 해상도와 관련하여 문제점이 있다는 사실을 확인하여 라즈베리파이 카메라 모듈 V2 제품으로는 정확하고 정밀한 TPG 촬영이 불가능하다는 사실을 확인하였다. 구매 요청한 제품이 도착하여 테스트 하는 경우에도 초점 조절에 따른 촬영 이미지 변화에 대해서도 고려해 봐야 한다는 사실을 알 수 있었다.
{: .notice--success}

<br>

## S/W 진행 사항

### 1) Clustering & DBSCAN 학습 이론 정리

- **클러스터링(Clustering)이란?**

유사한 성격을 가진 개체를 k개의 클러스터로 묶어 그룹으로 구분하는 것으로 정답 데이터가 없기 때문에 비지도 학습이다. 클러스터링 알고리즘은 거리기반의 K-means 알고리즘과 밀도기반의 DBSCAN 알고리즘으로 이루어져 있다.

<br>

- **K-means(Density-Based Spatial Clustering of Applications with Noise)이란?**

K-means는 클러스터링 알고리즘으로, 데이터를 주어진 클러스터 개수로 그룹화하고 중심과의 거리를 최소화하는 알고리즘이다. 초기 중심점 선택과 클러스터 개수는 결과에 영향을 미치며, 데이터의 불균일이나 이상치는 성능을 저하시킬 수 있다.

<br>

- **K-means (K-means clustering algorithm) 알고리즘 동작 과정**

1. 2개의 군집 중심점을 생성한다.
2. 각 데이터는 가장 가까운 중심점에 소속된다.
3. 중심점에 할당된 데이터들의 평균 중심으로 중심점을 이동한다.
4. 각 데이터는 이동된 중심점 기준으로 가장 가까운 중심점에 소속된다.
5. 다시 중심점에 할당된 데이터들의 평균 중심으로 중심점을 이동한다.
6. 중심점을 이동하였지만 데이터들의 중심점 소속 변경이 없으면 군집화가 완료된다.

<br>

- **DBSCAN(Density-Based Spatial Clustering of Applications with Noise)이란?**

**DBSCAN**이란 클러스터링 알고리즘으로 **multi dimension의 데이터**를 **밀도 기반**으로 가까운 데이터 포인터를 그룹화하는 알고리즘이다.

<br>

**-DBSCAN 클러스터링과 K-means 클러스터링의 차이**

![image-20240324230527838](/images/2024-03-24-Weekly diary(3주차)/image-20240324230527838.png){: .img-width-large}

<br>

- **DBSCAN의 주요 파라미터 2가지**

  - **eps(epsilon) :**

    클러스터의 반경을 정의하며 클러스터를 구성하는 최소 거리이다.

    너무 작은 epsilon은 더 많은 데이터 포인트를 1개의 클러스터로 묶게 된다.

    너무 큰 epsilon은 더 많은 노이즈를 생성할 수 있다.

  - **Min_samples :**

    클러스터로 간주되기 위한 최소 데이터 포인트의 수를 지정한다.

    min_samples는 클러스터로 간주되기 위한 최소 데이터 포인트의 수를 지정한다.

    값이 클수록 더 밀집된 클러스터가 형성, 값이 낮을수록 더 적은 데이터 포인트를 클러스터로 간주한다.

<br>

- **DBSCAN의 포인트 종류**

  - **핵심 포인트**(core) : 반경 내에 최소 군집 크기 이상의 데이터를 가지고 있을 경우이다.

  - **이웃 포인트**(neighbor) : 반경 내에 위치한 타 데이터이다.

  - **경계 포인트**(Border) : 반경에 최소 군집 크기 이상의 이웃 포인트를 가지고 있지 않지만, 핵심 포인트를 이웃으로 가지고 있다.

  - **잡음 포인트**(Noise) : 반경에 최소 군집 크기 이상의 이웃 포인트를 가지고 있지 않고, 핵심 포인트도 이웃으로 가지고 있지 않는다.

<br>

- **DBSCAN 알고리즘의 동작 과정**

1. 데이터 중 임의의 데이터 포인트를 선택한다.
2. 선택한 데이터와 Epsilon 거리 내에 있는 모든 데이터 포인트를 찾는다.
3. 찾은 데이터가 포인트의 개수가 Min points 이상이면, 해당 포인트를 중심으로 Cluster 생성한다.
4. 어떠한 포인트가 생성한 Cluster 안에 존재하는 다른 점 중, 다른 Cluster의 중심이 되는 데이터 포인트가 존재한다면 두 Cluster는 하나의 Cluster로 간주한다.
5. 1-4의 과정을 모든 포인트에 대해서 반복한다.
6. 어느 Cluster에도 포함되지 않는 데이터 포인트는 이상치로 처리한다.

=> <u>Clustering & DBSCAN 학습 참고 링크</u>

[클러스터링 알고리즘 (KMeans, DBSCAN) , 나만의 학원가 지도 만들기](https://www.youtube.com/watch?v=2EoBWieXcJQ)

[09-4: DBSCAN (Kor)](https://www.youtube.com/watch?v=O_EigN9iF6E)

<br>

- **Scikit-Learn 라이브러리**

  - python을 대표하는 머신 러닝 라이브러리이다.


  - scikit-learn을 설치하기 전에는 사전에 이하의 라이브러리에 대한 설치가 필요하다.

    - Numpy
    - Scipy
    - Pandas


  - scikit-learn의 주요 기능

    - 분류 : 주어진 데이터가 어느 클래스에 속하는지 판별한다.
      1. SGD : 대규모 데이터의 경우 추천한다. 선형의 클래스 분류 방법이다.
      1. 커넬 근사 : 비선형적인 클래스 분류 방법(대규모 데이터에 적합)이다.
      1. Linear SVC : 중소 규모의 경우 추천하며, 선형의 클래스 분류 방법이다.
      1. K근접 법: 비선형적인 클래스 분류 방법이다.

    - 회귀 : 전달된 데이터를 바탕으로 값을 예상한다.

    - 클러스터링 : 전달된 데이터를 규칙에 따라 나눈다.


- Scikit-Learn 학습 참고 링크

  [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)

  [사이킷런 scikit-learn 제대로 시작하기](https://www.youtube.com/watch?v=eVxGhCRN-xA)

<br>

### 2) K-means & DBSCAN 알고리즘 테스트

> Clustering & DBSCAN 이론을 통해 학습한 내용을 바탕으로 K-means & DBSCAN 코드들을 작성해보고 다색상 이미지를 통한 클러스터링 테스트 하고자 하며, 이를 위해 scikit-learn 라이브러리를 활용하여 코드를 작성해 보고자 한다.

<br>

**1. K-means Clustering 알고리즘**

- NumPy, Pandas, Matplotlib 및 scikit-learn에서 KMeans 클러스터링 알고리즘을 가져와서 데이터를 클러스터링한다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

<br>

- "color.jpg" 이미지 파일을 읽어와서 **`img`** 변수에 저장하고, 해당 이미지의 모양을 확인한다.

```python
img = plt.imread("color.jpg")
img.shape
```

출력 : `(646, 677, 3)`

<br>

- 이미지를 업로드한다.

```python
plt.imshow(img)
```

출력 :

![image-20240324230546261](/images/2024-03-24-Weekly diary(3주차)/image-20240324230546261.png){: .img-default-setting}

<br>

- 이미지를 픽셀 당 RGB 값으로 변환하고, 이를 2차원 배열로 재구성한 후 Pandas의 데이터프레임으로 변환한다. 이 데이터프레임은 각 열이 빨간색, 초록색, 파란색의 값을 나타내며, 각 픽셀의 RGB 값을 표시한다.

```python
color_tbl = img.reshape(-1,3)
pd.DataFrame(color_tbl, columns = ["Red", "Green", "Blue"])
```

출력 :

| Red    | Green | Blue |
| ------ | ----- | ---- |
| 0      | 214   | 182  |
| 1      | 206   | 174  |
| 2      | 204   | 172  |
| 3      | 203   | 171  |
| 4      | 217   | 185  |
| ...    | ...   | ...  |
| 437337 | 4     | 62   |
| 437338 | 0     | 58   |
| 437339 | 14    | 72   |
| 437340 | 0     | 55   |
| 437341 | 9     | 67   |

437342 rows × 3 columns

<br>

- 기본 클러스터 개수가 8개인 KMeans 클러스터링 모델을 초기화하고, 데이터를 사용하여 해당 모델을 fitting다. 이렇게 함으로써 데이터를 클러스터링하고 각 데이터 포인트를 해당하는 클러스터에 할당한다.

```python
km = KMeans() #기본 클러스터값은 8개
km.fit(color_tbl)
```

<br>

- KMeans 모델을 피팅한 후에 클러스터의 중심점을 나타내는 속성으로 이 배열에는 각 클러스터의 중심이 되는 점의 좌표가 포함되어있다.

```python
km.cluster_centers_
```

출력 :

```
array([[208.83649398, 189.54012167, 152.96648967],
       [ 70.77765619, 108.53261229, 175.33968698],
       [177.35653897,  95.08443857,  86.7992074 ],
       [ 14.77526749,  67.71367066, 163.82124131],
       [196.88952232, 195.20444982, 182.34264999],
       [102.52507455, 141.6152887 , 181.29601518],
       [182.337977  , 165.93462071, 149.85546411],
       [154.83309727,  74.73684904,  67.06831582]])
```

<br>

- KMeans 모델에서 얻은 클러스터의 중심점을 시각화하는 것으로 중심점은 각 클러스터의 대표적인 색상을 나타낸다.

```python
plt.imshow(km.cluster_centers_.reshape(2,4,3).astype(int))
```

출력 :

![image-20240324230558904](/images/2024-03-24-Weekly diary(3주차)/image-20240324230558904.png){: .img-default-setting}

<br>

- 클러스터링된 색상 중심점을 사용하여 이미지의 각 픽셀을 클러스터에 할당한 후, 이미지를 색상으로 다시 표시한. 각 픽셀은 해당 클러스터의 색상으로 치환된다.

```python
color8 = km.cluster_centers_.astype(int)[km.labels_].reshape(img.shape)
plt.figure(figsize=(15, 15))
plt.imshow(color8)1
```

출력 :

![image-20240324230611842](/images/2024-03-24-Weekly diary(3주차)/image-20240324230611842.png){: .img-default-setting}

K-means 알고리즘을 사용하여 테스트 해 본 결과 클러스터 개수를 사전에 지정해야 하며, 초기 중심점에 따라 결과가 달라지며 이상치에 만감하여 데이터가 균일하지 않으면 성능이 저하될 수 있다는 점을 알 수 있었다. 따라서 클러스터의 개수를 미리 지정할 필요가 없으며 **클러스터의 밀도를 고려하는 DBSCAN 알고리즘**을 사용한 시스템을 개발해보고자 한다.
{: .notice--success}

<br>

**2. DBSCAN Clustering 알고리즘**

- 이미지 클러스터링 및 시각화와 관련된 작업을 위해 필요한 주요 라이브러리 및 모듈을 가져온다.

  - NumPy 및 Pandas 라이브러리를 가져온다.

  - Matplotlib의 pyplot 모듈을 가져온다.

  - scikit-learn의 DBSCAN 클러스터링 알고리즘을 가져온다.

  - collections 모듈에서 Counter 클래스를 가져온다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
```

<br>

- "color6.jpg"라는 이미지 파일을 불러와서 **`plt.imshow()`** 함수를 사용하여 이미지를 플로팅한다.

```python
img = plt.imread("color.jpg")
#img = img[::2, ::2, :] #이미지 크기 /2로 조정
img.shape
```

출력 : `(646, 677, 3)`

<br>

- 이미지 파일을 불러온다.

```python
plt.imshow(img)
```

출력 :

![image-20240324230624603](/images/2024-03-24-Weekly diary(3주차)/image-20240324230624603.png){: .img-default-setting}

<br>

- 이미지를 픽셀 당 RGB 값으로 변환하고, DBSCAN 모델을 사용하여 픽셀 데이터를 클러스터링다. 그 후, 각 클러스터에 대한 색상을 지정하여 시각화에 사용할 준비를 한다.

```python
#DBSCAN 모델 생성
dbscan = DBSCAN(eps=5, min_samples=50)

color_tbl = img.reshape(-1, 3)  # 이미지를 픽셀 당 RGB값으로 변환
dbscan.fit(color_tbl)  # 모델 fitting

# 클러스터 레이블
labels = dbscan.labels_
unique_labels = set(labels)

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))] 
```

<br>

- 이미지의 각 픽셀에 대한 RGB 값이 테이블 형식으로 표시한다.

```python
color_tbl = img.reshape(-1, 3)
#이미지를 픽셀 당 RGB값으로 변환하며 -1은 나머지 차원을 자동으로 조정하라는 의미이며 3은 각 픽셀을 RGB값으로 레이블링 되는 것
pd.DataFrame(color_tbl, columns=["Red", "Green", "Blue"])
#변환된 배열을 pandas의 데이터 프레임으로 변환한다.
```

출력 :

|        | Red  | Green | Blue |
| ------ | ---- | ----- | ---- |
| 0      | 214  | 182   | 159  |
| 1      | 206  | 174   | 151  |
| 2      | 204  | 172   | 149  |
| 3      | 203  | 171   | 148  |
| 4      | 217  | 185   | 162  |
| ...    | ...  | ...   | ...  |
| 437337 | 4    | 62    | 159  |
| 437338 | 0    | 58    | 155  |
| 437339 | 14   | 72    | 169  |
| 437340 | 0    | 55    | 152  |
| 437341 | 9    | 67    | 164  |

=> <u>437342 rows × 3 columns</u>

<br>

- 클러스터링된 데이터 포인트가 3차원 공간에서 시각화한다. 각 클러스터의 대표 색상은 해당 클러스터의 데이터 포인트를 통해 추정한다.( 클러스터된 수 중 1개는 노이즈 영역이다.)

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
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
plt.title('Estimated number of clusters: %d' % (len(unique_labels)))
# 범례 표시
ax.legend()
# 그래프 출력
plt.show()
```



출력 :

```
Cluster 0 RGB: [195 183 161], Cluster 1 RGB: [170  85  77], Cluster 2 RGB: [ 76 115 185]  Cluster 3 RGB: [103 143 171], Cluster 4 RGB: [128 111  84], Cluster 5 RGB: [ 12  67 167]  Cluster 6 RGB: [59 75 88], Cluster 7 RGB: [111  94  63], Cluster 8 RGB: [167 125 100]  Cluster 9 RGB: [64 83 99], Cluster 10 RGB: [ 71  91 107]
```

![image-20240324230645039](/images/2024-03-24-Weekly diary(3주차)/image-20240324230645039.png){: .img-default-setting}

<br>

- 각 클러스터의 대표 색상을 추출하고 대표 색상에 해당하는 RGB 색상 팔레트를 생성한다.

```python
# unique_labels의 순서를 기준으로 색상을 선택하여 이미지로 출력
plt.figure(figsize=(15, 15))
plt.title("Clustered Colors")

num_cols = 3  # 한 행에 출력할 색상 개수

for i, k in enumerate(unique_labels):
    if k == -1:
        continue  # Noise는 제외
    col = colors[i]
    
    # 각 클러스터에 해당하는 대표 색상 계산
    xy = color_tbl[labels == k]
    representative_color = np.mean(xy, axis=0)
    
    # 대표 색상에 해당하는 이미지 출력
    plt.subplot(len(unique_labels) // num_cols + 1, num_cols, i + 1)
    cluster_color = np.full((100, 100, 3), representative_color, dtype=int)  
    # 대표 색으로 채워진 이미지 생성
    plt.imshow(cluster_color)
    plt.axis('off')
    plt.title(f"Cluster {k} RGB: {representative_color.astype(int)}")

plt.tight_layout()
plt.show()
```

출력 :

![image-20240324230656224](/images/2024-03-24-Weekly diary(3주차)/image-20240324230656224.png){: .img-default-setting}

<br>

- 이 코드는 클러스터링된 색상 중심점을 사용하여 이미지의 각 픽셀을 클러스터에 할당한 후, 이미지를 색상으로 다시 표시다. 각 픽셀은 해당 클러스터의 색상으로 치환한다.

```python
plt.figure(figsize=(15, 15))
plt.imshow(cluster_centers[labels].reshape(img.shape).astype(int))
for label, color in enumerate(cluster_centers):
    if label == -1:
        continue
    print(f"Cluster {label} RGB: {color.astype(int)}")
plt.show()
```

출력 :

![image-20240324230709087](/images/2024-03-24-Weekly diary(3주차)/image-20240324230709087.png){: .img-default-setting}

<br>

- DBSCAN 알고리즘을 적용한 코드를 테스트 해 봄으로 다음과 같은 개선사항에 대해 생각해 볼 수 있었다.  

1. 클러스터링 결과를 더 자세히 분석하여 클러스터링 알고리즘의 파라미터를 조정하고 최적화할 필요가 있다.   
2. 추출된 대표 색상들이 이미지에 얼마나 잘 반영되었는지에 대한 평가 및 분석이 필요하다.  
3. 시각화된 이미지나 그래프를 통해 클러스터링 결과를 해석하고 다른 데이터셋에 대한 일반화 가능성을 고려해봐야 한다. 

<br>

**3. DBSCAN Clustering 알고리즘 2**

- 원래 테스트 용으로 사용하고자 했던 이미지는 위 알고리즘에서 사용한 8 컬러 이미지와 같다. 주어진 이미지는 437,342개의 픽셀로 구성되어 있다. 그러나 클러스터링을 시도한 결과, **모든 색상이 하나의 클러스터로 묶이거나 너무 많은 클러스터가 생성되는 등 적절한 하이퍼 파라미터를 찾기 어려워 좋은 결과를 얻지 못했다.** 이는 이미지가 너무 많은 색상을 포함하고 있으며 고해상도이기 때문에 픽셀 수가 많아서 발생한 것으로 판단된다. 따라서 이러한 이유로, 이미지에서 주요한 두 가지 컬러 톤을 가진 작은 부분 이미지를 추출하여 픽셀 수를 줄인 후 클러스터링 테스트에 사용하였다. 아래의 이미지는 12,204개의 픽셀을 가지고 있다.

![image-20240324230725319](/images/2024-03-24-Weekly diary(3주차)/image-20240324230725319.png){: .img-default-setting}

```python
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN 
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
```

> 필요한 라이브러리 import

<br>

```python
def png_to_dataset(image_path):
    # 이미지 열기
    img = Image.open(image_path)
    
    # 이미지의 너비와 높이 가져오기
    width, height = img.size
    
    # 데이터셋 초기화
    dataset = []
    
    # 이미지의 각 픽셀에서 RGB 값을 추출하여 데이터셋에 추가
    for y in range(height):
        for x in range(width):
            # 해당 좌표의 RGB 값을 가져옴
            pixel = img.getpixel((x, y))
            # PNG 이미지의 경우, RGBA 값을 가져오므로 RGB로 변환하여 데이터셋에 추가
            if len(pixel) == 4:  # Check if it's RGBA
                r, g, b, a = pixel
                dataset.append((r, g, b))
            else:
                r, g, b = pixel
                dataset.append((r, g, b))
    
    return dataset
image_path = "test_image1.png"  # 이미지 파일 경로를 제공
dataset = png_to_dataset(image_path)
print("데이터셋 길이:", len(dataset))
```

> png → dataset convert

<br>

```python
# Figure 생성
fig = plt.figure()
fig.set_facecolor('white')

# 3D subplot 생성
ax = fig.add_subplot(111, projection='3d')

# 데이터셋에서 RGB 값을 x, y, z로 분리
x = [pixel[0] for pixel in dataset]
y = [pixel[1] for pixel in dataset]
z = [pixel[2] for pixel in dataset]

# 3D 산점도 그리기
ax.scatter(x, y, z)

# 축 레이블 설정
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()
```

> 3차원 RGB공간상에 모든 픽셀 표현

<br>

![image-20240324230738489](/images/2024-03-24-Weekly diary(3주차)/image-20240324230738489.png){: .img-default-setting}

```python
db = DBSCAN(eps=4, min_samples=10).fit(dataset)
labels = db.labels_
from collections import Counter
Counter(labels)
```

> **Counter(labels)**은 ****각 클러스터에 몇개의 데이터가 속해 있는지 알려준다.

<br>

- example case의 출력 
  - Counter({0: 4670, 4: 9, -1: 703, 1: 6802, 2: 10, 3: 10}) 
  - 0번 클러스터 : 4670개 
  - 1번 클러스터 : 6802개 
  - 2번 클러스터 : 10개 
  - 3번 클러스터 : 10개 
  - 4번 클러스터 : 9개 
  - -1번 클러스터 : 703개


> DBSCAN에서는 노이즈 포인트(*비핵심 포인트, 클러스터에 포함되지 않는 포인트*)를 -1번으로 레이블링 한다.

<br>

```python
# Figure 생성
fig = plt.figure()
fig.set_facecolor('white')

# 3D subplot 생성
ax = fig.add_subplot(111, projection='3d')

# 클러스터 레이블에 따라 산점도 그리기
for label in set(labels):
    # 클러스터에 속하는 데이터 포인트 인덱스 추출
    indices = np.where(labels == label)[0]
    # 데이터 포인트의 RGB 값을 추출하여 좌표로 사용
    x = [dataset[i][0] for i in indices]
    y = [dataset[i][1] for i in indices]
    z = [dataset[i][2] for i in indices]
    # 각 클러스터에 대한 산점도 그리기
    ax.scatter(x, y, z, label=f'Cluster {label}')

# 축 레이블 설정
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# 범례 추가
ax.legend()

plt.show()
```

![image-20240324232601365](/images/2024-03-24-Weekly diary(3주차)/image-20240324232601365.png){: .img-default-setting}

> 두가지 컬러톤의 집합이 각각 클러스터 0,1로 잘 분리되는 것을 알 수 있다.

<br>

- 클러스터링은 잘 분리되었지만, 이어서 확인해야 할 포인트는 다음과 같다.

1. 한 클러스터 내의 포인트들이 한 컬러톤으로 간주될 수 있는지 확인한다.
2. 다양한 테스트 이미지에서도 잘 작동하는지 확인한다.
3. 3색 이상의 다양한 색상에서도 적용 가능한지 확인한다.
4. 최적의 하이퍼 파라미터를 찾는다. 

<br>

### 3) TPG 팬톤 가이드 촬영 및 작성한 코드를 통한 Clustering Test

> Raspberry Pi와 카메라 세팅을 통해 가지고 있는 TPG 샘플을 찍어 편집할 계획이었지만 V2 카메라 모듈의 초점 조절 및 해상도 문제로 인하여 S20 스마트폰 카메라의 자동 보정 값을 제거한 후 임시로 제작한 H/W에서 6가지 TPG 샘플을 촬영하여 작성하였던 DBSCAN 클러스터링 코드를 통하여 테스트를 진행하고자 한다.

6개의 색상 중 5개의 색상은 잘 분류가 되었지만 1개의 색상이 잘 되지 않는 문제가 발생했다.  테스트 이미지를 확대했을 때 노이즈가 존재했기 때문에 노이즈를 제거해주는 스무딩 등의 전처리 과정이 필요한 것을 확인하였다.

<br>

- TPG 팬톤 가이드의 촬영할 샘플마다 H/W 바닥 면의 중심 점에 놓고  촬영을 진행하였다.

![image-20240324230814042](/images/2024-03-24-Weekly diary(3주차)/image-20240324230814042.png){: .img-default-setting}

<br>

- 아래는 촬영한 6가지 TPG 샘플의 일부분을 편집하여 이어 붙인 이미지이다.

![image-20240324230823493](/images/2024-03-24-Weekly diary(3주차)/image-20240324230823493.png){: .img-default-setting}

<br>

- 이미지 파일의 각 영역의 팬톤 색상은 다음과 같다. (맨 위 왼쪽부터 오른쪽 순서대로)

| Indigo Bunting | Gibraltar Sea | Blue Opal |
| -------------- | ------------- | --------- |
| Sailor Blue    | Moonlit Ocean | Deep Dive |

<br>

- 각 팬톤 색상 구역별로 임의의 좌표에서 그림판의 스포이드를 통해 측정한 RGB 값이다.

| Indigo Bunting (0,68,141) | Gibraltar Sea (11,31,58) | Blue Opal (10,33,65) |
| ------------------------- | ------------------------ | -------------------- |
| Sailor Blue (14,41,68)    | Moonlit Ocean (26,39,56) | Deep Dive (20,47,68) |

<br>

- 이미지를 작성한 코드를 바탕으로 DBSCAN 클러스터링 하였을 때의 각 색상의 대표 RGB값이다.

| Indigo Bunting (0,68,141)  | Gibraltar Sea (11,31,61) | Blue Opal (12,41,67) |
| -------------------------- | ------------------------ | -------------------- |
| Sailor Blue (14,41,68) (?) | Moonlit Ocean (26,39,57) | Deep Dive (20,47,68) |

<br>

- 촬영한 TPG 팬톤 컬러의 RGB값 데이터이다.

| Indigo Bunting (0, 108, 169) | Gibraltar Sea (18,56,80)   | Blue Opal (16,61,84)   |
| ---------------------------- | -------------------------- | ---------------------- |
| Sailor Blue (14, 58, 83)     | Moonlit Ocean (41, 49, 77) | Deep Dive (41, 73, 92) |

<br>

- TPG 팬톤 컬러의 RGB 값과 클러스터링을 진행하였을 때의 RGB의 각 영역 값의 차이다.

| Indigo Bunting (0, 40, 28) | Gibraltar Sea (7, 25, 22)  | Blue Opal (6, 28, 19)  |
| -------------------------- | -------------------------- | ---------------------- |
| Sailor Blue (0, 17, 15)    | Moonlit Ocean (15, 10, 21) | Deep Dive (21, 26, 24) |

<br>

- RGB값의 각 R, G, B 영역의 평균 차이 (RGB 보정 값)이다.
  - R 평균 차이  : 8.17
  - G 평균 차이 : 24.33
  - B 평균 차이 : 21.5

<br>

- 촬영한 이미지에 RGB 평균 차이 값을 더해보았다.

  > (팬톤 색상 이름) : PANTONE (팬톤 색상 RGB값)  
  > (촬영한 이미지의 RGB값) + (RGB 보정 값) = (이미지에 보정 값을 더한 RGB 값)

<br>

- **indigo Bunting : PANTONE (0, 108, 169)**

  - (0, 68, 141) + (8, 24, 21) = (8, 92, 162)

    ![image-20240324231004398](/images/2024-03-24-Weekly diary(3주차)/image-20240324231004398.png)

- **Gibraltar Sea : PANTONE (18, 56, 80)**

  - (11, 31, 58) + (8, 24, 21) = (19, 55, 79)

    ![image-20240324231017579](/images/2024-03-24-Weekly diary(3주차)/image-20240324231017579.png)

- **Blue Opal : PANTONE (16, 61, 84)**

  - (10, 33, 65) + (8, 24, 21) = (18, 57, 86)

    ![image-20240324231030622](/images/2024-03-24-Weekly diary(3주차)/image-20240324231030622.png)

- **Sailor Blue : PANTONE (14, 58, 83)**

  - (14, 41, 68) + (8, 24, 21) = (22, 65, 89)

    ![image-20240324231047116](/images/2024-03-24-Weekly diary(3주차)/image-20240324231047116.png)

- **Moonlit Ocean : PANTONE (41, 49, 77)**

  - (26, 39, 56) + (8, 24, 21) = (34, 63, 77)

    ![image-20240324231056736](/images/2024-03-24-Weekly diary(3주차)/image-20240324231056736.png)

- **Deep Dive : PANTONE (41, 73, 92)**

  - (20, 47, 68) + (8, 24, 21) = (28, 71, 89)

    ![image-20240324231104974](/images/2024-03-24-Weekly diary(3주차)/image-20240324231104974.png)

<br>

- 이미지에 보정 값을 더한 RGB 값이 해당 팬톤 색상의 RGB값보다 전체적으로 어둡게 나왔음을 확인하였다. 조금 더 밝게 만들 방안이 필요할 수도 있다.

1. 한 색 계열 전부를 찍어봐야 한다.
2. 모든 색 계열 전부를 찍어봐야 한다.
3. 이를 바탕으로 Calibration 방안을 고안하고 Clustering을 진행해야 한다.

<br>

- TPG RGB값을 TCX의 RGB 값으로의 변경이 필요한데, API를 사용할 수 있을지 확인 필요하다.

  - 다음은 현재 팬톤 TPG 컬러와 가장 유사한 TCX 컬러 찾을 수 있는 방법이다.

  1. 팬톤 컬러 찾기 사이트 Pantone Connect에 접속한다.
  2. 팬톤 커넥트 로그인 후 변환 카테고리에 접속한다.
  3. TPG 컬러에 해당하는 RGB값을 입력한 후 변환하는 시스템을 PANTONE FHI Cotton TCX 2625로 선택한 후 Convert하여 가장 유사한 RGB값을 가진 TCX 팬톤 컬러를 확인한다.

- 회전도 고려하여 알고리즘을 설계해야 한다.

- 색 계열 뿐만 아니라 색 속성(HSV)에 의한 밝기 차이도 존재하므로, 색 속성에 따른 변화 차이도 고려해야 하는 사항이다.

<br>

------

# 4. 공통 작업

## Github 메인 페이지 개선

![image-20240324231124703](/images/2024-03-24-Weekly diary(3주차)/image-20240324231124703.png)

- Weekly Diary Table 카테고리 추가
- Projects 카테고리 추가
- Team Repository 카테고리 추가

<br>

------

# 5. 향후 계획

## Calibration 팀 계획

- [ ] TCX문의 : 수요일 오전 11시 패션산업학과 교수님 대면 미팅
- [ ] H/W 재료 확정: 아크릴 재설계 및 주문, Linc 3.0 지원서 (*추가 작성, 교수님 날인 받기*)
- [ ] TPG를 기반으로 한 Calibration 알고리즘 설계 및 테스트 :   
  색상 매칭 보정값, 조도 차이에 의한 지역별 보정값 생성 방법 고안

<br>

## Clustering 팀 계획

- [ ]  DBSCAN 알고리즘 개선 + 최적화 파리미터 값 테스트
- [ ] 라즈베리파이 5 및 구매 요청한 H/W 제품에 대한 작동 테스트
- [ ] TPG 추가 학습을 통한 데이터셋에 대한  일반화 가능성 고려

<br>

## 공통 계획

팀원들과 논의한 결과, 향후 정규 회의는 다음과 같이 진행하고자 한다.

- (월) 12:30 ~ 13:30
- (목) 11:00 ~ 12:00

<br>