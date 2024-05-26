---
layout: single
title:  "Weekly Diary 12주차(2024.05.20 ~ 2024.05.26)"
excerpt: "12주차 개발 기록 정리"
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

# 2. 공통 활동

## **1) TPG 촬영**

> 데이터셋 TPG 약 10400장 가량을 모두 촬영하였다.

![image-20240526234216787](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234216787.png){: .img-default-setting}

![image-20240526234223834](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234223834.png){: .img-default-setting}

전체 데이터셋 촬영에 총 72시간이 소요된 만큼, 금주의 활동은 대체로 촬영을 진행하였다.

<br>

## **2) 코드 클래스화**

> 실제 GUI를 통해 사용자가 시스템을 이용할 수 있도록, 모든 기능을 구현하는 하나의 통합 코드가 필요하다. 따라서, 각 기능을 구현하는 자신의 코드를 클래스화 한다.

**완료**

- Camera Capture
- Hue Balancing

**미완료**

- DBSCAN
- Color Inference
- FC(Fringing Correction)
- AC(Adaptive Convolution)
- BC(Brightness Correction)

<br>

------

# 3. CI 팀

## **1) CNN MobileNet 학습 시간 개선**

> 라즈베리 파이에서 MobileNet CNN(  10 epoch lr=0.001 )모델을 사용하여 기존의 100x100 사이즈의 800장 DataSet을 학습하는데도 많은 시간이 소요되었기 때문에 기존의 PyTorch 대신 ***1) 경량화된 PyTorch Mobile로 모델 학습 및 추론하는 방식***과 ***2) 로컬 머신에서 학습을 완료한 후, 모델을 라즈베리 파이로 전송하여 추론 과정만 수행하는 방식***  에 대해 확인해 보았다.

<br>

***1) PyTorch Mobile 방식***

- PyTorch Mobile은 메모리 사용량이 적고, 모바일 및 임베디드 장치에서 효율적으로 실행되도록 최적화된 모델로 기존의 PyTorch보다는 효율적이며 기본적으로 PyTorch와 동일한 모델 아키텍처와 가중치를 사용하기 때문에 모델의 정확성에는 차이가 없지만 모델 최적화 과정에서 약간의 손실이 발생할 수 있다.

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.mean(image.view(3, -1), dim=1)  # 채널별 평균 계산
        
        return image, label

# 데이터셋 및 데이터로더 생성
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomDataset(root_dir='new_train', transform=transform)
data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

class MobileNetLike(nn.Module):
    def __init__(self):
        super(MobileNetLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
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

model = MobileNetLike()
model.eval()

# 입력 예제를 사용하여 모델을 추론하고 TorchScript로 변환합니다.
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 모델을 TorchScript로 저장합니다.
traced_model.save("mobilenet_like.pt")

# 로드된 모델을 추론에 사용할 수 있습니다.
# loaded_model = torch.jit.load("mobilenet_like.pt")
# outputs = loaded_model(example_input)

# 학습을 위해 GPU를 사용할 수 있는지 확인합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 코드
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    
    for images, labels in data_loader:
        if images is None or labels is None:
            continue
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}')
```

기존의 PyTorch와 다른 부분은 PyTorch는 모델 저장 시 TorchScript를 사용하여 모델을 직렬화 하지만 PyTorch Mobile 방식에서는 모델을 저장하기 전에 **`torch.utils.mobile_optimizer`**를 사용하여 모델을 모바일 장치에 맞게 최적화하며 TorchScript 모델을 로드하여 추론에 사용하는 방식에서 차이점이 있다. 실행 시간에서는 우선적으로 PC에서 CUDA를 적용하여 테스트한 결과 학습률 0.001, epoch은 20 기준으로 PyTorch는 2m50s , PyTorch Mobile은 2m20s 로 30초의 학습 시간 차이를 확인 할 수 있었다. 이를 통해 라즈베리파이에서 CNN 모델을 학습한다고 가정 시 PyTorch Mobile 버전을 사용하면 기존보다 확연한 개선점을 얻을 수 있을 것이라고 기대해 볼 수 있을 것 같다.

<br>

***2) 로컬 머신에서 학습을 완료한 후, 모델을 라즈베리 파이로 전송하여 추론 과정만 수행하는 방식***

PyTorch Mobile을 통해 충분한 개선된 사항을 확인할 수 있었지만 여전히 학습 시간에서는 오랜 시간이 걸렸음을 확인할 수 있었기 때문에 학습을 로컬머신에서 학습 후 모델을 라즈베리파이로 전송하여 라즈베리파이에서는 추론 과정만 수행하는 방식을 구동해 보았다.

1. 라즈베리 파이로 모델 파일 전송을 위한 Paramiko와 SCP
   - 모델 저장 : 학습된 PyTorch 모델의 파라미터를 `model.pth` 파일로 저장한다.
2. 라즈베리 파이 정보 설정
   - 모델 파일이 저장될 라즈베리 파이의 경로와, 라즈베리 파이의 IP 주소, 사용자 이름, 비밀번호를 설정한다.
3. SCP 클라이언트 생성
   - Paramiko를 사용하여 SSH 클라이언트를 생성하고, SCP 클라이언트를 반환하는 함수이다.
4. 파일 전송
   - SCP 클라이언트를 생성하고, **`model.pth`** 파일을 라즈베리파이로 전송합니다. 전송 완료 후 SCP 연결을 닫는다.

```python
import os
import torch
import paramiko
from scp import SCPClient

# 모델을 파일에 저장
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)

# 라즈베리 파이로 전송할 모델 파일을 저장한 경로
destination_path_on_pi = '/home/pi/DBSCAN_test/model.pth'

# 라즈베리 파이의 IP 주소 또는 호스트 이름
pi_address = '192.168.0.20'
pi_username = 'pi'
pi_password = '000'  # 라즈베리 파이의 비밀번호를 입력하세요

# Paramiko를 사용하여 파일을 전송
def create_scp_client(host, port, username, password):
    """Creates an SCP client connected to the specified host."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, username, password)
    scp = SCPClient(client.get_transport())
    return scp

# 파일 전송
try:
    scp = create_scp_client(pi_address, 22, pi_username, pi_password)
    scp.put(model_path, destination_path_on_pi)
    scp.close()
    print(f"Model file {model_path} successfully transferred to {destination_path_on_pi} on {pi_address}")
except Exception as e:
    print(f"Error transferring model file: {e}")
```

***⇒***

***Model file model.pth successfully transferred to /home/pi/DBSCAN_test/model.pth on 192.168.0.20***

학습된 모델을 라즈베리 파이에 전송하여 추론만 수행하는 방식이 정상적으로 작동함을 확인할 수 있었다. 이 방법을 통해 라즈베리 파이에서 모델 추론을 문제 없이 수행할 수 있음을 확인할 수 있었기 때문에 PyTorch Mobile의 추가적인 최적화가 어려운 경우, 로컬 머신에서 모델 학습을 완료한 후 라즈베리 파이로 전송하여 추론 과정만 수행하는 방식을 채택하는 것이 더 적절할 것으로 보인다.

<br>

## 2) GUI 개발

> 지난주에 이어서 GUI를 개발하는한편 다른 모듈화한 코드들에 최적화된 UI를 위해 구조를 재설계하고 보완하며 개발을 진행했다.

### Flow chart 보완

현재 추가할 기능을 고려해서 최소한의 동작을 하도록 설계하기 위해 flow chart를 고안해보았다.

![image-20240526234245670](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234245670.png){: .img-default-setting}

<br>

### 파일 구조 설계

효율적인 개발과 코드 재사용, 수정 용이성을 위해 파일 구조를 재설계 하였다.

**GUI**  
config: 각종 구성 요소  
img: 출력할 이미지  
label: 기록을 불러올때 사용할 사진 이름과 index저장  
page: 출력한 페이지 코드  
scripts: 색상 추론 등 불러와서 사용할 실행 코드  
src: 배경 이미지, 로딩 창 등 소스

<br>

### 실행 예시

![image-20240526234256039](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234256039.png){: .img-default-setting}

<br>

### 코드 설명

**페이지 class 공통 구조**

```python
class ex(tk.Toplevel):
	def __init__(self, master,main_window, ready_callback=None):
        super().__init__(master)
        self.main_window = main_window
        self.ready_callback = ready_callback
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Create buttons for analysis options
        button1 = tk.Button(self, text="ex1", command=self.open_page, font=self.window_config.main_font)
        back_button = tk.Button(self, text="Back", command=self.go_back, font=("Arial", 8))
        
        # Place buttons
        button1.place(x=self.button_config.button_pos1_x, y=self.button_config.button_pos1_y, width=self.button_config.button_type1_width, height=self.button_config.button_type1_height)
        back_button.place(x=self.button_config.back_button_x, y=self.button_config.back_button_y, width=self.button_config.back_button_width, height=self.button_config.back_button_height)

    def prepare_page(self):
        """Prepare the page by arranging widgets and loading data, then call the ready callback."""
        if self.ready_callback:
            self.ready_callback()

    def open_page(self):
        """Open another page"""
        self.analysis_page = new_page(self.master, self.main_window, lambda: self.master.destroy())
        
    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
```

window_config, button_config, txt_config는 각각 윈도우창, 버튼, 텍스트 라벨링에 대한 설정 함수를 가지고있고 배경 설정과 상단 바 제거등의 함수는 window_config class에 포함되어있다. prepare_page함수를 통해 다음 페이지가 로딩이 완료되었을때 콜백을 보내고 레디 콜백을 받은 이후 현재 페이지를 지우고 새 페이지를 플로팅해 페이지 이동 간 공백이 생기지 않도록 한다. 초기 main_window를 계속 전달해 주면서 돌아갈 main_screen을 지정한다.

<br>

**서브 프로세스를 통한 코드 실행**

```python
def run_CI(self):
        """Run the CI process by executing a script."""
        script_path = os.path.join(os.path.dirname(__file__), "../scripts", "waiting.py")
        if os.path.exists(script_path):
            self.process = subprocess.Popen(["python3", script_path], cwd=os.path.dirname(script_path))
            self.after(100, self.check_process)
        else:
            messagebox.showerror("Error", f"Script not found: {script_path}")

def check_process(self):
    """Check the status of the CI process and handle completion."""
    if self.process.poll() is None:
        self.after(100, self.check_process)
    else:
        if self.process.returncode == 0:
            self.analysis_page = AnalysisPage1_2(self.master, self.main_window, lambda: self.master.destroy())
        else:
            messagebox.showerror("Error", f"Script failed with return code {self.process.returncode}")
```

UI가 외부 스크립트의 실행으로 멈추지 않고 비동기적으로 외부 스크립트를 실행하기 위한 코드이다. run_CI함수에서 프로세스를 실행하고 check_process함수를 실행한다. check_process 함수에서는 self.process.poll()를 호출하여 프로세스가 실행중인지 확인하고 아직 실행중이라면 100ms이후 다시 확인한다. 만약 프로세스가 완료되었다면 poll()의 반환코드가 0이고 이때 다음 페이지로 이동한다.

<br>

**카메라 프리뷰**

```python
def update_preview_new(self):
        """Update the preview with a new image from the camera."""
        command = "rpicam-jpeg -o /home/pi/project/GUI/tempPreview/img.jpg -t 100 --width 480 --height 360 -n"
        subprocess.run(command, shell=True)

        """Load the image from the camera and display it."""
        if os.path.exists("/home/pi/project/GUI/tempPreview/img.jpg"):
            image = Image.open("/home/pi/project/GUI/tempPreview/img.jpg")
            photo = ImageTk.PhotoImage(image)

            """Display the image in the label."""
            self.image_label.image = photo
            self.image_label.configure(image=photo)
```

사용자가 스와치 촬영을 위해 기기 내부에 스와치를 두었을때 적절한 위치에 놓았는지 확인하기 위한 프리뷰를 제공하는 함수이다. 비동기적으로 shell커맨드를 통해 rpicam촬영 명령을 해서 임시 저장소에 저장을 하고 불러와 label의 이미지로 설정한다. 추후 시간이 남는다면 각종 오픈소스 코드를 이용해서 구역내에 놓인 스와치를 인식해서 적절히 두었는지에 대한 피드백을 주는 방법 또한 고려할 수 있다.

<br>

------

# 4. AC 팀

## **1) Brightness Addaptive Convolution**

> 지역적 조도 차이를 보정하기 위한 밝기 가중 컨볼루션이다. MCU의 도입을 통해 특정 구역별 색상을 명확하게 구분하며 이미지를 축소할 수 있다. 이전 단계에서 제시한 Nesting AC 알고리즘을 통해 지역별 대표 색상을 극대화한다.

![image-20240526234309723](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234309723.png){: .img-default-setting}

코드번호 19-1606의 2500*2500 TPG combined 이미지이다. 이 이미지를 두 번의 MCU 축소 과정을 거치면 다음과 같은 이미지를 생성할 수 있다.

![image-20240526234334672](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234334672.png){: .img-default-setting}

밝기를 구하는 공식은 다음과 같다.

![image-20240526234341249](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234341249.png)

이 값을 기반으로 픽셀별 밝은 부분에 비중을 두어 조도 차이를 보정한다.

![image-20240526234347424](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234347424.png){: .img-default-setting}

<br>

## 2) Finging Correction

색수차 보정(Fringing Correction)은 저가형 렌즈 사용시 백색광에 포함된 여러 파장의 빛이 렌즈를 통과하여 굴절될 때 동시에 상이 맺히지 못하여 생기는 현상으로, 추론에 영향을 줄 수 있기 때문에 별도의 보정이 필요하다. 이에 따라 S/W적으로 보정을 시도하였다.

<br>

- **예시 이미지 - 1**

![image-20240526234354721](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234354721.png){: .img-default-setting}

위 이미지는 촬영된 이미지 데이터 중 하나를 가져온 것이다. 확인 결과, 촬영 과정에서 미처 처리하지 못한 흰색 잡티(각종 *먼지와 원단 실로 확인됨*)와 푸른색 색수차가 나타나는 것을 확인할 수 있다.

- **보정 결과 - 1**

![image-20240526234402420](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234402420.png){: .img-default-setting}

보정 과정을 수행하면 위 이미지를 이와 같이 보정할 수 있게 된다.

![image-20240526234409193](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234409193.png){: .img-default-setting}

보정된 이미지에 대해서 밝기 Heatmap을 적용하면 이와 같이 나온다. 여전히 미세한 밝기 차이가 잡히는 것은 확인되나, 이후에 AC 과정을 거치면 세세한 지역적 특성보다는 전반적인 경향성을 가져가기 때문에, 보정 전 보다 훨씬 더 좋은 결과를 얻을 수 있을 것이라고 생각한다.

<br>

- **예시 이미지 - 2**

![image-20240526234421163](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234421163.png){: .img-default-setting}

- **보정 결과 - 2**

![image-20240526234427763](/images/2024-05-26-Weekly Diary(12주차)/image-20240526234427763.png){: .img-default-setting}

<br>

본 과정을 수행하는 과정을 대략적으로 설명하면 다음과 같다.

1. 이미지의 세로 방향에서 나타나는 색수차를 보정한다.
2. 이미지에서 나타나는 가로 방향 색수차 영역을 제외한, 나머지 상/하 영역을 보정한다.
3. 이미지에서 나타나는 가로 방향 색수차 영역을 보정한다.
4. 보정된 이미지를 새로운 이미지로 만든다.

> 각 과정은 먼저 BoxPlot을 사용해 Fringing이 나타나는 Row, Column, Pixel을 파악하고, 주변 값을 사용해 interpolation을 한 다음, 적절한 밝기를 표현하기 위하여 Kernel Sliding을 거친 후에 Random Shuffling을 수행하는 방식을 포함한다.

<br>

대체적으로 문제 없이 적용이 가능했으나 한 가지 문제가 발생했다. 현재 렌즈의 특성 상 푸른색 색수차가 나타나기 때문에 색수차가 나타나는 Pixel을 판단하는 기준을 RGB 값 중 Blue Value가 Anomaly Point를 넘어가는 경우에 대해서 적용되도록 설정했다. 그런데, 1번 보정을 수행하는 과정에서 Blue Pixel 값의 분산이 너무 작을 경우 잘못된 결과가 나타나는 것을 확인했다.

이를 방지하기 위해서 특정 영역의 Blue Pixel 값들의 첨도를 계산하여 일정 첨도 이하가 나올 경우. 즉, 값이 너무 심하게 한 곳으로 몰려있는 것이 확인될 경우 보정이 수행되지 않도록 했다. 이렇게 할 경우 잡티를 잡아내지 못하게 되는데, 이는 AC 과정에서 어느 정도 대처가 될 수 있지만 완벽한 알고리즘 설계를 위해 Blue Pixel이 아닌 밝기를 반영하여 보정하는 방식으로 수정이 필요할 것으로 생각된다.

위의 작업이 끝날 경우, 최종적으로 AC와 조도 보정 알고리즘을 적용하여 모델 학습을 위한 데이터를 생성하도록 한다.

<br>

- **fringing_correction.py의 main part**

```python
# --------------------------------------------------------------------------- #

"""
    2024.05.15, jdk
    이미지 세팅 파일 분리를 통해 모듈화로 세팅값 가져오기
"""

image_settings_file_path = "./config/image_settings.json"
with open(image_settings_file_path, 'r', encoding='utf-8') as file:
    image_setting = json.load(file)

area_size = getImageArraySize(image_setting)

# Fringing Row Correction을 수행하기 위한 세팅값
fringing_row_margin = image_setting['fringing_row_margin']
fringing_col_margin = image_setting['fringing_col_margin']
non_fringing_row_margin = image_setting['non_fringing_row_margin']
non_fringing_col_margin = image_setting['non_fringing_col_margin']
upper_pruning_index_value = image_setting['upper_pruning_index_value']
lower_pruning_index_value = image_setting['lower_pruning_index_value']

# Fringing Column Correction을 수행하기 위한 세팅값
fringing_col_width = area_size[3][3] - area_size[3][2]
col_upper_pruning_index_value = image_setting['col_upper_pruning_index_value']
col_lower_pruning_index_value = image_setting['col_lower_pruning_index_value']

# Brightness Correction을 수행하기 위한 세팅값
brightness_margin = image_setting['brightness_margin']

# Random Shuffle Kernel Sliding을 수행하기 위한 세팅값
shuffle_margin = image_setting['shuffle_margin']
random_shuffle_kernel_size = image_setting['random_shuffle_kernel_size']
random_shuffle_stride_size = image_setting['random_shuffle_stride_size']

col_shuffle_margin = image_setting['col_shuffle_margin']
col_random_shuffle_kernel_size = image_setting['col_random_shuffle_kernel_size']
col_random_shuffle_stride_size = image_setting['col_random_shuffle_stride_size']

gaussian_smoothing_kernel_size = image_setting['gaussian_smoothing_kernel_size']
gaussian_smoothing_stride_size = image_setting['gaussian_smoothing_stride_size']

kurtosis_threshold = image_setting['kurtosis_threshold']
side_outer_area_kurtosis_threshold = image_setting['side_outer_area_kurtosis_threshold']

anomaly_threshold = image_setting['anomaly_threshold']

# 4개의 영역을 기반으로 Image를 분할하여 얻은 numpy array를 갖고 있는 list
column_fringing_area_array = getColumnFringingAreaArray(image, area_size, image_height)

image_correction_setting = ImageCorrectionSetting(
    brightness_margin= brightness_margin,
    fringing_row_margin=fringing_row_margin,
    fringing_col_margin=fringing_col_margin,
    non_fringing_row_margin=non_fringing_row_margin,
    non_fringing_col_margin=non_fringing_col_margin,
    upper_pruning_index_value=upper_pruning_index_value,
    lower_pruning_index_value=lower_pruning_index_value,
    col_upper_pruning_index_value=col_upper_pruning_index_value,
    col_lower_pruning_index_value=col_lower_pruning_index_value,
    shuffle_margin=shuffle_margin,
    random_shuffle_kernel_size=random_shuffle_kernel_size,
    random_shuffle_stride_size=random_shuffle_stride_size,
    col_shuffle_margin=col_shuffle_margin,
    col_random_shuffle_kernel_size=col_random_shuffle_kernel_size,
    col_random_shuffle_stride_size=col_random_shuffle_stride_size,
    gaussian_smoothing_kernel_size=gaussian_smoothing_kernel_size,
    gaussian_smoothing_stride_size=gaussian_smoothing_stride_size,
    kurtosis_threshold=kurtosis_threshold,
    side_outer_area_kurtosis_threshold=side_outer_area_kurtosis_threshold,
    anomaly_threshold=anomaly_threshold
)

# --------------------------------------------------------------------------- #

# fringing columns correction 실시
print("\\n\\nCorrecting column fringing area")
column_fringing_area_array = correctFringingColumnsOfArea(column_fringing_area_array, image_correction_setting)
image = changeCorrectedFringingAreaOfOriginalImage(image, area_size, image_height, column_fringing_area_array)

side_outer_area_array_list = getSideOuterAreaArrayList(image, area_size)

# side outer area correction 실시
print("\\n\\nCorrecting side outer area_1")
side_outer_area_array_list[0] = correctSideOuterArea(side_outer_area_array_list[0], image_correction_setting)

print("\\n\\nCorrecting side outer area_2")
side_outer_area_array_list[1] = flipArray(side_outer_area_array_list[1])
side_outer_area_array_list[1] = flipArray(correctSideOuterArea(side_outer_area_array_list[1], image_correction_setting))

image = makeNewImageFromSideOuterArrays(image, side_outer_area_array_list, area_size)
area_array_list = getAreaArrayList(image, area_size)

# fringing rows correction 실시
print("\\n\\nCorrecting row fringing area_1")
area_array_list[0] = correctFringingRowsOfArea(area_array_list[0], image_correction_setting)

print("\\n\\nCorrecting row fringing area_2")
area_array_list[1] = correctFringingRowsOfArea(area_array_list[1], image_correction_setting)

print("\\n\\nCorrecting row fringing area_3")
area_array_list[2] = correctFringingRowsOfArea(area_array_list[2], image_correction_setting)

# corrected_area_image를 하나로 합성
image = makeNewImageFromCorrectedArrays(image, area_array_list, area_size)
image.save(corrected_image_file_path)

###############################################################################
```

각 과정의 자세한 코드는 Github -> project repository에 업로드하였음.



---

# 5. 향후 계획

## 1) CI 팀

> 보정이 완료된 이후 학습을 진행해서 성능을 확인하고 튜닝한다.

- [ ]  모델 성능 확인 및 튜닝

## 2) AC 팀

> 밝기 가중 컨볼루션을 마무리하고 조도를 보정한다.

- [ ]  Fringing Correction 최종 검토
- [ ]  AC 최종 검토
- [ ]  조도 보정 알고리즘 설계 및 적용

## 3) 공통 계획

> 촬영한 TPG에 대해 전체 코드를 순차적으로 동작시켜 결과를 확인하고 색상을 추론해본다. 또한 모든 코드를 통합하여 실제로 스와치를 넣어 프로그램을 동작할 수 있는 알고리즘으로 개선한다.

- [ ]  코드 통합
- [ ]  TPG색상 추론 및 학습

<br>