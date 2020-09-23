# YasuoNet - LCK 하이라이트 추출 AI

“YasuoNet”은 주어진 게임 대회 영상의 음성과 영상을 통해 맥락을 파악하여,
 기존의 영상에서 하이라이트 부분을 예측하는  Video Summarization 모델을 구현한 프로젝트입니다.
해당 프로젝트를 통해 편집자에겐 하이라이트 편집의 편의성을 증진 시킬 수 있고,
편집자가 따로 없는 중소규모의 대회에서도 하이라이트를 손쉽게 제작할 수 접근성이 늘어날 것입니다.

영상 분석은 여러 프레임의 Feature를 한번에 추출하기 위해 C3D를 활용하였고,
음성 분석은 MFCC 형태로 CNN을 통해 진행하였습니다. 여기에 LSTM을 추가하여 영상의 맥락을 좀더 장기적으로 파악할 수 있게 하였습니다.

다음과 같은 단계가 다루어집니다:
1. Dataset 구성을 위해 영상을 다운로드 합니다.
2. 모델 훈련에 사용되도록 영상을 labling합니다. 혹은 제공되는 lable을 사용합니다.
3. 모델을 훈련하고 저장합니다.
4. test를 위해 h5파일을 test폴더에 저장합니다.
5. 하이라이트 결과 영상을 제작합니다.


## 순서

1. 데이터 생성에 필요한 몇 몇의 영상을 다운로드 받습니다.
2. 영상과 영상을 통해 만들어진 레이블들을 모델에 입력하여 훈련합니다.
3. 하이라이트 추출할 영상을 다운로드 받습니다.
4. 영상은 이전에 훈련되었던 모델을 통해 하이라이트가 뽑힌 영상으로 변환됩니다.

## 동영상
모델을 통해 추출된 하이라이트 영상
[![](doc/images/example_capture.jpg)](https://drive.google.com/file/d/1FRyQU5uxCQ2iS5__s2Q4hDWWIOekgziz/view)


## 단계

이 과정을 구성하고 실행하기 위해 아래 단계를 따르십시오. 이 단계들은 아래쪽에 상세하게 기술되어 있습니다.

1. [사전 준비 사항 설치하기](#1-사전-준비-사항-설치하기)
2. [영상 자료 생성하기](#2-영상-자료-생성하기)
3. [영상 label 생성하기](#3-영상-label-생성하기)
4. [모델 훈련시키기](#4-모델-훈련시키기)
5. [모델 사용해보기](#5-모델-사용해보기)


### 1. 사전 준비 사항 설치하기

여러분의 시스템에 이 과정을 위한 python 요구사항이 설치되어 있어 합니다. 저장소의 최상위 디렉토리에서 다음을 실행하십시오:

```
pip install -r requirements.txt
```

**참고:** Windows 사용자인 경우, _scipy_ 패키지가 **pip** 를 통해 설치되지 않습니다. _scipy_ 사용을 위해
[scientific Python distribution](https://www.scipy.org/install.html#scientific-python-distributions) 설치를 추천합니다.
좀 더 유명한 것 중 하나는 [Anaconda](https://www.anaconda.com/download/)입니다.
그러나, Windows에서는 [여기](http://www.lfd.uci.edu/%7Egohlke/pythonlibs/#scipy)에 있는 인스톨러를 사용하여 수동으로 _scipy_ 패키지를 설치할 수 있습니다.

### 2. 영상 자료 생성하기



이 저장소의 tools 디렉토리에는 [naverTV_download.py](./tools/naverTV_download.py)가 있습니다.
이 스크립트는 주어진 텍스트 파일에 있는 주소의 영상들을 모두 다운로드 받습니다.
텍스트 파일의 형식은 다음과 같습니다.
#### txt file format
url1,save_file_name1
url2,save_file_name2
...

예시
```
https://tv.naver.com/v/12204239,T1 vs DWG 1세트
https://tv.naver.com/v/12204239,T1 vs DWG 2세트
```
저희가 사용한 비디오에 대한 text파일은 data 디렉토리에 있습니다.
다른 데이터셋을 만들고 싶다면, 위와 같은 형식으로 영상을 다운로드 받으면 되겠습니다.
txt파일을 생성했다면, [naverTV_download.py](./tools/naverTV_download.py)로 영상 다운로드를 진행할 수 있습니다.
```
$ python naverTV_download.py file_name.txt resolution
```
- file_name.txt : url, 저장할 이름이 저장된 txt파일
- resolution : 저장할 해상도(144, 360, 480, 720, 1080)

얼마나 많은 영상을 다운로드 하는지에 따라 이 스크립트를 완료하기까지 걸리는 시간이 달라집니다.
#### Output
Output File: save_file_name1.mp4, save_file_name2.mp4, ...
이 스크립트가 완료되면, data 디렉토리에 텍스트에 쓴 파일 이름으로 영상이 모두 저장됩니다.

----------------
추가예정
### 3. 영상 label 생성하기

### 4. 모델 훈련시키기

### 5. 모델 사용해보기

## 링크


## 라이센스


