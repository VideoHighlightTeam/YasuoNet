# VideoSummarizer
하이라이트 추출 모델을 학습/검증/테스트하는 기능을 구현

## 0. Common
### Requirements
```
# pip install -r requirements.txt
```
or
```
# conda install --file requirements.txt
```
moviepy는 반드시 v1.0.3 버전을 설치해야 한다. Anaconda 환경에서 v1.0.3 버전이 설치되지 않을 경우에는 아래와 같이 소스를 직접 빌드하여 설치한다.
```
# git clone https://github.com/Zulko/moviepy.git
# cd moviepy
# git checkout v1.0.3
# python setup.py install
```

## 1. Data Converter
게임 원본 영상을 일정 길이의 작은 segment로 자르고 그 안에서 지정된 sample rate로 영상 프레임과 음성 프레임을 추출한다.<br>
특히 영상은 지정된 width와 height로 resize를 함께 수행하며 추출된 segment 데이터는 pickle 파일(.pkl)로 저장한다.

### Usage
```
# python data_converter.py input_video_dir segment_length(sec) video_sample_rate(fps) video_width(px) video_height(px) audio_sample_rate(hz) apply_mfcc(0 or 1) [output_dataset_dir]
```
Examples:
```
# python data_converter.py data/raw 5 3 64 64 11025 0
```
data/raw 디렉터리의 mp4, txt 파일로부터 segment 파일(.pkl)을 생성한다. 저장 디렉터리는 현재 디렉터리 하위에 생성된다.<br>
이 때, segment 길이는 5초, 비디오 프레임에서 초당 3프레임을 추출하며 해상도는 64x64픽셀이 된다. 오디오는 11025Hz로 변환하여 추출하고 MFCC 전처리는 적용하지 않는다.
```
# python data_converter.py data/raw 3 6 128 128 22050 1 data/dataset1
```
data/raw 디렉터리의 mp4, txt 파일로부터 segment 파일(.pkl)을 생성한다. 변환된 파일은 data/dataset1에 저장된다.<br>
이 때, segment 길이는 3초, 비디오 프레임에서 초당 6프레임을 추출하며 해상도는 128x128픽셀이 된다. 오디오는 22050Hz로 변환하여 추출하고 MFCC 전처리를 적용한다.

## 2. Data Loader
Data Converter가 생성한 pkl 파일들을 학습/검증/테스트셋으로 나누어 모델에 공급하는 역할을 한다.<br>
학습/검증/테스트 서브셋으로 나누는 기준은 원본 영상 단위로 나누고 있으며, 향후에도 일관성과 확장성을 유지할 수 있도록 hash 함수를 사용하여 같은 영상은 항상 같은 서브셋에 속하도록 구현했다.<br>
이 방식의 단점은 지정한 비율대로 정확히 데이터셋이 나눠지지 않을 수 있다는 것이다. 데이터 규모가 많이 크다면 지정한 비율에 수렴할 수 있겠지만 그렇지 않을 때는 오차가 발생한다.

## 3. Model Trainer
딥러닝 모델을 정의하고 Data Loader로부터 공급받는 데이터를 이용하여 모델을 학습 및 평가한다.<br>
학습된 모델은 지정된 checkpoint 경로에 .hdf5 파일로 저장되는데 모든 모델이 저장되는 것은 아니고 검증 데이터에 대한 loss가 이전 모델보다 낮아졌을 때만 저장한다.<br>
Google Colab과 Jupyter Notebook, Shell 모두에서 실행 가능하다.

### Usage
#### * Google Colab
구글드라이브 "YasuoNet" 공유폴더에 **colab_run_trainer.ipynb** 노트북 파일이 있으며 이것을 자신의 드라이브에 사본으로 저장한 뒤 실행한다.<br>
첫번째 셀부터 차례대로 스크립트를 실행하면 데이터와 소스코드를 서버에 복사하고 모델 생성 및 학습을 진행할 수 있다.

#### * Jupyter Notebook
우선 구글드라이브에서 사용할 데이터셋 파일을 로컬에 복사하고 압축을 해제한다. (예: dataset14_sl5_vsr3_vw64_vh64_asr11025.zip)
소스코드 중 **run_trainer.ipynb** 노트북 파일이 있으며 이것을 실행하고 데이터셋의 압축을 푼 경로를 dataset_dir에 지정해주면 모델 생성 및 학습을 진행할 수 있다.

#### * Shell
```
# python main.py [args]
  Arguments:
    --learning_rate: 학습률
    --epochs: 학습을 실행할 epoch 수
    --batch_size: 배치 크기
    --data_dir: 데이터셋 경로
    --ckpt_dir: 학습된 모델을 checkpoint 형태로 저장할 경로
```
