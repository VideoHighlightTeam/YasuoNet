import sys
import module.naverTV_download as dl
from module.data_converter import convert_data
from module.YasuoNet import YasuoNet

def generate(url):

    input_video_dir = "./video"
    segment_length = 3
    video_sample_rate = 2
    video_width, video_height = 64, 64
    audio_sample_rate = 22050
    apply_mfcc=True
    output_dataset_dir = './data'
    
    downloader = dl.Downloader("360")
    print("영상 다운로드를 시작합니다...")
    downloader.download(url, input_video_dir + "./raw")
    print("영상 다운로드  완료")
    
    print("데이터 생성 시작")
    convert_data(input_video_dir, segment_length, video_sample_rate, video_width, video_height, audio_sample_rate, apply_mfcc, output_dataset_dir)

    print("하이라이트 생성 시작")
    YasuoNet(output_dataset_dir, input_video_dir).generate()

def main():
    if len(sys.argv) < 2:
        print("비디오 url을 입력하십시오.")
        quit(1)

    url = sys.argv[1]

    generate(url)


if __name__ == '__main__':
    main()
