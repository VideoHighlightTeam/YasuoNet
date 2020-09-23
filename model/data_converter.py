import os
import sys
import glob
import numpy as np
import pandas as pd
import math
from datetime import datetime as dt
import moviepy.editor as moviepy
import librosa
from pprint import pprint as pp
import multiprocessing as mp
from itertools import repeat
import json

import util.collection_util as cu


DEBUG_PRINT = False


def to_hms(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def iter_segment_raw_data(config, title, video_path):
    segment_length = config['segment_length']
    video_sample_rate = config['video_sample_rate']
    video_width = config['video_width']
    video_height = config['video_height']
    audio_sample_rate = config['audio_sample_rate']
    apply_mfcc = config['apply_mfcc']

    clip = moviepy.VideoFileClip(video_path)
    # video/audio clip에 sample rate 변경, resize 적용
    video_clip = clip.resize(newsize=(video_width, video_height)).set_fps(video_sample_rate)
    audio_clip = clip.audio.set_fps(audio_sample_rate).audio_normalize()

    print(f'{title} :: video_frame_rate: {clip.fps:.2f}, duration: {clip.duration:.2f}, size: {clip.size}, audio_frame_rate: {clip.audio.fps}')

    # 모든 segment들의 경계선
    segment_bound_list = np.arange(0, math.ceil(video_clip.duration / segment_length) + 1) * segment_length

    # 각 segment 구간별로 subclip 생성하여 video/audio 데이터 수집
    for segment_start_sec, segment_end_sec in zip(segment_bound_list[:-1], segment_bound_list[1:]):
        # print(segment_start_sec, segment_end_sec)

        subclip = video_clip.subclip(segment_start_sec, min(segment_end_sec, video_clip.duration))

        # video frame 추출
        video_frames = np.array(list(subclip.iter_frames())).astype(np.float16)

        # normalize to (0, 1)
        video_frames /= 255

        correct_video_frame_count = int(segment_length * video_sample_rate)
        # 마지막 segment일 때 frame count가 모자라면 zero padding 적용
        if segment_end_sec >= video_clip.duration and video_frames.shape[0] < correct_video_frame_count:
            pad_size = correct_video_frame_count - video_frames.shape[0]
            video_frames = np.pad(video_frames, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

        # frame count가 항상 일정해야 함
        try:
            assert video_frames.shape[0] == correct_video_frame_count
        except AssertionError as e:
            print(f'ERROR: {title} :: video frame count is not correct. ({video_frames.shape[0]}, {correct_video_frame_count})')
            raise e

        video_data = video_frames
        # print(video_data.shape, video_data.dtype)

        # audio 추출
        subclip = audio_clip.subclip(segment_start_sec, min(segment_end_sec, video_clip.duration))
        audio_frames = subclip.to_soundarray().astype(np.float16)   # value range (-1, 1)

        correct_audio_frame_count = int(segment_length * audio_sample_rate)
        # 마지막 segment일 때 frame count가 모자라면 zero padding 적용
        if segment_end_sec >= audio_clip.duration and audio_frames.shape[0] < correct_audio_frame_count:
            pad_size = correct_audio_frame_count - audio_frames.shape[0]
            audio_frames = np.pad(audio_frames, ((0, pad_size), (0, 0)))

        # frame count가 항상 일정해야 함
        try:
            assert audio_frames.shape[0] == correct_audio_frame_count
        except AssertionError as e:
            print(f'ERROR: {title} :: audio frame count is not correct. ({audio_frames.shape[0]}, {correct_audio_frame_count})')
            raise e

        if audio_frames.ndim == 2 and audio_frames.shape[1] > 1:
            # merge stereo to mono
            audio_frames = audio_frames.mean(axis=1)

        if apply_mfcc:
            # audio waveform에 dfcc 적용하여 특징값 추출
            mfccs = librosa.feature.mfcc(y=audio_frames, sr=audio_sample_rate, n_mfcc=40).astype(np.float16)

            # normalize by standard normal distribution
            mfccs = (mfccs - mfccs.mean()) / (mfccs.std() + 1e-6)

            audio_data = mfccs
        else:
            audio_data = audio_frames

        # print(audio_data.shape, audio_data.dtype)

        yield video_data, audio_data, segment_start_sec, segment_end_sec, clip.duration, clip.fps


def get_segment_label(segment_start_frame, segment_end_frame, hl_section_df):
    # segment에 포함된 원래 frame 수
    segment_length = segment_end_frame - segment_start_frame

    # print(int(segment_start_frame), int(segment_end_frame))

    label = 0

    # 이 segment가 조금이라도 겹쳐있는 highlight 구간들
    intersection_frames = hl_section_df[(segment_start_frame < hl_section_df['hl_end_frame']) & (segment_end_frame > hl_section_df['hl_start_frame'])].to_numpy()

    if len(intersection_frames) > 0:
        # segment와 highlight 구간이 정확히 겹치는 시간을 계산
        if intersection_frames[0, 0] < segment_start_frame < intersection_frames[0, 1]:
            intersection_frames[0, 0] = segment_start_frame
        if intersection_frames[-1, 0] < segment_end_frame < intersection_frames[-1, 1]:
            intersection_frames[-1, 1] = segment_end_frame

        highlight_length = (intersection_frames[:, 1] - intersection_frames[:, 0]).sum()
        # 정확히 겹치는 시간이 segment 길이의 1/2 이상이면 label=1
        label = 1 if highlight_length / segment_length > 0.5 else 0

    return label


def generate_segment_data(config, title, video_path, hl_section_path, output_dataset_dir):
    # print(title, hl_section_path, video_path)

    # highlight 구간 정보 로드
    hl_section_df = pd.read_csv(hl_section_path, header=None, names=['hl_start_frame', 'hl_end_frame', 'hl_start_time', 'hl_end_time'])
    # print(hl_section_df)

    output_segment_dir = os.path.join(output_dataset_dir, title)
    if not os.path.exists(output_segment_dir):
        os.mkdir(output_segment_dir)

    start = dt.now()

    output_path_list = []
    for i, raw_data in enumerate(iter_segment_raw_data(config, title, video_path)):
        # segment별로 추출된 raw data
        video_data, audio_data, segment_start_sec, segment_end_sec, total_duration, original_frame_rate = raw_data

        segment_start_frame = segment_start_sec * original_frame_rate
        segment_end_frame = segment_end_sec * original_frame_rate

        # highlight 구간 정보를 기반으로 레이블(0, 1) 확정
        label = get_segment_label(segment_start_frame, segment_end_frame, hl_section_df)

        if DEBUG_PRINT and label == 1:
            print(f'{title} :: [{i}] {segment_start_frame:.2f} ({to_hms(segment_start_sec)}) - {segment_end_frame:.2f} ({to_hms(segment_end_sec)}) >>> label: {label}')

        segment_data = {
            'video': video_data,
            'audio': audio_data,
            'label': label,
            'start_sec': segment_start_sec,
            'end_sec': segment_end_sec,
            'total_duration': total_duration
        }

        # segment 데이터 저장
        output_path = os.path.join(output_segment_dir, f'seg_{i:05d}_{label}.pkl')
        cu.save(segment_data, output_path)

        output_path_list.append(output_path)

    end = dt.now()
    print(f'{title} :: {len(output_path_list)} segments saved, elapsed time: {end - start}s')

    return output_path_list


def convert_data(input_video_dir, segment_length, video_sample_rate, video_width, video_height, audio_sample_rate, apply_mfcc, output_dataset_dir=None):
    # output_dataset_dir이 주어지지 않은 경우 기본값으로 정의
    if not output_dataset_dir:
        output_dataset_dir = f'dataset_sl{segment_length}_vsr{video_sample_rate}_vw{video_width}_vh{video_height}_asr{audio_sample_rate}{"_mfcc" if apply_mfcc else ""}'

    config = {
        'segment_length': segment_length,
        'video_sample_rate': video_sample_rate,
        'video_width': video_width,
        'video_height': video_height,
        'audio_sample_rate': audio_sample_rate,
        'apply_mfcc': apply_mfcc
    }

    print(f'input_video_dir: {input_video_dir}')
    print(f'output_dataset_dir: {output_dataset_dir}')
    print('config:')
    pp(config)
    print()

    # output_dataset_dir이 이미 존재하는 경우 종료
    if os.path.exists(output_dataset_dir):
        print(f'WARNING: output_dataset_dir \"{output_dataset_dir}\" already exists.')
        if input('Remove all contents and continue? (y/n): ').lower().strip() == 'y':
            import shutil
            shutil.rmtree(output_dataset_dir)
        else:
            quit(1)

    os.makedirs(output_dataset_dir, exist_ok=True)

    # highlight 구간 파일 경로를 기준으로 영상 파일 경로와 제목을 생성
    hl_section_path_list = glob.glob(os.path.join(input_video_dir, '*.txt'))
    video_path_list = [os.path.splitext(path)[0] + '.mp4' for path in hl_section_path_list]
    title_list = [os.path.splitext(os.path.split(path)[1])[0] for path in hl_section_path_list]

    # video_path_list 중 존재하지 않는 파일이 있으면 종료
    video_path_exists = list(map(os.path.exists, video_path_list))
    if not all(video_path_exists):
        for video_path in [video_path for exists, video_path in zip(video_path_exists, video_path_list) if not exists]:
            print(f'ERROR: {video_path} not exists')
        quit(1)

    start = dt.now()

    # 각 원본영상과 highlight 구간 파일을 읽고 segment 단위로 나누어 저장
    with mp.Pool() as pool:
        params = zip(repeat(config), title_list, video_path_list, hl_section_path_list, repeat(output_dataset_dir))
        output_path_list_list = pool.starmap(generate_segment_data, params)
        total_segment_count = sum(map(len, output_path_list_list))

    # get video/audio data shape
    segment_data = cu.load(output_path_list_list[0][0])
    video_data_shape = segment_data['video'].shape
    audio_data_shape = segment_data['audio'].shape

    # metadata 기록
    metadata = {
        'created': str(dt.now()),
        'config': config,
        'data_shape': {'video': video_data_shape, 'audio': audio_data_shape},
        'total_segment_count': total_segment_count,
        'segment_counts': {title: count for title, count in zip(title_list, map(len, output_path_list_list))}
    }
    metadata_path = os.path.join(output_dataset_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    end = dt.now()

    print()
    print(f'Total {total_segment_count} segments saved, elapsed time: {end - start}s')


def print_usage():
    print(f'Usage: python {sys.argv[0]} input_video_dir segment_length(sec) video_sample_rate(fps) video_width(px) video_height(px) audio_sample_rate(hz) apply_mfcc(0 or 1) [output_dataset_dir]')
    print()
    print('Example:')
    print(f'python {sys.argv[0]} data/raw 5 3 64 64 11025 0')
    print(f'python {sys.argv[0]} data/raw 3 6 128 128 22050 1 data/dataset1')


def main():
    if len(sys.argv) < 8:
        print_usage()
        quit(1)

    input_video_dir = sys.argv[1]
    segment_length = float(sys.argv[2])
    video_sample_rate = float(sys.argv[3])
    video_width, video_height = map(int, sys.argv[4:6])
    audio_sample_rate = int(sys.argv[6])
    apply_mfcc = bool(sys.argv[7])
    output_dataset_dir = sys.argv[8] if len(sys.argv) > 8 else None

    convert_data(input_video_dir, segment_length, video_sample_rate, video_width, video_height, audio_sample_rate, apply_mfcc, output_dataset_dir)


if __name__ == '__main__':
    main()
