import math
import os
import glob
import numpy as np
import pandas as pd
import json
from functools import reduce
from operator import itemgetter
import itertools

import util.collection_util as cu


class DataLoader:
    """ Data Converter가 생성한 segment 파일(.pkl)을 학습/검증/테스트 절차에 맞게 배치 단위로 공급하는 기능 구현 """

    CLASS_COUNT = 2

    def __init__(self, dataset_dir, x_includes, x_expand=0):
        self.dataset_dir = dataset_dir
        self.x_includes = x_includes
        self.x_expand = x_expand
        self.metadata = None

        self._load_metadata()

        all_segment_list = self._get_all_segment_list()
        self.all_segment_df = pd.DataFrame(all_segment_list)
        # segment sequence 데이터 생성 시 성능 향상을 위해 dictionary 구조 병행 사용
        self.all_segment_dict = cu.list_to_dict(all_segment_list, itemgetter('title'))

        self.test_segment_df = self.all_segment_df

    def _load_metadata(self):
        metadata_path = os.path.join(self.dataset_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def get_metadata(self):
        return self.metadata

    def _get_all_segment_list(self):
        all_segment_path_list = sorted(glob.glob(self.dataset_dir + '/*/*.pkl'))

        # segment 정보를 1차원 리스트로 나열
        all_segment_list = []
        for path in all_segment_path_list:
            # title: 원본영상 이름, name: 파일 이름
            title, name = os.path.normpath(path).split(os.sep)[-2:]
            name = os.path.splitext(name)[0]
            # 원본영상 내의 segment index
            index = int(name.split('_')[1])

            all_segment_list.append({'title': title, 'name': name, 'index': index, 'path': path})

        return all_segment_list

    def get_test_data_count(self):
        return len(self.test_segment_df)

    def get_all_data_count(self):
        return len(self.all_segment_df)

    def get_test_batch_count(self, batch_size):
        return math.ceil(self.get_test_data_count() / batch_size)   # ceiling

    def iter_test_batch_data(self, batch_size, repeat=False):
        """ 테스트 데이터에서 batch_size만큼씩 순차적으로 반환 """
        for batch_data in self._iter_subset_batch_data(self.test_segment_df, batch_size, repeat, False):
            yield batch_data

    def iter_all_batch_data(self, batch_size, repeat=False):
        """ 전체 데이터에서 batch_size만큼씩 순차적으로 반환 """
        for batch_data in self._iter_subset_batch_data(self.all_segment_df, batch_size, repeat, False):
            yield batch_data

    def _iter_subset_batch_data(self, subset_df, batch_size, repeat, shuffle):
        # 전체 데이터를 순회할 때까지 반복
        while True:
            # 주어진 데이터에서 현재 iterator의 위치
            i = 0

            if shuffle:
                subset_df = subset_df.sample(frac=1)

            while True:
                # 배치 데이터 slicing
                batch_df = subset_df.iloc[i: i + batch_size]

                # 데이터가 없으면 iteration 종료
                if len(batch_df) == 0:
                    break

                # 모든 배치 데이터에 대해 segment 데이터 읽어와서 리스트 생성
                batch_data = []
                for _, segment in batch_df.iterrows():
                    target_segment_data = cu.load(segment['path'])

                    # 시간값 추가
                    target_segment_data['time'] = target_segment_data['start_sec'] / target_segment_data['total_duration']

                    if self.x_expand > 0:
                        # 앞/뒤 segment 결합
                        segment_data_list = []
                        title = segment['title']
                        target_index = segment['index']
                        title_segment_list = self.all_segment_dict[title]
                        segment_data_zero = {'video': np.zeros_like(target_segment_data['video']), 'audio': np.zeros_like(target_segment_data['audio'])}

                        for index in range(target_index - self.x_expand, target_index + self.x_expand + 1):
                            if index == target_index:
                                segment_data_list.append(target_segment_data)
                            elif 0 <= index < len(title_segment_list):
                                segment_data_list.append(cu.load(title_segment_list[index]['path']))
                            else:
                                segment_data_list.append(segment_data_zero)

                        video_list, audio_list = zip(*[(s['video'], s['audio']) for s in segment_data_list])

                        # 원본 데이터 교체
                        target_segment_data['video'] = np.array(video_list)
                        target_segment_data['audio'] = np.array(audio_list)

                    batch_data.append(target_segment_data)

                # x, y 데이터 분리
                batch_x_video, batch_x_audio, batch_x_time, = zip(*[(s['video'], s['audio'], s['time']) for s in batch_data])

                batch_x_video = np.array(batch_x_video, dtype=np.float16)
                batch_x_audio = np.array(batch_x_audio, dtype=np.float16)
                batch_x_time = np.array(batch_x_time, dtype=np.float16)

                # 데이터를 iterator로 반환
                batch_x = []
                if 'video' in self.x_includes:
                    batch_x.append(batch_x_video)
                if 'audio' in self.x_includes:
                    batch_x.append(batch_x_audio)
                if 'time' in self.x_includes:
                    batch_x.append(batch_x_time)

                yield batch_x

                i += batch_size

            if not repeat:
                break
