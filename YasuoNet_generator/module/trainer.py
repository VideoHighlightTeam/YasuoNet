import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from datetime import datetime as dt
from pytz import timezone
from tqdm.auto import tqdm
from functools import cmp_to_key
import pandas as pd
from sklearn.metrics import confusion_matrix

from module.data_loader import DataLoader


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def calc_f1score(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-7)


class ModelMetric:
    def __init__(self, epoch, loss, accuracy, precision, recall, checkpoint=None):
        self.epoch = epoch
        self.loss = loss
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1score = calc_f1score(precision, recall)
        self.checkpoint = checkpoint

    @staticmethod
    def compare(a, b):
        """ f1score, accuracy, loss를 각각 1, 2, 3순위로 하여 비교.
        a의 metric이 b의 것보다 더 좋으면 1, 같으면 0, 나쁘면 -1을 반환 """
        if a.f1score > b.f1score:
            return 1
        elif a.f1score == b.f1score:
            if a.accuracy > b.accuracy:
                return 1
            elif a.accuracy == b.accuracy:
                if a.loss < b.loss:
                    return 1
                elif a.loss == b.loss:
                    return 0
        return -1


class Trainer:
    def __init__(self, model: Model, data_loader: DataLoader, ckpt_dir):
        self.model = model
        self.data_loader = data_loader
        self.ckpt_dir = ckpt_dir

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    def test(self, batch_size):
        self.model.reset_metrics()

        # progress bar와 metric 표시를 위한 tqdm 생성
        batch_generator = self.data_loader.iter_test_batch_data(batch_size)
        batch_count = self.data_loader.get_test_batch_count(batch_size)
        description = f'Test'
        with tqdm(batch_generator, total=batch_count, desc=description) as pbar:
            # batch 수만큼 반복
            for x, y in pbar:
                loss, accuracy, precision, recall = self.model.test_on_batch(x, y, reset_metrics=False)
                f1score = calc_f1score(precision, recall)

                # print metrics
                metric_str = f'loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1score: {f1score:.4f}'
                pbar.set_postfix_str(metric_str)

        return loss, accuracy, precision, recall, f1score

    def test_prediction(self, batch_size):
        y_pred_list = []

        batch_generator = self.data_loader.iter_test_batch_data(batch_size)
        batch_count = self.data_loader.get_test_batch_count(batch_size)
        description = f'Test'
        with tqdm(batch_generator, total=batch_count, desc=description) as pbar:
            # batch 수만큼 반복
            for x in pbar:
                pred = self.model.predict_on_batch(x)
                pred = pred if isinstance(pred, np.ndarray) else pred.numpy()
                y_pred = (pred > 0.5) * 1
                y_pred_list.append(y_pred.squeeze())
        return np.hstack(y_pred_list)

    def predict(self, batch_size):
        y_pred_list = []
        for x, _ in self.data_loader.iter_test_batch_data(batch_size):
            pred = self.model.predict_on_batch(x)
            pred = pred.numpy() if not isinstance(pred, np.ndarray) else pred
            y_pred = (pred > 0.5) * 1

            y_pred_list.append(y_pred.squeeze())

        return np.concatenate(y_pred_list)
