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

from data_loader import DataLoader


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

    def train(self, optimizer, epochs, batch_size, class_weights=None):
        train_start_time = dt.now(tz=timezone('Asia/Seoul')).strftime('%Y%m%d-%H%M%S')

        print('Training started at', train_start_time)
        print('optimizer:', optimizer.get_config())
        print('epochs:', epochs)
        print('batch size:', batch_size)
        print('class weights:', class_weights)

        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

        if class_weights:
            # 클래스별 데이터 수를 고려하여 weight의 가중 평균이 1이 되도록 조정
            class_counts = self.data_loader.all_segment_df['label'].value_counts(sort=False)
            class_weights = np.array(class_weights) * class_counts.sum() / (class_weights * class_counts).sum()
            class_weight_dict = {c: w for c, w in enumerate(class_weights)}
            print('normalized class weights:', class_weights)
        else:
            class_weight_dict = None

        print()

        top5_model_metric_list = []

        try:
            # epoch 수만큼 반복
            for step in range(1, epochs + 1):
                # 학습 실행
                self.model.reset_metrics()

                # progress bar와 metric 표시를 위한 tqdm 생성
                batch_generator = self.data_loader.iter_train_batch_data(batch_size)
                batch_count = self.data_loader.get_train_batch_count(batch_size)
                description = f'Train {step}/{epochs}'
                with tqdm(batch_generator, total=batch_count, desc=description) as pbar:
                    # batch 수만큼 반복
                    for x, y in pbar:
                        loss, accuracy, precision, recall = self.model.train_on_batch(x, y, class_weight=class_weight_dict, reset_metrics=False)
                        f1score = calc_f1score(precision, recall)

                        metric_str = f'loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1score: {f1score:.4f}'
                        pbar.set_postfix_str(metric_str)

                # 검증 실행
                self.model.reset_metrics()

                # progress bar와 metric 표시를 위한 tqdm 생성
                batch_generator = self.data_loader.iter_valid_batch_data(batch_size)
                batch_count = self.data_loader.get_valid_batch_count(batch_size)
                description = f'Validation {step}/{epochs}'
                with tqdm(batch_generator, total=batch_count, desc=description) as pbar:
                    # batch 수만큼 반복
                    for x, y in pbar:
                        loss, accuracy, precision, recall = self.model.test_on_batch(x, y, reset_metrics=False)
                        model_metric = ModelMetric(step, loss, accuracy, precision, recall)
                        f1score = model_metric.f1score
                        is_top5 = (len(top5_model_metric_list) < 5) or (ModelMetric.compare(model_metric, top5_model_metric_list[-1]) > 0)

                        metric_str = f'loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1score: {f1score:.4f}'
                        metric_str += ' #' if is_top5 else ''
                        pbar.set_postfix_str(metric_str)

                # 성능 향상된 모델 저장
                if is_top5:
                    checkpoint_name = f'ckpt-{train_start_time}-{step:04d}-{model_metric.f1score:.4f}'
                    checkpoint_path = os.path.join(self.ckpt_dir, checkpoint_name + '.h5')
                    self.model.save_weights(checkpoint_path)
                    print(f'model saved to {checkpoint_path}')

                    # top5에 추가
                    model_metric.checkpoint = checkpoint_name
                    top5_model_metric_list.append(model_metric)
                    top5_model_metric_list = sorted(top5_model_metric_list, key=cmp_to_key(ModelMetric.compare), reverse=True)[:5]

            print('Train finished')
        except KeyboardInterrupt:
            print('Train stopped')

        print()
        print('Top5 models')
        top5_df = pd.DataFrame([m.__dict__ for m in top5_model_metric_list])
        top5_df.set_index('epoch', inplace=True)
        print(top5_df)

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
        y_true_list = []
        y_pred_list = []

        batch_generator = self.data_loader.iter_test_batch_data(batch_size)
        batch_count = self.data_loader.get_test_batch_count(batch_size)
        description = f'Test'
        conf_mat = np.zeros((2, 2), dtype=np.int32)
        with tqdm(batch_generator, total=batch_count, desc=description) as pbar:
            # batch 수만큼 반복
            for x, y_true in pbar:
                pred = self.model.predict_on_batch(x)
                pred = pred if isinstance(pred, np.ndarray) else pred.numpy()
                y_pred = (pred > 0.5) * 1

                y_true_list.append(y_true.squeeze())
                y_pred_list.append(y_pred.squeeze())

                # print metrics
                conf_mat += confusion_matrix(y_true, y_pred, labels=range(DataLoader.CLASS_COUNT))

                accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / (conf_mat.sum() + 1e-7)
                precision = conf_mat[1, 1] / (conf_mat[:, 1].sum() + 1e-7)
                recall = conf_mat[1, 1] / (conf_mat[1, :].sum() + 1e-7)
                f1score = calc_f1score(precision, recall)

                metric_str = f'accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1score: {f1score:.4f}'
                pbar.set_postfix_str(metric_str)

        return np.concatenate(y_true_list), np.concatenate(y_pred_list)

    def predict(self, batch_size):
        y_pred_list = []
        for x, _ in self.data_loader.iter_test_batch_data(batch_size):
            pred = self.model.predict_on_batch(x)
            pred = pred.numpy() if not isinstance(pred, np.ndarray) else pred
            y_pred = (pred > 0.5) * 1

            y_pred_list.append(y_pred.squeeze())

        return np.concatenate(y_pred_list)
