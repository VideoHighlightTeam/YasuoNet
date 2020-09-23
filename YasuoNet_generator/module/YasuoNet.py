import os
import math
import numpy as np
import pandas as pd
import shutil
import glob
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip, concatenate_videoclips
from module.data_loader import DataLoader
from module.trainer import Trainer
from module.data_converter import to_hms

class YasuoNet:
    def __init__(self, dataset_dir, video_dir):
        self.dataset_dir = dataset_dir
        self.video_dir = video_dir
        self.ckpt_dir = 'ckpt'
        self.batch_size = 1

    def load_data(self):
        data_loader = DataLoader(self.dataset_dir, x_includes=['video', 'audio'], x_expand=2)
        return data_loader

    def load_model(self):
        checkpoint_name = 'ckpt-20200906-011837-0006-0.7346_model'
        model_path = os.path.join(self.ckpt_dir, checkpoint_name + '.h5')
        model_restored = load_model(model_path)
        return model_restored

    def predict(self, model_restored, data_loader):
        trainer = Trainer(model_restored, data_loader, self.ckpt_dir)
        y_pred = trainer.test_prediction(self.batch_size)
        return y_pred

    def make_highlight(self, data_loader, y_pred):
        segment_length = data_loader.get_metadata()['config']['segment_length']
        segment_df = data_loader.test_segment_df.copy()
        segment_df['pred'] = y_pred
        segment_df['start_sec'] = (segment_df['index'] * segment_length)
        segment_df['end_sec'] = ((segment_df['index'] + 1) * segment_length)
        start = np.array(segment_df['start_sec'][segment_df['pred'] == 1])
        end = np.array(segment_df['end_sec'][segment_df['pred'] == 1])
        name = "raw"
        
        i=1
        while(i<len(end)):
            if end[i] - 3 == end[i-1]:
                end[i-1] = end[i]
                start = np.delete(start, i)
                end = np.delete(end, i)
            else:
                i+=1

        clip = VideoFileClip(os.path.join(self.video_dir,name+".mp4"))
        subclips = []
        for i in range(len(start)):
            start_lim = start[i]
            end_lim = end[i]
            subclips.append(clip.subclip(start_lim, end_lim))
        final_clip=concatenate_videoclips(subclips)
        final_clip.write_videofile("./Highlights" + name + ".mp4") #Enter the desired output highlights filename.
        for i in subclips:
            i.close()
        clip.close()

    def generate(self):
        data_loader = self.load_data()
        model_restored = self.load_model()
        y_pred = self.predict(model_restored, data_loader)
        self.make_highlight(data_loader, y_pred)


