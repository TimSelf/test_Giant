import os
import subprocess
import numpy as np
import matplotlib.image as mpimg
from datetime import datetime
import time

class Dataset(object):
    def __init__(self, data_root, t_w=800, t_h=600, timed=False):
        self.file_depth = open(os.path.join(data_root, 'depth', 'per_frame_timestamps.txt'))
        self.file_touch = open(os.path.join(data_root, 'touch', 'per_observation_timestamps.txt'))
        self.time_depth_pre = float(next(self.file_depth).strip())
        self.time_depth_cur = float(next(self.file_depth).strip())
        self.depth_processed = 0
        self.returned = 0
        self.t_w, self.t_h = t_w, t_h
        self.timed = timed
        if timed:
            self.starttime = None
            self.delayed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.timed:
            if not self.starttime:
                self.starttime = datetime.now()

        # get current timestamp
        cur_time = next(self.file_touch)
        cur_time = float(cur_time.strip())

        # get current observation
        with open(os.path.join(data_root, 'touch', f'observation-{str(int(self.returned)).zfill(6)}.txt')) as f:
            obs = f.read()  # check!
        obs = list(map(float, obs.strip().split(' ')))

        # get current frame from .mp4
        input_file = os.path.join(data_root, 'rgb', 'video.mp4')
        command = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(cur_time / 1000),
                   '-i', input_file,
                   '-vf', 'scale=%d:%d' % (self.t_w, self.t_h),
                   '-vframes', '1',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo', '-']
        rgb_pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = rgb_pipe.communicate()
        if err:
            print('error', err)
            self.returned += 1
            return None
        rgb_image = np.frombuffer(out, dtype='uint8').reshape((self.t_h, self.t_w, 3))

        # get current depth image
        while self.time_depth_cur <= cur_time:
            self.time_depth_pre = self.time_depth_cur
            try:
                self.time_depth_cur = float(next(self.file_depth).strip())
                self.depth_processed += 1
            except StopIteration:
                break
        if cur_time - self.time_depth_pre < self.time_depth_cur - cur_time:  # select closest
            depth_num = self.depth_processed
            depth_time = self.time_depth_pre
        else:
            depth_num = self.depth_processed + 1
            depth_time = self.time_depth_cur
        depth_image = mpimg.imread(os.path.join(data_root, 'depth', f'frame-{str(int(depth_num)).zfill(6)}.png'))

        if self.timed:
            delay_time = max(cur_time, depth_time)
            delay_time -= self.delayed
            self.delayed += delay_time
            time.sleep(delay_time/1000)

        # increment and return
        self.returned += 1
        return cur_time, obs, rgb_image, depth_image
