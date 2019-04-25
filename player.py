from abc import ABCMeta, abstractmethod

import numpy as np
from PIL import Image
import copy


class Extension(metaclass=ABCMeta):
    """
    抽象基底クラス
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self, obs):
        pass

    @abstractmethod
    def episode_begin(self, obs):
        pass

    @abstractmethod
    def episode_end(self):
        pass

    @abstractmethod
    def play_end(self):
        pass


class Logger(Extension):
    """
    エピソード毎にステップ数を出力
    最後にエピソード数と平均ステップ数を出力
    """

    def __init__(self):
        super().__init__()
        self.total_steps = 0
        self.total_episode = 0
        self.steps = 0

    def episode_begin(self, obs):
        self.steps = 0
        self.total_episode += 1

    def step(self, obs):
        self.steps += 1
        self.total_steps += 1

    def episode_end(self):
        print("episode:{} step:{}".format(self.total_episode, self.steps))

    def play_end(self):
        mean_step = self.total_steps / self.total_episode
        print("total_step:{} mean step:{}".format(self.total_steps, mean_step))


class GIFSaver(Extension):
    """
    エピソード毎にGIFファイルを保存する
    """

    def __init__(self):
        super().__init__()
        self.frames = None

    def episode_begin(self, obs):
        self.frames = []

    def step(self, obs):
        img = Image.fromarray(copy.deepcopy(np.transpose(obs, (1, 0, 2))), mode="RGB")
        self.frames.append(img)

    def episode_end(self):
        self.frames[0].save("result.gif", save_all=True, append_images=self.frames[1:], optimize=False, duration=20,
                            loop=0)

    def play_end(self):
        pass


class Player():
    def __init__(self, extensions=None):
        self.extensions = [ext for ext in extensions]

    def step(self, obs):
        for ext in self.extensions:
            ext.step(obs)

    def episode_begin(self, obs):
        for ext in self.extensions:
            ext.episode_begin(obs)

    def episode_end(self):
        for ext in self.extensions:
            ext.episode_end()

    def play_end(self):
        for ext in self.extensions:
            ext.play_end()
