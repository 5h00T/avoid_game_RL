from abc import ABCMeta, abstractmethod


class Logger():
    """
    エピソード毎にステップ数を出力
    最後にエピソード数と平均ステップ数を出力
    """
    def __init__(self):
        self.total_steps = 0
        self.total_episode = 0
        self.steps = 0

    def episode_begin(self):
        self.steps = 0
        self.total_episode += 1

    def step(self):
        self.steps += 1
        self.total_steps += 1

    def episode_end(self):
        print("episode:{} step:{}".format(self.total_episode, self.steps))

    def play_end(self):
        mean_step = self.total_steps / self.total_episode
        print("total_step:{} mean step:{}".format(self.total_steps, mean_step))


class Player(metaclass=ABCMeta):
    """
    抽象基底クラス
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, obs):
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


class LoggingPlayer(Player):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger()

    def get_action(self, obs):
        self.logger.step()

    def episode_begin(self, obs):
        self.logger.episode_begin()

    def episode_end(self):
        self.logger.episode_end()

    def play_end(self):
        self.logger.play_end()
