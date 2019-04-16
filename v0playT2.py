import copy
import random
import argparse

import pygame
from pygame.locals import *
from play_env import v0PlayT2Env
from player import LoggingPlayer, Logger

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import sys
from PIL import Image
from scipy import misc


class Human(LoggingPlayer):
    """
    人間が操作
    """

    def __init__(self, logger):
        super().__init__(logger)

    def get_action(self, obs):
        """
        キー入力によってactionの値を決定
        :param obs: 状態(RGB値)
        :return: キー入力によって決定されたアクション
        """
        super().get_action(obs)
        pressed_key = pygame.key.get_pressed()
        action = 0

        # 2ボタン同時押しを先に判定
        if pressed_key[K_UP] and pressed_key[K_RIGHT]:
            action = 2
        elif pressed_key[K_DOWN] and pressed_key[K_LEFT]:
            action = 6
        elif pressed_key[K_DOWN] and pressed_key[K_RIGHT]:
            action = 4
        elif pressed_key[K_LEFT] and pressed_key[K_UP]:
            action = 8
        elif pressed_key[K_UP]:
            action = 1
        elif pressed_key[K_RIGHT]:
            action = 3
        elif pressed_key[K_DOWN]:
            action = 5
        elif pressed_key[K_LEFT]:
            action = 7

        return action

    def episode_begin(self, obs):
        super().episode_begin(obs)

    def episode_end(self):
        super().episode_end()


def make_agent():
    """
    AIクラスに渡すagentを作成する
    :return: agent
    """

    class QFunction(chainer.Chain):

        def __init__(self, n_actions, n_hidden_channels=500):
            super().__init__()
            with self.init_scope():
                self.l0 = L.Convolution2D(4, 32, ksize=8, stride=4, initial_bias=0.1)
                self.l1 = L.Convolution2D(32, 64, ksize=4, stride=2, initial_bias=0.1)
                self.l2 = L.Convolution2D(64, 64, ksize=2, stride=1, initial_bias=0.1)
                self.l3 = L.Linear(3136, n_hidden_channels, initial_bias=0.1)
                self.l4 = L.Linear(n_hidden_channels, n_actions)

        def __call__(self, x, test=False):
            h1 = F.relu(self.l0(x))
            h2 = F.relu(self.l1(h1))
            h3 = F.relu(self.l2(h2))
            h4 = F.relu(self.l3(h3))
            return chainerrl.action_value.DiscreteActionValue(self.l4(h4))

    n_actions = 9
    q_func = QFunction(n_actions)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(q_func)

    gamma = 0.98

    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.1, random_action_func=random.randint(0, 9))

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=5 * (10 ** 4))

    def dqn_phi(screens):
        assert len(screens) == 4
        raw_values = np.asarray(screens, dtype=np.float32)
        # [0,255] -> [0, 1]
        raw_values = copy.deepcopy(raw_values) / 255.0
        return raw_values

    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=50000, update_interval=5,
        target_update_interval=10000, phi=dqn_phi)

    agent.load("models/v0T1model")

    return agent


def preprocess(observation):
    return misc.imresize(np.asarray(Image.fromarray(observation).convert("L")), (80, 80))


class AI(LoggingPlayer):
    def __init__(self, agent, preprocess, logger):
        super().__init__(logger)
        self.agent = agent
        self.preprocess = preprocess
        self.obs4step = np.zeros((4, 80, 80))

    def get_action(self, obs):
        super().get_action(obs)
        obs = self.preprocess(obs)
        self.obs4step = np.roll(self.obs4step, 1, axis=0)
        self.obs4step[0] = obs
        action = self.agent.act(self.obs4step)

        return action

    def episode_begin(self, obs):
        super().episode_begin(obs)
        self.obs4step = np.zeros((4, 80, 80))
        self.obs4step[:] = self.preprocess(obs)

    def episode_end(self):
        super().episode_end()


def play_game(env, player, times):
    """
    ゲームを実行する関数
    :param env: 環境
    :param player:プレイヤー(HumanまたはAI)
    :param times: 回数
    :return:
    """
    clock = pygame.time.Clock()

    for i in range(times):
        done = False
        obs = env.reset()
        player.episode_begin(obs)
        while not done:
            clock.tick(60)  # 60fpsで動かす
            pygame.display.update()

            action = player.get_action(obs)

            obs, reward, done, _ = env.step(action)

            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        sys.exit()

        player.episode_end()
    player.play_end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("player", help="Human or AI")
    parser.add_argument("--times", help="回数", default=1, type=int)
    args = parser.parse_args()
    times = args.times

    if args.player == "Human":
        # 人間のプレイ
        env = v0PlayT2Env()
        player = Human(Logger)
        play_game(env, player, times)
    elif args.player == "AI":
        # AIのプレイ
        env = v0PlayT2Env()
        player = AI(make_agent(), preprocess, Logger)
        play_game(env, player, times)
