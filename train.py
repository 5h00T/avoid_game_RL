import time

import gym
import gym_avoid_game
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from PIL import Image
from scipy import misc
import sys
import copy


class QFunction(chainer.Chain):

    def __init__(self, n_actions, n_hidden_channels=100):
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


def preprocess(observation):
    img = Image.fromarray(observation).convert("L")
    return np.asarray(img)


env = gym.make("avoid_game_t2-v1")
n_actions = env.action_space.n
q_func = QFunction(n_actions)
# q_func.to_gpu()

optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(q_func)

gamma = 0.9

explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.15, random_action_func=env.action_space.sample, )

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 5)


def dqn_phi(screens):
    assert len(screens) == 4
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values = copy.deepcopy(raw_values) / 255
    return raw_values


agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=50000, update_interval=5,
    target_update_interval=10000, phi=dqn_phi)

steps = []
n_episodes = 100000
start = time.time()
total_steps = 0
for i in range(1, n_episodes + 1):
    obs_4steps = np.zeros((4, 80, 80), dtype=np.float32)
    obs = env.reset()
    step = 0
    reward = 0
    done = False
    obs = preprocess(obs)
    obs = misc.imresize(obs, (80, 80))
    obs_4steps[:] = obs

    while not done:
        step += 1

        env.render()

        action = agent.act_and_train(obs_4steps, reward)
        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess(next_obs)
        next_obs = misc.imresize(next_obs, (80, 80))
        obs_4steps = np.roll(obs_4steps, 1, axis=0)
        obs_4steps[0] = next_obs

    steps.append(step)
    sys.stdout.write('\r episode: {} step: {} statistics: {}'.format(i, step, agent.get_statistics()))
    sys.stdout.flush()
    agent.stop_episode_and_train(obs_4steps, reward, done)

print('Finished, elapsed time : {}'.format(time.time() - start))
