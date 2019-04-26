import numpy as np
import pygame

import avoid_game_env
from gym_avoid_game.envs.resource.scene import Scene
from gym_avoid_game.envs.resource.task import task1, task2
from gym_avoid_game.envs.resource import task_manager
from gym_avoid_game.envs.resource.config import WINDOW_HEIGHT, WINDOW_WIDTH


class PlayEnv():
    def __init__(self):
        self.screen = None
        self.task_manager = None
        self.done = None

    def step(self, a, x_axis_only):
        reward = 1

        result = self.task_manager.update(a, x_axis_only)
        self.screen.fill((255, 255, 255))
        self.task_manager.draw(self.screen)
        if result == Scene.QUIT:
            self.done = True
            reward = 0

        rgb_array = pygame.surfarray.array3d(self.screen)

        return np.asarray(rgb_array, dtype=np.uint8), reward, self.done, {}

    def reset(self):
        pass

    def close(self):
        pygame.quit()


class PlayT1Env(PlayEnv):
    def __init__(self):
        super().__init__()

    def step(self, a, x_axis_only):
        return super().step(a, x_axis_only)

    def reset(self):
        super().reset()
        self.done = False
        pygame.init()
        self.task_manager = task_manager.TaskManager(task1.Task1)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("shooting_env")

        self.screen.fill((255, 255, 255))
        self.task_manager.draw(self.screen)
        rgb_array = pygame.surfarray.array3d(self.screen)

        return np.asarray(rgb_array, dtype=np.uint8)

    def close(self):
        super().close()


class PlayT2Env(PlayEnv):
    def __init__(self):
        super().__init__()

    def step(self, a, x_axis_only):
        return super().step(a, x_axis_only)

    def reset(self):
        super().reset()
        self.done = False
        pygame.init()
        self.task_manager = task_manager.TaskManager(task2.Task2)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("shooting_env")

        self.screen.fill((255, 255, 255))
        self.task_manager.draw(self.screen)
        rgb_array = pygame.surfarray.array3d(self.screen)

        return np.asarray(rgb_array, dtype=np.uint8)

    def close(self):
        super().close()


class v0PlayT1Env(PlayT1Env):
    def step(self, a):
        return super().step(a, False)


class v1PlayT1Env(PlayT1Env):
    def step(self, a):
        return super().step(a, True)


class v0PlayT2Env(PlayT2Env):
    def step(self, a):
        return super().step(a, False)


class v1PlayT2Env(PlayT2Env):
    def step(self, a):
        return super().step(a, True)