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
