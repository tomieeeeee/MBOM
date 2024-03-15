from DRL_MARL_homework.MBAM.env_wapper.mpe.make_env import make_env
import numpy as np


class Complex_RPS(object):
    def __init__(self, eps_max_step = 100):
        super(Complex_RPS, self).__init__()
        self.n_agent = 2
        self.n_state = 14  #以自己为中心，超出4格以外则结束  [0, 1, 2, 3 ,4 ,5, 6, 7, 8]
        self.n_action = 9
        self.n_opponent_action = 9

        self.actions_trans = np.array([[0.3, 0, 0, 0, 0],   [0, 0.3, 0, 0, 0],      [0, 0, 0.3, 0, 0],
                                       [0, 0, 0, 0.3, 0],   [0, 0, 0, 0, 0.3],      [0, 0.3, 0, 0.3, 0],
                                       [0, 0.3, 0, 0, 0.3], [0, 0, 0.3, 0.3, 0],    [0, 0, 0.3, 0, 0.3]])
        self.env_state_running = False
        self.eps_max_step = eps_max_step
        self.cur_step = 0
        self.env = make_env('rps')

    def reset(self):
        self.cur_step = 0
        self.env_state_running = True
        return self.env.reset()

    def step(self, actions):
        self.cur_step += 1
        assert self.env_state_running == True, "Env is stoped, please reset()"
        a = self.actions_trans[actions[0]]
        oppo_a = self.actions_trans[actions[1]]
        obs_, rew, _, info = self.env.step([a, oppo_a])
        done = False
        if self.cur_step == self.eps_max_step:
            done = True
            self.env_state_running = False
        return obs_, rew, done, info

    def render(self):
        self.env.render()

if __name__ == "__main__":
    env = Complex_RPS()
    for _ in range(100000):
        obs = env.reset()
        while True:
            env.render()
            action = np.random.randint(0, 9)
            oppo_action = np.random.randint(0, 9)
            actions = np.hstack([action, oppo_action])
            obs, rew, done, _ = env.step(actions)
            #print(obs)
            print(rew)
            if done:
                break