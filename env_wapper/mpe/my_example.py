import numpy as np
import env_wapper.mpe.make_env as make_env


if __name__ == "__main__":
    env = make_env.make_env('rps')
    s = env.reset()
    #print(s)
    for i in range(100):
        while True:
            env.render()
            random_action = [[0.1, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 0.1, 0], [0, 0, 0, 0, 0.1]]
            #random_action = [0, 0]
            #random_action[0] = [np.array([1, 0, 0, 0, 0]), np.array([0, 0.05, 0.00, 0, 0])]
            #random_action[1] = [np.array([0, 0.05, 0, 0, 0]), np.array([0, 0.05, 0.00, 0, 0])]
            #actions = [random_action[np.random.randint(1, 5)], random_action[np.random.randint(1, 5)]]
            actions = [random_action[np.random.randint(1, 5)], [0, 0, 0, 0, 0]]
            #print(actions)
            s_, r, d, _ = env.step(actions)
            #s_, r, d, _ = env.step([np.array([1,0,0.01,0,0.01]),np.array([0,0,0,0,0])])
            #s_, r, d, _ = env.step([env.action_space[0].sample(),env.action_space[0].sample()])
            # 空   右  左   上
            #print(random_action)
            print(r)
            s = s_
            #print(s)
            if d[0]:
                break
