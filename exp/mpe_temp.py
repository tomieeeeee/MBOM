from env_wapper.mpe.make_env import make_env
import numpy as np


if __name__ == "__main__":
    env = make_env('simple_tag_v2', is_contain_done=False)


    for i in range(100):
        step = 0
        s = env.reset()
        while True:
            env.render()
            actions = [np.eye(5)[env.action_space[0].sample()] for a_s in env.action_space]
            s_, r, d, _ = env.step(actions)
            step += 1
            for j in range(4):
            #    print(s_[j][:2])
            #    print(s_[j][-2:])
                print(r)
            if np.any(d) == True or step > 100:
                break