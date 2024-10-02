
import sys
sys.path.append("D:/document/3v3")
from env_wapper.simple_tag.simple_tag import Simple_Tag
import time
import random


if __name__ == '__main__':


    env = Simple_Tag()
    score=[]
    score1=[]

    for i in range(100):
        dis_oppo = 0
        dis_me = 0
        temp=[]
        temp1=[]
        step = 0
        s = env.reset()
        while True:
            time.sleep(0.2)
            env.render("mode=rgb_array")
            action = []
            oppo_a = []
            a = []
            for r in range(6):
                action.append(random.randint(0,4))

            for group in range(10):
                oppo_a.append(action)
                a.append(action)

            actions = [oppo_a,a]

            s_, rew, done, _ = env.step(actions)
            step+=1
            s = s_
            if done:
                break
