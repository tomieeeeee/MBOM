from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
import numpy as np
#import env_wapper.mpe.make_env as make_env
import make_env as make_env
#import line_profiler as lp

def simulate():
    for i in range(100):
        while True:
            env.render()
            random_action = [[0.1, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 0.1, 0], [0, 0, 0, 0, 0.1]]
            actions = []
            #random_action = [0, 0]
            #random_action[0] = [np.array([1, 0, 0, 0, 0]), np.array([0, 0.05, 0.00, 0, 0])]
            #random_action[1] = [np.array([0, 0.05, 0, 0, 0]), np.array([0, 0.05, 0.00, 0, 0])]
            #actions = [random_action[np.random.randint(1, 5)], random_action[np.random.randint(1, 5)]]
            #3v3#actions = [random_action[np.random.randint(1, 5)], random_action[np.random.randint(1, 5)],random_action[np.random.randint(1, 5)],random_action[np.random.randint(1, 5)], random_action[np.random.randint(1, 5)],random_action[np.random.randint(1, 5)]]
            for i in range(12):
                actions.append(random_action[np.random.randint(1, 5)])
            ##actions = [np.eye(5)[env.action_space[0].sample()] for a_s in env.action_space]
            #print(actions)
            s_, r, d, _ = env.step(actions)
            print(s_)
            #s_, r, d, _ = env.step([np.array([1,0,0.01,0,0.01]),np.array([0,0,0,0,0])])
            #s_, r, d, _ = env.step([env.action_space[0].sample(),env.action_space[0].sample()])
            # 空   右  左   上
            #print(random_action)
            #print(r)
            s = s_
            #print("状态",s)
            if d[0]:
                break


if __name__ == "__main__":
  #graphviz = GraphvizOutput()
  #graphviz.output_file = 'bbasic.png'
  #with PyCallGraph(output=graphviz):
    env = make_env.make_env('simple_tag_v6', is_contain_done=False)
    s = env.reset()
    print("zhuangtai",s)
    #simulate()
    #lp = lp()
    #lp_wrapper = lp(simulate)
    #lp_wrapper(numbers)
    #lp.print_stats()
    #print(s)

    


